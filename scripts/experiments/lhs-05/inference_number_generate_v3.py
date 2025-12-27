import os
import re
import ast
import time
from collections import Counter

import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

# -----------------------
# Env
# -----------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/data/ephemeral/hf_cache")

MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Qwen3-30B-A3B-bnb-4bit")
TEST_PATH  = os.environ.get("TEST_PATH", "data/test.csv")
OUT_PATH   = os.environ.get("OUT_PATH", "submission_number_generate_v2_generate.csv")

# v2 옵션(환경변수로 조정)
# - margin이 낮을수록(애매할수록) self-consistency 실행
MARGIN_THRESHOLD = float(os.environ.get("MARGIN_THRESHOLD", "1.4"))
# - 애매 문항에서 샘플 몇 번 할지
SC_SAMPLES = int(os.environ.get("SC_SAMPLES", "5"))
# - 샘플링 온도
SC_TEMPERATURE = float(os.environ.get("SC_TEMPERATURE", "0.8"))
# - 애매 문항 비율이 너무 커지는 것을 방지(상한)
MAX_SC_RATE = float(os.environ.get("MAX_SC_RATE", "0.70"))  # 전체의 최대 25%까지만 SC 실행
# - seed 고정(재현성)
SEED = int(os.environ.get("SEED", "42"))

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------
# Utils
# -----------------------
def safe_parse_problems(x):
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    try:
        return ast.literal_eval(x)
    except Exception:
        return {}

def get_single_token_id(tokenizer, s: str):
    ids = tokenizer.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None

def build_prompt(tokenizer, paragraph, question, question_plus, choices, allow_labels):
    """
    v2: 더 간결하게. (작년 상위팀 결론: 간결 프롬프트가 유리)
    """
    qplus = "" if (question_plus is None or (isinstance(question_plus, float) and pd.isna(question_plus))) else str(question_plus)
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])

    user = (
        "정답 번호 숫자 1개만 출력하라. (공백/줄바꿈/설명 금지)\n"
        f"허용 번호: {', '.join(map(str, allow_labels))}\n\n"
        f"[지문]\n{paragraph}\n\n"
        f"[질문]\n{question}{qplus}\n\n"
        f"[보기]\n{choices_str}\n\n"
        "정답:"
    )

    messages = [
        {"role": "system", "content": "너는 객관식 채점기다. 출력 규칙을 절대 위반하지 마라."},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

class RestrictTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        super().__init__()
        self.allowed = set(int(x) for x in allowed_token_ids if x is not None)

    def __call__(self, input_ids, scores):
        if len(self.allowed) == 0:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        mask[:, list(self.allowed)] = 0.0
        return scores + mask

def allowed_digit_token_ids(tokenizer, allow_labels):
    ids = set()
    for k in allow_labels:
        for form in [str(k), " " + str(k)]:
            tid = get_single_token_id(tokenizer, form)
            if tid is not None:
                ids.add(tid)
    return sorted(ids)

def parse_digit(text, allow_labels):
    m = re.search(r"([1-5])", text)
    if not m:
        return None
    v = int(m.group(1))
    return str(v) if v in allow_labels else None

@torch.inference_mode()
def next_token_scores(model, tokenizer, inputs, allow_labels):
    """
    생성 전에 forward 한 번으로:
    - allow_labels 내에서 best/second logits
    - margin 계산
    """
    out = model(**inputs)
    logits = out.logits[0, -1]  # [vocab]

    cand = []
    for k in allow_labels:
        scores = []
        tid1 = get_single_token_id(tokenizer, str(k))
        tid2 = get_single_token_id(tokenizer, " " + str(k))
        if tid1 is not None:
            scores.append(logits[tid1].item())
        if tid2 is not None:
            scores.append(logits[tid2].item())
        if scores:
            cand.append((k, max(scores)))
        else:
            cand.append((k, float("-inf")))

    cand.sort(key=lambda x: x[1], reverse=True)
    best_k, best_s = cand[0]
    second_s = cand[1][1] if len(cand) > 1 else float("-inf")
    margin = best_s - second_s
    return str(best_k), margin

@torch.inference_mode()
def generate_one_token(model, tokenizer, inputs, allow_labels, do_sample=False, temperature=1.0):
    """
    1토큰 생성(허용 숫자만) -> 숫자 파싱
    """
    allowed_ids = allowed_digit_token_ids(tokenizer, allow_labels)
    logits_processor = LogitsProcessorList([RestrictTokensLogitsProcessor(allowed_ids)])

    gen = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=do_sample,
        temperature=temperature,
        top_p=1.0,
        logits_processor=logits_processor,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = gen[0][inputs["input_ids"].shape[1]:]
    out_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    pred = parse_digit(out_text, allow_labels)
    return pred, out_text

@torch.inference_mode()
def predict_with_sc(model, tokenizer, prompt, allow_labels, sc_samples, sc_temperature):
    """
    애매한 문항에서만 self-consistency:
    - 허용 숫자 1토큰 샘플링을 N번
    - 다수결 (동률이면 가장 먼저 나온 값)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    votes = []
    for _ in range(sc_samples):
        pred, _raw = generate_one_token(
            model, tokenizer, inputs, allow_labels,
            do_sample=True, temperature=sc_temperature
        )
        if pred is not None:
            votes.append(pred)

    if not votes:
        # 실패하면 greedy로
        pred, _ = generate_one_token(model, tokenizer, inputs, allow_labels, do_sample=False, temperature=1.0)
        return pred if pred is not None else str(allow_labels[0])

    c = Counter(votes)
    # 최빈값
    best = c.most_common(1)[0][0]
    return best

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer

def main():
    model, tokenizer = load_model()
    df = pd.read_csv(TEST_PATH)

    if "problems" in df.columns:
        probs = df["problems"].apply(safe_parse_problems)
        df["question"] = probs.apply(lambda d: d.get("question", ""))
        df["choices"]  = probs.apply(lambda d: d.get("choices", []))
    if "question_plus" not in df.columns:
        df["question_plus"] = ""

    answer_counter = Counter()
    results = []

    # stats
    used_sc = 0
    total = len(df)
    sc_budget = int(total * MAX_SC_RATE)

    start = time.time()
    pbar = tqdm(range(len(df)), desc="Number-Generate-v2(GEN)", dynamic_ncols=True)

    for i in pbar:
        row = df.iloc[i]
        _id = row["id"] if "id" in row else i

        paragraph = row.get("paragraph", "")
        question = row.get("question", "")
        qplus = row.get("question_plus", "")
        choices = row.get("choices", [])

        if isinstance(choices, str):
            try:
                choices = ast.literal_eval(choices)
            except Exception:
                choices = []

        if not isinstance(choices, list) or len(choices) == 0:
            pred = "1"
            answer_counter[pred] += 1
            results.append({"id": _id, "answer": pred})
            continue

        if len(choices) <= 4:
            choices = choices[:4]
            allow_labels = [1,2,3,4]
        else:
            choices = choices[:5]
            allow_labels = [1,2,3,4,5]

        prompt = build_prompt(tokenizer, paragraph, question, qplus, choices, allow_labels)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 1) forward로 margin 산출 (애매함 판단)
        greedy_from_logits, margin = next_token_scores(model, tokenizer, inputs, allow_labels)

        # 2) 기본은 greedy 1토큰 생성
        pred, _raw = generate_one_token(model, tokenizer, inputs, allow_labels, do_sample=False, temperature=1.0)
        if pred is None:
            pred = greedy_from_logits  # fallback

        # 3) 애매한 문제만 self-consistency
        if used_sc < sc_budget and margin < MARGIN_THRESHOLD:
            pred = predict_with_sc(
                model, tokenizer, prompt, allow_labels,
                sc_samples=SC_SAMPLES,
                sc_temperature=SC_TEMPERATURE
            )
            used_sc += 1

        answer_counter[pred] += 1
        results.append({"id": _id, "answer": pred})

        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
        remaining = (len(df) - (i + 1)) / rate if rate > 0 else float("inf")

        pbar.set_postfix({
            "it/s": f"{rate:.2f}",
            "eta(min)": f"{remaining/60:.1f}" if remaining != float("inf") else "inf",
            "sc_used": used_sc,
            "cnt1": answer_counter.get("1", 0),
            "cnt2": answer_counter.get("2", 0),
            "cnt3": answer_counter.get("3", 0),
            "cnt4": answer_counter.get("4", 0),
            "cnt5": answer_counter.get("5", 0),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_PATH, index=False)

    print("\n[Done]")
    print(f"- saved: {OUT_PATH}")
    print(f"- answer counts: {dict(answer_counter)}")
    print(f"- self-consistency used: {used_sc} / {total} ({used_sc/total*100:.1f}%)")

if __name__ == "__main__":
    main()
