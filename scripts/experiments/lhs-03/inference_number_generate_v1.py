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
OUT_PATH   = os.environ.get("OUT_PATH", "submission_number_generate.csv")

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
    정답은 반드시 "숫자 1개"만 출력하도록 강제.
    Qwen3: enable_thinking=False로 <think> 차단.
    """
    qplus = "" if (question_plus is None or (isinstance(question_plus, float) and pd.isna(question_plus))) else str(question_plus)
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])

    user = (
        "아래 지문과 문항을 읽고 보기 중 정답 번호만 출력하시오.\n"
        "규칙:\n"
        "1) 출력은 반드시 정답 번호 숫자 1개만 (예: 1)\n"
        "2) 공백/줄바꿈/설명/해설/문장/기호 추가 출력 금지\n"
        f"3) 허용 정답 번호: {', '.join(map(str, allow_labels))}\n\n"
        f"[지문]\n{paragraph}\n\n"
        f"[질문]\n{question}{qplus}\n\n"
        f"[보기]\n{choices_str}\n\n"
        "정답 번호:"
    )

    messages = [
        {"role": "system", "content": "너는 객관식 채점기다. 사용자가 요구한 출력 규칙을 절대 위반하지 마라."},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return prompt

class RestrictTokensLogitsProcessor(LogitsProcessor):
    """
    생성 토큰을 허용 목록(allowed_token_ids)으로만 제한.
    (숫자 1개만 뱉도록 강제할 때 매우 효과적)
    """
    def __init__(self, allowed_token_ids):
        super().__init__()
        self.allowed = set(int(x) for x in allowed_token_ids if x is not None)

    def __call__(self, input_ids, scores):
        # scores: [batch, vocab]
        if len(self.allowed) == 0:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        mask[:, list(self.allowed)] = 0.0
        return scores + mask

def allowed_digit_token_ids(tokenizer, allow_labels):
    """
    숫자 토큰화가 '1' 또는 ' 1' 등으로 갈릴 수 있어 둘 다 허용.
    또한 가끔 '\n1' 같은 형태는 1토큰이 아닐 가능성이 높으니 제외.
    """
    ids = set()
    for k in allow_labels:
        for form in [str(k), " " + str(k)]:
            tid = get_single_token_id(tokenizer, form)
            if tid is not None:
                ids.add(tid)
    return sorted(ids)

def parse_digit(text, allow_labels):
    """
    디코딩 결과에서 허용 라벨(1~5) 중 첫 숫자만 추출.
    """
    m = re.search(r"([1-5])", text)
    if not m:
        return None
    v = int(m.group(1))
    return str(v) if v in allow_labels else None

@torch.inference_mode()
def predict_generate_one_token(model, tokenizer, prompt, allow_labels):
    """
    - generate를 사용하되, logits processor로 허용 숫자 토큰만 생성 가능하게 제한
    - max_new_tokens=1로 1토큰만 생성
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    allowed_ids = allowed_digit_token_ids(tokenizer, allow_labels)
    logits_processor = LogitsProcessorList([RestrictTokensLogitsProcessor(allowed_ids)])

    gen = model.generate(
        **inputs,
        max_new_tokens=1,          # 핵심: 1토큰만
        do_sample=False,
        temperature=1.0,           # do_sample=False면 영향 거의 없음
        logits_processor=logits_processor,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = gen[0][inputs["input_ids"].shape[1]:]
    out_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    pred = parse_digit(out_text, allow_labels)
    if pred is not None:
        return pred, out_text

    # ---- fallback 1: next-token logits argmax (허용라벨에서) ----
    out = model(**inputs)
    logits = out.logits[0, -1]  # [vocab]
    best = None
    best_score = None
    for k in allow_labels:
        scores = []
        tid1 = get_single_token_id(tokenizer, str(k))
        tid2 = get_single_token_id(tokenizer, " " + str(k))
        if tid1 is not None: scores.append((logits[tid1].item(), str(k)))
        if tid2 is not None: scores.append((logits[tid2].item(), str(k)))
        for s, label in scores:
            if best_score is None or s > best_score:
                best_score = s
                best = label
    if best is not None:
        return best, out_text

    # ---- fallback 2: 그냥 1 ----
    return str(allow_labels[0]), out_text

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

    # baseline 데이터 구조 대응: problems 문자열 dict
    if "problems" in df.columns:
        probs = df["problems"].apply(safe_parse_problems)
        df["question"] = probs.apply(lambda d: d.get("question", ""))
        df["choices"]  = probs.apply(lambda d: d.get("choices", []))

    if "question_plus" not in df.columns:
        df["question_plus"] = ""

    answer_counter = Counter()
    results = []

    start = time.time()
    pbar = tqdm(range(len(df)), desc="Number-Generate", dynamic_ncols=True)

    for i in pbar:
        row = df.iloc[i]
        _id = row["id"] if "id" in row else i

        paragraph = row.get("paragraph", "")
        question = row.get("question", "")
        question_plus = row.get("question_plus", "")
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

        # 4지/5지 대응
        n = len(choices)
        if n <= 4:
            choices = choices[:4]
            allow_labels = [1, 2, 3, 4]
        else:
            choices = choices[:5]
            allow_labels = [1, 2, 3, 4, 5]

        prompt = build_prompt(tokenizer, paragraph, question, question_plus, choices, allow_labels)
        pred, raw = predict_generate_one_token(model, tokenizer, prompt, allow_labels)

        answer_counter[pred] += 1
        results.append({"id": _id, "answer": pred})

        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
        remaining = (len(df) - (i + 1)) / rate if rate > 0 else float("inf")

        pbar.set_postfix({
            "it/s": f"{rate:.2f}",
            "eta(min)": f"{remaining/60:.1f}" if remaining != float("inf") else "inf",
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

if __name__ == "__main__":
    main()
