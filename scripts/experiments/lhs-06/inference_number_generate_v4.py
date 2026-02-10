import os
import re
import ast
import time
import math
from collections import Counter, defaultdict

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
OUT_PATH   = os.environ.get("OUT_PATH", "submission_number_generate_v4.csv")

# v4 gating thresholds (performance-first defaults; tune)
MARGIN_THRESHOLD   = float(os.environ.get("MARGIN_THRESHOLD", "1.4"))
ENTROPY_THRESHOLD  = float(os.environ.get("ENTROPY_THRESHOLD", "1.10"))  # 5-class max ln(5)=1.609
TOP1P_THRESHOLD    = float(os.environ.get("TOP1P_THRESHOLD", "0.60"))

# Adaptive SC settings
SC_SAMPLES_BASE    = int(os.environ.get("SC_SAMPLES_BASE", "5"))
SC_TEMPERATURE_BASE= float(os.environ.get("SC_TEMPERATURE_BASE", "0.8"))
SC_SAMPLES_HARD    = int(os.environ.get("SC_SAMPLES_HARD", "7"))
SC_TEMPERATURE_HARD= float(os.environ.get("SC_TEMPERATURE_HARD", "0.95"))
HARD_MARGIN        = float(os.environ.get("HARD_MARGIN", "0.7"))  # margin below this => "very ambiguous"

MAX_SC_RATE        = float(os.environ.get("MAX_SC_RATE", "0.70"))
SEED               = int(os.environ.get("SEED", "42"))

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_grad_enabled(False)

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
    # 프롬프트는 v3에서 유지(요청사항)
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
        if not self.allowed:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        mask[:, list(self.allowed)] = 0.0
        return scores + mask

def build_digit_id_maps(tokenizer):
    """
    - label -> token_ids (가능한 '1' / ' 1' 등)
    - token_id -> label (샘플링 결과를 라벨로 매핑)
    """
    label_to_ids = {}
    id_to_label = {}
    for k in [1,2,3,4,5]:
        ids = []
        for form in [str(k), " " + str(k)]:
            tid = get_single_token_id(tokenizer, form)
            if tid is not None:
                ids.append(tid)
                id_to_label[tid] = str(k)
        label_to_ids[str(k)] = sorted(set(ids))
    return label_to_ids, id_to_label

def allowed_digit_token_ids(label_to_ids, allow_labels):
    ids = set()
    for k in allow_labels:
        for tid in label_to_ids[str(k)]:
            ids.add(tid)
    return sorted(ids)

@torch.inference_mode()
def compute_uncertainty(model, tokenizer, inputs, label_to_ids, allow_labels):
    """
    allow_labels(1~4 or 1~5)에서:
    - label logits (각 label은 여러 토큰 후보 중 max logit)
    - margin = top1 - top2
    - top1_prob = softmax over labels의 top1 확률
    - entropy = softmax over labels의 엔트로피
    """
    out = model(**inputs, use_cache=True)
    logits = out.logits[0, -1]  # [vocab]

    # label score: max over token variants
    label_scores = []
    for k in allow_labels:
        ids = label_to_ids[str(k)]
        if not ids:
            s = float("-inf")
        else:
            s = max(logits[tid].item() for tid in ids)
        label_scores.append((str(k), s))

    label_scores.sort(key=lambda x: x[1], reverse=True)
    best_label, best_logit = label_scores[0]
    second_logit = label_scores[1][1] if len(label_scores) > 1 else float("-inf")
    margin = best_logit - second_logit

    # softmax over labels
    scores_tensor = torch.tensor([s for _, s in label_scores], dtype=torch.float32, device=logits.device)
    probs = torch.softmax(scores_tensor, dim=0)
    top1_prob = probs[0].item()

    # entropy
    # avoid log(0)
    entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())

    return best_label, margin, top1_prob, entropy

@torch.inference_mode()
def generate_one_token_with_logprob(model, tokenizer, inputs, logits_processor):
    """
    1토큰 생성 + 해당 토큰의 logprob 반환.
    return: (token_id, decoded_text, logprob)
    """
    gen = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=True,               # 샘플링 여부는 호출 측에서 logits_processor로 제한
        temperature=1.0,              # 호출 측에서 scores를 직접 온도조절하는 대신 generate 인자 사용
        top_p=1.0,
        logits_processor=logits_processor,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    # gen.scores: 길이 1 리스트, 각 원소 [batch, vocab] (pre-softmax logits)
    scores = gen.scores[0][0]  # [vocab]
    # 생성된 새 토큰 id
    new_token_id = int(gen.sequences[0, inputs["input_ids"].shape[1]].item())

    # logprob
    log_probs = torch.log_softmax(scores, dim=-1)
    logprob = float(log_probs[new_token_id].item())

    decoded = tokenizer.decode([new_token_id], skip_special_tokens=True)
    return new_token_id, decoded, logprob

@torch.inference_mode()
def greedy_one_token(model, tokenizer, inputs, logits_processor):
    """
    greedy 1토큰 생성(허용 숫자만) + logprob
    """
    gen = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        temperature=1.0,
        logits_processor=logits_processor,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    scores = gen.scores[0][0]
    new_token_id = int(gen.sequences[0, inputs["input_ids"].shape[1]].item())
    log_probs = torch.log_softmax(scores, dim=-1)
    logprob = float(log_probs[new_token_id].item())
    decoded = tokenizer.decode([new_token_id], skip_special_tokens=True)
    return new_token_id, decoded, logprob

def pick_sc_params(margin):
    """
    margin이 매우 작으면 더 강하게 탐색.
    """
    if margin < HARD_MARGIN:
        return SC_SAMPLES_HARD, SC_TEMPERATURE_HARD
    return SC_SAMPLES_BASE, SC_TEMPERATURE_BASE

@torch.inference_mode()
def self_consistency_logprob_sum(
    model, tokenizer, prompt, inputs,
    logits_processor, id_to_label, allow_labels,
    margin
):
    """
    SC를 logprob 합산으로 결정:
    - 샘플 N회
    - 각 라벨별 logprob 합계를 누적
    - 합계가 최대인 라벨 선택
    """
    n_samples, temp = pick_sc_params(margin)

    # temperature를 generate 인자로 쓰려면, 호출별로 넣어야 해서 여기서는 logits_processor는 동일,
    # generate 인자 do_sample=True, temperature=temp 로 호출
    # => generate_one_token_with_logprob를 temp 반영 버전으로 호출
    label_logsum = defaultdict(float)
    any_valid = False

    for _ in range(n_samples):
        gen = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=temp,
            top_p=1.0,
            logits_processor=logits_processor,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        scores = gen.scores[0][0]
        new_token_id = int(gen.sequences[0, inputs["input_ids"].shape[1]].item())
        label = id_to_label.get(new_token_id, None)

        if label is None or int(label) not in allow_labels:
            continue

        logprob = float(torch.log_softmax(scores, dim=-1)[new_token_id].item())
        label_logsum[label] += logprob
        any_valid = True

    if not any_valid:
        # fallback: greedy
        tid, _dec, _lp = greedy_one_token(model, tokenizer, inputs, logits_processor)
        return id_to_label.get(tid, str(allow_labels[0]))

    # pick max logprob-sum
    best_label = max(label_logsum.items(), key=lambda x: x[1])[0]
    return best_label

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
    label_to_ids, id_to_label = build_digit_id_maps(tokenizer)

    df = pd.read_csv(TEST_PATH)
    if "problems" in df.columns:
        probs = df["problems"].apply(safe_parse_problems)
        df["question"] = probs.apply(lambda d: d.get("question", ""))
        df["choices"]  = probs.apply(lambda d: d.get("choices", []))
    if "question_plus" not in df.columns:
        df["question_plus"] = ""

    answer_counter = Counter()
    results = []

    total = len(df)
    sc_budget = int(total * MAX_SC_RATE)
    used_sc = 0

    # monitor
    last_margins = []
    last_entropies = []
    last_top1ps = []

    start = time.time()
    pbar = tqdm(range(total), desc="Number-Generate-v4(GEN)", dynamic_ncols=True)

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

        # allowed token restriction (only digit tokens)
        allowed_ids = allowed_digit_token_ids(label_to_ids, allow_labels)
        logits_processor = LogitsProcessorList([RestrictTokensLogitsProcessor(allowed_ids)])

        # uncertainty metrics from logits (no generate yet)
        best_label_logit, margin, top1p, entropy = compute_uncertainty(
            model, tokenizer, inputs, label_to_ids, allow_labels
        )
        last_margins.append(margin)
        last_entropies.append(entropy)
        last_top1ps.append(top1p)
        if len(last_margins) > 50:
            last_margins.pop(0); last_entropies.pop(0); last_top1ps.pop(0)

        # base prediction: greedy generate 1 token (still "generation-based")
        tid, dec, lp = greedy_one_token(model, tokenizer, inputs, logits_processor)
        pred = id_to_label.get(tid, best_label_logit)
        if int(pred) not in allow_labels:
            pred = str(allow_labels[0])

        # gating for SC (performance-first)
        need_sc = (
            (margin < MARGIN_THRESHOLD) or
            (entropy > ENTROPY_THRESHOLD) or
            (top1p < TOP1P_THRESHOLD)
        )

        if need_sc and used_sc < sc_budget:
            pred = self_consistency_logprob_sum(
                model, tokenizer, prompt, inputs,
                logits_processor, id_to_label, allow_labels,
                margin
            )
            used_sc += 1

        answer_counter[pred] += 1
        results.append({"id": _id, "answer": pred})

        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
        remaining = (total - (i + 1)) / rate if rate > 0 else float("inf")

        mrg50 = sum(last_margins) / max(1, len(last_margins))
        H50 = sum(last_entropies) / max(1, len(last_entropies))
        p150 = sum(last_top1ps) / max(1, len(last_top1ps))

        pbar.set_postfix({
            "it/s": f"{rate:.2f}",
            "eta(min)": f"{remaining/60:.1f}" if remaining != float("inf") else "inf",
            "sc_used": used_sc,
            "mrg50": f"{mrg50:.2f}",
            "H50": f"{H50:.2f}",
            "p150": f"{p150:.2f}",
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
