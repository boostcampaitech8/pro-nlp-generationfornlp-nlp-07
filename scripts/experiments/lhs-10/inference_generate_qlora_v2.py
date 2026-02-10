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

from peft import PeftModel


# -----------------------
# Env
# -----------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/data/ephemeral/hf_cache")

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-32B")
ADAPTER_MODEL = os.environ.get("ADAPTER_MODEL", "NLP-07-ODQA/qwen3-32b-qlora-v1")

TEST_PATH  = os.environ.get("TEST_PATH", "data/test.csv")
OUT_PATH   = os.environ.get("OUT_PATH", "submission_number_generate_v4_qlora.csv")

# v4 gating thresholds (same as your v4)
MARGIN_THRESHOLD   = float(os.environ.get("MARGIN_THRESHOLD", "1.4"))
ENTROPY_THRESHOLD  = float(os.environ.get("ENTROPY_THRESHOLD", "1.10"))  # 5-class max ln(5)=1.609
TOP1P_THRESHOLD    = float(os.environ.get("TOP1P_THRESHOLD", "0.60"))

# Adaptive SC settings (same structure as your v4)
SC_SAMPLES_BASE     = int(os.environ.get("SC_SAMPLES_BASE", "5"))
SC_TEMPERATURE_BASE = float(os.environ.get("SC_TEMPERATURE_BASE", "0.8"))
SC_SAMPLES_HARD     = int(os.environ.get("SC_SAMPLES_HARD", "7"))
SC_TEMPERATURE_HARD = float(os.environ.get("SC_TEMPERATURE_HARD", "0.95"))
HARD_MARGIN         = float(os.environ.get("HARD_MARGIN", "0.7"))  # margin below this => very ambiguous

MAX_SC_RATE        = float(os.environ.get("MAX_SC_RATE", "0.70"))
SC_ACCEPT_MARGIN = float(os.environ.get("SC_ACCEPT_MARGIN", "1.0"))
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
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return {}
    try:
        return ast.literal_eval(x)
    except Exception:
        return {}

def get_single_token_id(tokenizer, s: str):
    ids = tokenizer.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None

def build_prompt(tokenizer, paragraph, question, question_plus, choices, allow_labels):
    # prompt: keep v4/v3 style (single digit only)
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
    for k in [1, 2, 3, 4, 5]:
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
def compute_uncertainty(model, inputs, label_to_ids, allow_labels):
    """
    allow_labels(1~4 or 1~5)에서:
    - label logits (각 label은 여러 토큰 후보 중 max logit)
    - margin = top1 - top2
    - top1_prob = softmax over labels의 top1 확률
    - entropy = softmax over labels의 엔트로피
    """
    out = model(**inputs, use_cache=True)
    logits = out.logits[0, -1]  # [vocab]

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

    scores_tensor = torch.tensor([s for _, s in label_scores], dtype=torch.float32, device=logits.device)
    probs = torch.softmax(scores_tensor, dim=0)
    top1_prob = probs[0].item()

    entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())
    return best_label, margin, top1_prob, entropy

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
    if margin < HARD_MARGIN:
        return SC_SAMPLES_HARD, SC_TEMPERATURE_HARD
    return SC_SAMPLES_BASE, SC_TEMPERATURE_BASE

@torch.inference_mode()
def self_consistency_logprob_sum(
    model, tokenizer, inputs,
    logits_processor, id_to_label, allow_labels,
    margin
):
    """
    SC를 logprob 합산으로 결정 + (best-second) gap도 함께 반환.
    - 반환: (best_label, gap)
      - gap = best_logsum - second_logsum
    """
    n_samples, temp = pick_sc_params(margin)

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

        if label is None:
            continue
        try:
            if int(label) not in allow_labels:
                continue
        except Exception:
            continue

        logprob = float(torch.log_softmax(scores, dim=-1)[new_token_id].item())
        label_logsum[label] += logprob
        any_valid = True

    if not any_valid:
        tid, _dec, _lp = greedy_one_token(model, tokenizer, inputs, logits_processor)
        fallback = id_to_label.get(tid, str(allow_labels[0]))
        return fallback, 0.0

    # best vs second gap 계산
    items = sorted(label_logsum.items(), key=lambda x: x[1], reverse=True)
    best_label, best_sum = items[0]
    second_sum = items[1][1] if len(items) > 1 else float("-inf")
    gap = float(best_sum - second_sum)

    return best_label, gap

# -----------------------
# Load model (QLoRA)
# -----------------------
def load_model():
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required to load org/private adapter.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=HF_TOKEN)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_MODEL, token=HF_TOKEN)
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

    last_margins = []
    last_entropies = []
    last_top1ps = []

    start = time.time()
    pbar = tqdm(range(total), desc="Number-Generate-v4(QLoRA-3gate)", dynamic_ncols=True)

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

        allowed_ids = allowed_digit_token_ids(label_to_ids, allow_labels)
        logits_processor = LogitsProcessorList([RestrictTokensLogitsProcessor(allowed_ids)])

        # --- uncertainty from logits ---
        best_label_logit, margin, top1p, ent = compute_uncertainty(
            model, inputs, label_to_ids, allow_labels
        )

        last_margins.append(margin)
        last_entropies.append(ent)
        last_top1ps.append(top1p)
        if len(last_margins) > 50:
            last_margins.pop(0); last_entropies.pop(0); last_top1ps.pop(0)

        # --- base prediction: greedy 1 token ---
        tid, _dec, _lp = greedy_one_token(model, tokenizer, inputs, logits_processor)
        pred = id_to_label.get(tid, best_label_logit)
        if int(pred) not in allow_labels:
            pred = str(allow_labels[0])

        sc_label, sc_gap = None, 0.0
        
        # --- 3-threshold gating (same as your v4) ---
        need_sc = (
            (margin < MARGIN_THRESHOLD) or
            (ent > ENTROPY_THRESHOLD) or
            (top1p < TOP1P_THRESHOLD)
        )

        if need_sc and used_sc < sc_budget:
            pred = self_consistency_logprob_sum(
                model, tokenizer, inputs,
                logits_processor, id_to_label, allow_labels,
                margin
            )
            used_sc += 1
            if sc_label is not None and sc_label != pred:
                if sc_gap >= SC_ACCEPT_MARGIN:
                    pred = sc_label
                else:
                    pred = sc_label or pred
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
    print("\n[Models]")
    print(f"- BASE_MODEL={BASE_MODEL}")
    print(f"- ADAPTER_MODEL={ADAPTER_MODEL}")
    print("\n[Thresholds]")
    print(f"- MARGIN_THRESHOLD={MARGIN_THRESHOLD}")
    print(f"- ENTROPY_THRESHOLD={ENTROPY_THRESHOLD}")
    print(f"- TOP1P_THRESHOLD={TOP1P_THRESHOLD}")

if __name__ == "__main__":
    main()
