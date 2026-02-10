import os
import re
import ast
import time
import math
from collections import Counter, defaultdict, deque

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
# Env / Config
# -----------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/data/ephemeral/hf_cache")

MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Qwen3-30B-A3B-bnb-4bit")
TEST_PATH  = os.environ.get("TEST_PATH", "data/test.csv")
OUT_PATH   = os.environ.get("OUT_PATH", "submission_number_generate_v5.csv")

SEED = int(os.environ.get("SEED", "42"))

# Prompt ensemble
PROMPT_ENSEMBLE = int(os.environ.get("PROMPT_ENSEMBLE", "2"))  # 1 or 2

# Uncertainty thresholds (performance-first; tune)
MARGIN_THRESHOLD = float(os.environ.get("MARGIN_THRESHOLD", "6.0"))   # logit margin(top1-top2)
P1_THRESHOLD     = float(os.environ.get("P1_THRESHOLD", "0.90"))      # top1 prob among labels
H_THRESHOLD      = float(os.environ.get("H_THRESHOLD", "0.08"))       # normalized entropy (0~1)

# Adaptive SC (v4-style)
MAX_SC_RATE         = float(os.environ.get("MAX_SC_RATE", "0.70"))    # cap SC usage ratio
HARD_MARGIN         = float(os.environ.get("HARD_MARGIN", "3.0"))      # very ambiguous threshold

SC_SAMPLES_BASE     = int(os.environ.get("SC_SAMPLES_BASE", "9"))
SC_TEMPERATURE_BASE = float(os.environ.get("SC_TEMPERATURE_BASE", "0.8"))
SC_TOP_P_BASE       = float(os.environ.get("SC_TOP_P_BASE", "0.90"))

SC_SAMPLES_HARD     = int(os.environ.get("SC_SAMPLES_HARD", "13"))
SC_TEMPERATURE_HARD = float(os.environ.get("SC_TEMPERATURE_HARD", "0.95"))
SC_TOP_P_HARD       = float(os.environ.get("SC_TOP_P_HARD", "0.90"))

FORCE_SC = int(os.environ.get("FORCE_SC", "0"))  # 1 => always SC

# Logging window
WIN = int(os.environ.get("STAT_WINDOW", "50"))

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

def parse_digit(text, allow_labels):
    m = re.search(r"([1-5])", text)
    if not m:
        return None
    v = int(m.group(1))
    return str(v) if v in allow_labels else None

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def entropy_norm(ps):
    n = len(ps)
    if n <= 1:
        return 0.0
    h = 0.0
    for p in ps:
        if p > 0:
            h -= p * math.log(p + 1e-12)
    return h / math.log(n)

def logsumexp(vals):
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals) + 1e-12)


# -----------------------
# Prompt
# -----------------------
def build_prompt(tokenizer, paragraph, question, question_plus, choices, allow_labels, variant: int = 0):
    qplus = "" if (question_plus is None or (isinstance(question_plus, float) and pd.isna(question_plus))) else str(question_plus)
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])

    if variant == 0:
        sys = "너는 수능 객관식 채점기다. 출력 규칙을 절대 위반하지 마라."
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
    else:
        sys = "너는 한국 수능형 객관식 문제를 푸는 모델이다. 숫자 1개만 출력하라."
        user = (
            "다음 객관식 문제의 정답 번호만 출력하라.\n"
            "풀이/설명은 절대 쓰지 말고, 마지막 출력은 숫자 하나만.\n"
            "조건(시기/인물/개념/원인-결과)으로 오답을 제거하라.\n"
            f"출력 가능한 번호는 {', '.join(map(str, allow_labels))} 뿐이다.\n\n"
            f"[지문]\n{paragraph}\n\n"
            f"[질문]\n{question}{qplus}\n\n"
            f"[선택지]\n{choices_str}\n\n"
            "정답:"
        )

    messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# -----------------------
# Token restriction
# -----------------------
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

def allowed_digit_token_ids(tokenizer, allow_labels):
    ids = set()
    for k in allow_labels:
        for form in [str(k), " " + str(k)]:
            tid = get_single_token_id(tokenizer, form)
            if tid is not None:
                ids.add(tid)
    return sorted(ids)

def label_scores_from_logits(tokenizer, logits_last, allow_labels):
    labels = [str(k) for k in allow_labels]
    scores = []
    for k in allow_labels:
        vals = []
        tid1 = get_single_token_id(tokenizer, str(k))
        tid2 = get_single_token_id(tokenizer, " " + str(k))
        if tid1 is not None: vals.append(float(logits_last[tid1].item()))
        if tid2 is not None: vals.append(float(logits_last[tid2].item()))
        if not vals:
            vals = [-1e9]
        scores.append(logsumexp(vals))

    probs = softmax(scores)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    i1 = order[0]
    i2 = order[1] if len(order) > 1 else order[0]
    top = labels[i1]
    margin = scores[i1] - scores[i2]
    p1 = probs[i1]
    H = entropy_norm(probs)
    return labels, scores, probs, top, margin, p1, H


# -----------------------
# Model load
# -----------------------
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


# -----------------------
# Inference core
# -----------------------
@torch.inference_mode()
def forward_label_dist(model, tokenizer, prompt, allow_labels):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs)
    logits_last = out.logits[0, -1]
    return label_scores_from_logits(tokenizer, logits_last, allow_labels)

def aggregate_ensemble(dists):
    labels = dists[0][0]
    sum_scores = [0.0 for _ in labels]
    for (_, scores, _, _, _, _, _) in dists:
        for i in range(len(labels)):
            sum_scores[i] += scores[i]
    probs = softmax(sum_scores)
    order = sorted(range(len(sum_scores)), key=lambda i: sum_scores[i], reverse=True)
    i1, i2 = order[0], (order[1] if len(order) > 1 else order[0])
    top = labels[i1]
    margin = sum_scores[i1] - sum_scores[i2]
    p1 = probs[i1]
    H = entropy_norm(probs)
    return labels, sum_scores, probs, top, margin, p1, H

def pick_sc_params(margin):
    if margin < HARD_MARGIN:
        return SC_SAMPLES_HARD, SC_TEMPERATURE_HARD, SC_TOP_P_HARD
    return SC_SAMPLES_BASE, SC_TEMPERATURE_BASE, SC_TOP_P_BASE

def should_use_sc(margin, p1, H, used_sc, sc_budget):
    if FORCE_SC == 1:
        return used_sc < sc_budget
    return (margin < MARGIN_THRESHOLD) or (p1 < P1_THRESHOLD) or (H > H_THRESHOLD)

@torch.inference_mode()
def sc_logprob_sum(model, tokenizer, prompt, allow_labels, temperature, top_p):
    """
    v4-style: sample 1 token multiple times, accumulate label logprob sum, pick max.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    allowed_ids = allowed_digit_token_ids(tokenizer, allow_labels)
    lp = LogitsProcessorList([RestrictTokensLogitsProcessor(allowed_ids)])

    label_logsum = defaultdict(float)
    any_valid = False

    # We'll use model.generate with output_scores to compute logprob of sampled token
    for _ in range(SC_SAMPLES):
        gen = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            logits_processor=lp,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        scores = gen.scores[0][0]  # [vocab]
        new_token_id = int(gen.sequences[0, inputs["input_ids"].shape[1]].item())
        decoded = tokenizer.decode([new_token_id], skip_special_tokens=True)
        pred = parse_digit(decoded, allow_labels)
        if pred is None:
            continue
        logprob = float(torch.log_softmax(scores, dim=-1)[new_token_id].item())
        label_logsum[pred] += logprob
        any_valid = True

    if not any_valid:
        return None

    return max(label_logsum.items(), key=lambda x: x[1])[0]


# -----------------------
# Main
# -----------------------
def main():
    global SC_SAMPLES  # we will set per item by pick_sc_params
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

    total = len(df)
    sc_budget = int(total * MAX_SC_RATE)
    used_sc = 0

    win_mrg = deque(maxlen=WIN)
    win_H   = deque(maxlen=WIN)
    win_p1  = deque(maxlen=WIN)

    start = time.time()
    pbar = tqdm(range(total), desc="Number-Generate-v5(GEN)", dynamic_ncols=True)

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

        if len(choices) <= 4:
            choices = choices[:4]
            allow_labels = [1,2,3,4]
        else:
            choices = choices[:5]
            allow_labels = [1,2,3,4,5]

        prompts = [build_prompt(tokenizer, paragraph, question, question_plus, choices, allow_labels, variant=0)]
        if PROMPT_ENSEMBLE >= 2:
            prompts.append(build_prompt(tokenizer, paragraph, question, question_plus, choices, allow_labels, variant=1))

        dists = [forward_label_dist(model, tokenizer, p, allow_labels) for p in prompts]
        labels, scores, probs, top, margin, p1, H = aggregate_ensemble(dists)

        win_mrg.append(margin)
        win_H.append(H)
        win_p1.append(p1)

        pred = top

        # SC decision
        if used_sc < sc_budget and should_use_sc(margin, p1, H, used_sc, sc_budget):
            n_samples, temp, top_p = pick_sc_params(margin)

            # set global for loop inside sc_logprob_sum
            SC_SAMPLES = n_samples

            # run SC on each prompt and then combine by another logprob-sum across prompts
            # (more stable than voting)
            total_logsum = defaultdict(float)
            any_sc = False

            for p in prompts:
                # sc_logprob_sum returns label chosen by logprob-sum, but we want to ADD logprob sums.
                # To keep it simple and robust: run sc_logprob_sum and treat it as a "vote" with weight.
                sc_label = sc_logprob_sum(model, tokenizer, p, allow_labels, temperature=temp, top_p=top_p)
                if sc_label is None:
                    continue
                total_logsum[sc_label] += 1.0
                any_sc = True

            if any_sc:
                # pick the most frequent among prompt-variants (2 max)
                pred = max(total_logsum.items(), key=lambda x: x[1])[0]
                used_sc += 1

        answer_counter[pred] += 1
        results.append({"id": _id, "answer": pred})

        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
        remaining = (total - (i + 1)) / rate if rate > 0 else float("inf")

        mrg50 = sum(win_mrg) / max(1, len(win_mrg))
        H50   = sum(win_H) / max(1, len(win_H))
        p150  = sum(win_p1) / max(1, len(win_p1))

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
    print("\n[v5 params]")
    print(f"- PROMPT_ENSEMBLE={PROMPT_ENSEMBLE}")
    print(f"- MARGIN_THRESHOLD={MARGIN_THRESHOLD}, P1_THRESHOLD={P1_THRESHOLD}, H_THRESHOLD={H_THRESHOLD}")
    print(f"- HARD_MARGIN={HARD_MARGIN}")
    print(f"- SC_BASE(samples/temp/top_p)=({SC_SAMPLES_BASE}/{SC_TEMPERATURE_BASE}/{SC_TOP_P_BASE})")
    print(f"- SC_HARD(samples/temp/top_p)=({SC_SAMPLES_HARD}/{SC_TEMPERATURE_HARD}/{SC_TOP_P_HARD})")
    print(f"- MAX_SC_RATE={MAX_SC_RATE}, FORCE_SC={FORCE_SC}")
    print(f"- SEED={SEED}")


if __name__ == "__main__":
    main()
