import os
import ast
import time
from collections import Counter

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -----------------------
# Env
# -----------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/data/ephemeral/hf_cache")

MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Qwen3-30B-A3B-bnb-4bit")
TEST_PATH  = os.environ.get("TEST_PATH", "data/test.csv")
OUT_PATH   = os.environ.get("OUT_PATH", "submission_choice_scoring.csv")

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

def pick_marker_tokens(tokenizer):
    """
    'O'/'X'가 1토큰이 아닐 수도 있어 대비.
    가능한 1토큰 쌍을 자동으로 선택.
    """
    candidates = [
        (" O", " X"),
        ("O", "X"),
        (" T", " F"),
        ("T", "F"),
        (" Y", " N"),
        ("Y", "N"),
    ]
    for pos, neg in candidates:
        pos_id = get_single_token_id(tokenizer, pos)
        neg_id = get_single_token_id(tokenizer, neg)
        if pos_id is not None and neg_id is not None:
            return (pos, neg, pos_id, neg_id)
    # 최후 fallback: pos만이라도 1토큰인 걸 찾고, neg는 무시(점수는 pos만 사용)
    for pos in [" O", "O", " T", "T", " Y", "Y"]:
        pos_id = get_single_token_id(tokenizer, pos)
        if pos_id is not None:
            return (pos, None, pos_id, None)
    raise RuntimeError("No single-token marker found. Try different markers or tokenizer.")

def build_choice_scoring_prompts(tokenizer, paragraph, question, question_plus, choices):
    """
    선택지별로 '정답이면 O, 아니면 X 하나만' 출력하도록 강하게 제한.
    Qwen3: enable_thinking=False로 <think> 차단.
    """
    qplus = "" if (question_plus is None or (isinstance(question_plus, float) and pd.isna(question_plus))) else str(question_plus)

    base = (
        "다음은 객관식 독해 문제다.\n"
        "규칙:\n"
        "1) 너의 출력은 반드시 한 글자만: O 또는 X\n"
        "2) 설명/해설/공백/줄바꿈/추가 문자는 절대 출력하지 마라.\n"
        "3) 선택지가 정답이면 O, 정답이 아니면 X.\n\n"
        f"[지문]\n{paragraph}\n\n"
        f"[질문]\n{question}{qplus}\n\n"
    )

    prompts = []
    for idx, ch in enumerate(choices, start=1):
        user = (
            base
            + f"[검증할 선택지 {idx}]\n{ch}\n\n"
            + "이 선택지가 정답인가?\n"
            + "출력:"
        )
        messages = [
            {"role": "system",
             "content": "너는 채점기다. 사용자가 요구한 출력 규칙을 절대 위반하지 마라."},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompts.append(prompt)
    return prompts

@torch.inference_mode()
def score_choices_O_logit(model, tokenizer, prompts, pos_token_id):
    """
    각 prompt에 대해 다음 토큰 logits에서 pos_token_id(O)의 점수만 뽑음.
    (생성 X → 안정/속도)
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    out = model(**inputs)
    # out.logits: [B, T, V]
    logits_last = out.logits[:, -1, :]         # [B, V]
    scores = logits_last[:, pos_token_id]      # [B]
    return scores.detach().float().cpu().tolist()

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
        low_cpu_mem_usage=True
    )
    model.eval()
    return model, tokenizer

def main():
    model, tokenizer = load_model()
    pos_s, neg_s, pos_id, neg_id = pick_marker_tokens(tokenizer)
    print(f"[Marker tokens] pos={repr(pos_s)} (id={pos_id}), neg={repr(neg_s)}")

    df = pd.read_csv(TEST_PATH)

    # problems 파싱 (baseline 데이터 구조)
    if "problems" in df.columns:
        probs = df["problems"].apply(safe_parse_problems)
        df["question"] = probs.apply(lambda d: d.get("question", ""))
        df["choices"]  = probs.apply(lambda d: d.get("choices", []))

    if "question_plus" not in df.columns:
        df["question_plus"] = ""

    answer_counter = Counter()
    results = []

    start = time.time()
    pbar = tqdm(range(len(df)), desc="Choice-Scoring", dynamic_ncols=True)

    for i in pbar:
        row = df.iloc[i]
        _id = row["id"] if "id" in row else i

        paragraph = row.get("paragraph", "")
        question = row.get("question", "")
        question_plus = row.get("question_plus", "")
        choices = row.get("choices", [])

        # 혹시 choices가 문자열이면 파싱
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
        if len(choices) == 4:
            choices = choices[:4]
        else:
            choices = choices[:5]

        prompts = build_choice_scoring_prompts(tokenizer, paragraph, question, question_plus, choices)
        scores = score_choices_O_logit(model, tokenizer, prompts, pos_id)

        best_idx = max(range(len(scores)), key=lambda k: scores[k])  # 0-based
        pred = str(best_idx + 1)

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
