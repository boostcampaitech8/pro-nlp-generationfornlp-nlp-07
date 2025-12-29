#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference (logit classification + SC budget) prompt-aligned to train_cot.py.

Fixes v1:
- Robustly parses 'problems' column used by this competition (question/choices nested there).
- If top-level 'question'/'choices' columns are absent, we extract from problems[0].
- Keeps the previously requested prompt alignment:
  * ChatML via tokenizer.apply_chat_template on USER turn only + response_part suffix
  * choice_range formatted as "1, 2, ..., K"
  * truncation_side="left"
- Core logit classification + SC logic unchanged.
"""

import os
import json
from ast import literal_eval
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

# -----------------------------
# Env / Config
# -----------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
ADAPTER_MODEL = os.environ.get("ADAPTER_MODEL", "").strip()
TEST_PATH = os.environ.get("TEST_PATH", "data/test.csv")
OUT_PATH = os.environ.get("OUT_PATH", "submission.csv")
SEED = int(os.environ.get("SEED", "42"))

MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))

MARGIN_THRESHOLD = float(os.environ.get("MARGIN_THRESHOLD", "0.75"))
ENTROPY_THRESHOLD = float(os.environ.get("ENTROPY_THRESHOLD", "1.40"))
TOP1P_THRESHOLD = float(os.environ.get("TOP1P_THRESHOLD", "0.40"))

MAX_SC_RATE = float(os.environ.get("MAX_SC_RATE", "0.10"))
SC_TRIGGER_K = int(os.environ.get("SC_TRIGGER_K", "3"))
CONF_MARGIN = float(os.environ.get("CONF_MARGIN", "2.0"))
CONF_TOP1P = float(os.environ.get("CONF_TOP1P", "0.80"))
CONF_ENTROPY = float(os.environ.get("CONF_ENTROPY", "0.60"))
HARD_MARGIN = float(os.environ.get("HARD_MARGIN", "0.40"))

SC_SAMPLES_BASE = int(os.environ.get("SC_SAMPLES_BASE", "9"))
SC_TEMPERATURE_BASE = float(os.environ.get("SC_TEMPERATURE_BASE", "0.65"))
SC_SAMPLES_HARD = int(os.environ.get("SC_SAMPLES_HARD", "13"))
SC_TEMPERATURE_HARD = float(os.environ.get("SC_TEMPERATURE_HARD", "0.80"))

SC_ACCEPT_MARGIN = float(os.environ.get("SC_ACCEPT_MARGIN", "1.0"))
SC_ACCEPT_MARGIN_STRICT = float(os.environ.get("SC_ACCEPT_MARGIN_STRICT", "1.8"))

FIRST_SAMPLE_DEBUG = int(os.environ.get("FIRST_SAMPLE_DEBUG", "0")) == 1
SC_DEBUG = int(os.environ.get("SC_DEBUG", "0")) == 1
SC_DEBUG_MAX = int(os.environ.get("SC_DEBUG_MAX", "200"))

# -----------------------------
# Chat template (train_cot.py aligned)
# -----------------------------
CHAT_TEMPLATE_CONFIG: Dict[str, Any] = {
    "default_chat_template": """{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}""",
    "response_part": "<|im_start|>assistant\n",
}

def _auto_response_part(tokenizer: AutoTokenizer) -> str:
    base = tokenizer.apply_chat_template(
        [{"role": "user", "content": "X"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    with_prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "X"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if with_prefix.startswith(base):
        rp = with_prefix[len(base):]
        return rp if rp else CHAT_TEMPLATE_CONFIG["response_part"]
    return CHAT_TEMPLATE_CONFIG["response_part"]

def apply_chat_template_safe(tokenizer: AutoTokenizer, user_content: str) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=False,
    )
    text += CHAT_TEMPLATE_CONFIG["response_part"]
    return text

# -----------------------------
# Helpers
# -----------------------------
def _safe_isnan(x: Any) -> bool:
    try:
        return isinstance(x, float) and np.isnan(x)
    except Exception:
        return False

def parse_problems_cell(v: Any) -> Optional[Any]:
    """Parse the 'problems' cell into Python object (list/dict), robustly."""
    if v is None or _safe_isnan(v):
        return None
    if isinstance(v, (list, dict)):
        return v
    if not isinstance(v, str):
        return None

    s = v.strip()
    if not s:
        return None

    # Sometimes it's double-encoded JSON: '"[...]"'
    for _ in range(2):
        try:
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                obj = json.loads(s)
            else:
                obj = literal_eval(s)
            # If decoding yields a string that looks like JSON, loop once more
            if isinstance(obj, str):
                s2 = obj.strip()
                if (s2.startswith("{") and s2.endswith("}")) or (s2.startswith("[") and s2.endswith("]")):
                    s = s2
                    continue
            return obj
        except Exception:
            # try alternate parse
            try:
                obj = literal_eval(s)
                return obj
            except Exception:
                return None
    return None

def extract_from_problems(problems_obj: Any) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Extract (question, choices) from problems object.
    Handles:
    - dict with keys: question/choices or query/options/candidates
    - list[dict] with above keys in first element
    """
    if problems_obj is None:
        return None, None

    def _get_qc(d: Dict[str, Any]) -> Tuple[Optional[str], Optional[List[str]]]:
        q = d.get("question") or d.get("query") or d.get("q") or d.get("stem")
        c = d.get("choices") or d.get("options") or d.get("candidates") or d.get("answer_choices")
        if isinstance(q, (int, float)):
            q = str(q)
        if isinstance(q, str):
            q = q.strip()
        else:
            q = None

        if isinstance(c, str):
            try:
                c = literal_eval(c)
            except Exception:
                try:
                    c = json.loads(c)
                except Exception:
                    c = None
        if isinstance(c, list):
            out = [str(x).strip() for x in c if str(x).strip()]
            c = out if out else None
        else:
            c = None
        return q, c

    if isinstance(problems_obj, dict):
        return _get_qc(problems_obj)
    if isinstance(problems_obj, list) and len(problems_obj) > 0 and isinstance(problems_obj[0], dict):
        return _get_qc(problems_obj[0])

    return None, None

def extract_choices_fallback(row: Dict[str, Any]) -> List[str]:
    """Fallback extraction if problems doesn't contain choices."""
    # direct "choices"/"options"
    for key in ("choices", "options"):
        v = row.get(key)
        if v is None or _safe_isnan(v):
            continue
        try:
            if isinstance(v, str):
                vv = v.strip()
                if vv.startswith("[") and vv.endswith("]"):
                    v = literal_eval(vv)
            if isinstance(v, list) and v:
                out = [str(x).strip() for x in v if str(x).strip()]
                if out:
                    return out
        except Exception:
            pass

    # A..E columns
    cols = ["A", "B", "C", "D", "E"]
    if all((c in row and row[c] is not None and not _safe_isnan(row[c]) and str(row[c]).strip()) for c in cols):
        return [str(row[c]).strip() for c in cols]

    # numbered choice columns
    for prefix in ("choice", "option", "choice_", "option_"):
        out = []
        for i in range(1, 6):
            k = f"{prefix}{i}" if not prefix.endswith("_") else f"{prefix}{i}"
            if k in row:
                v = row.get(k)
                if v is None or _safe_isnan(v):
                    continue
                s = str(v).strip()
                if s:
                    out.append(s)
        if out:
            return out

    return []

# -----------------------------
# Prompt builder (train_cot aligned)
# -----------------------------
def build_prompt(paragraph: str, question: str, question_plus: Optional[str], choices: List[str]) -> str:
    qps = f"\n<보기>:\n{question_plus}" if question_plus else ""
    ch = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    nums = ", ".join(str(i) for i in range(1, len(choices) + 1))
    return (
        "지문을 읽고 질문의 답을 구하세요.\n\n"
        f"지문:\n{paragraph}\n\n"
        f"질문:\n{question}{qps}\n\n"
        f"선택지:\n{ch}\n\n"
        f"{nums} 중에 하나를 정답으로 고르세요.\n"
        "정답:"
    )

# -----------------------------
# Logit classification + SC
# -----------------------------
@torch.inference_mode()
def logits_label_stats(
    model, tokenizer: AutoTokenizer, prompt: str, allow_labels: List[int]
) -> Tuple[str, float, float, float]:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    ).to(model.device)

    out = model(**inputs)
    last_logits = out.logits[0, -1, :]

    ids = [tokenizer.encode(str(l), add_special_tokens=False)[-1] for l in allow_labels]
    sub = last_logits[torch.tensor(ids, device=last_logits.device)]
    probs = torch.softmax(sub, dim=-1)

    if sub.numel() >= 2:
        top2 = torch.topk(sub, k=2).values
        margin = float((top2[0] - top2[1]).item())
    else:
        margin = float("inf")

    entropy = float((-probs * torch.log(probs + 1e-12)).sum().item())
    top1p = float(probs.max().item())

    pred_idx = int(torch.argmax(sub).item())
    pred_label = str(allow_labels[pred_idx])
    return pred_label, margin, entropy, top1p

class RestrictTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allow_token_ids: set[int], mask_value: float = -1e9):
        self.allow_token_ids = allow_token_ids
        self.mask_value = mask_value

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, self.mask_value)
        allow = torch.tensor(list(self.allow_token_ids), device=scores.device, dtype=torch.long)
        mask[:, allow] = 0.0
        return scores + mask

@torch.inference_mode()
def self_consistency_logprob_sum(
    model, tokenizer: AutoTokenizer, prompt: str,
    allow_labels: List[int], n_samples: int, temperature: float
) -> Tuple[str, float]:
    allow_token_ids = set(tokenizer.encode(str(l), add_special_tokens=False)[-1] for l in allow_labels)
    lp = LogitsProcessorList([RestrictTokensLogitsProcessor(allow_token_ids)])

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    ).to(model.device)

    label_scores = defaultdict(float)
    id_to_label = {tokenizer.encode(str(l), add_special_tokens=False)[-1]: str(l) for l in allow_labels}

    for _ in range(n_samples):
        gen = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=temperature,
            logits_processor=lp,
            return_dict_in_generate=True,
            output_scores=True,
        )
        scores = gen.scores[0][0]
        probs = torch.softmax(scores, dim=-1)
        for tid, lab in id_to_label.items():
            p = float(probs[tid].item())
            label_scores[lab] += float(np.log(max(p, 1e-12)))

    items = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = items[0]
    second_score = items[1][1] if len(items) > 1 else -1e30
    gap = float(best_score - second_score)
    return best_label, gap

# -----------------------------
# Main
# -----------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(TEST_PATH)
    if "id" not in df.columns:
        if "sample_id" in df.columns:
            df = df.rename(columns={"sample_id": "id"})
        else:
            df.insert(0, "id", range(len(df)))

    allow_labels = [1, 2, 3, 4, 5]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    tokenizer.chat_template = CHAT_TEMPLATE_CONFIG["default_chat_template"]
    CHAT_TEMPLATE_CONFIG["response_part"] = _auto_response_part(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    if ADAPTER_MODEL:
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ADAPTER_MODEL)
            model.eval()
        except Exception as e:
            print(f"[WARN] Failed to load adapter '{ADAPTER_MODEL}': {e}")

    sc_budget = max(1, int(round(len(df) * MAX_SC_RATE)))
    used_sc = 0
    sc_accept = 0
    cnt = {str(i): 0 for i in allow_labels}
    rows_out = []

    pbar = tqdm(total=len(df), desc="Batch Inference (logit+SC)", ncols=120)
    for idx in range(len(df)):
        row = df.iloc[idx].to_dict()

        paragraph = str(row.get("paragraph", "") or row.get("context", "") or "")
        question_plus = row.get("question_plus", None)
        if question_plus is not None and _safe_isnan(question_plus):
            question_plus = None
        if question_plus is not None:
            question_plus = str(question_plus)

        # Competition format: question/choices live under "problems"
        problems_obj = parse_problems_cell(row.get("problems"))
        q_from_prob, c_from_prob = extract_from_problems(problems_obj)

        question = str(row.get("question", "") or "")
        if not question.strip() and q_from_prob:
            question = q_from_prob

        choices = c_from_prob if (c_from_prob and len(c_from_prob) >= 2) else extract_choices_fallback(row)

        if not choices:
            # print the raw problems cell for debugging
            raw = row.get("problems")
            raise KeyError(
                f"Could not extract choices for row idx={idx} (id={row.get('id')}). "
                f"Columns={list(df.columns)}. "
                f"problems_type={type(raw)} problems_head={str(raw)[:200]}"
            )

        user_prompt = build_prompt(paragraph, question, question_plus, choices)
        prompt = apply_chat_template_safe(tokenizer, user_prompt)

        if FIRST_SAMPLE_DEBUG and idx == 0:
            tok_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            print("[DEBUG] columns:", list(df.columns))
            print("[DEBUG] prompt_token_len:", tok_len)
            print("[DEBUG] response_part(repr):", repr(CHAT_TEMPLATE_CONFIG["response_part"]))
            print("[DEBUG] question(extracted):", question[:120])
            print("[DEBUG] choices_len:", len(choices))
            print("[DEBUG] user_tail:", user_prompt[-160:].replace("\n", "\\n"))

        pred_base, margin, ent, top1p = logits_label_stats(model, tokenizer, prompt, allow_labels)
        pred = pred_base

        need_sc = (margin < MARGIN_THRESHOLD) or (ent > ENTROPY_THRESHOLD) or (top1p < TOP1P_THRESHOLD)
        if (margin >= CONF_MARGIN) and (top1p >= CONF_TOP1P) and (ent <= CONF_ENTROPY):
            need_sc = False

        accepted = False
        if need_sc and used_sc < sc_budget:
            if margin < HARD_MARGIN:
                n_samples = SC_SAMPLES_HARD
                temp = SC_TEMPERATURE_HARD
                accept_margin = SC_ACCEPT_MARGIN_STRICT
            else:
                n_samples = SC_SAMPLES_BASE
                temp = SC_TEMPERATURE_BASE
                accept_margin = SC_ACCEPT_MARGIN

            sc_label, sc_gap = self_consistency_logprob_sum(
                model, tokenizer, prompt, allow_labels, n_samples=n_samples, temperature=temp
            )
            used_sc += 1

            if sc_gap >= accept_margin:
                pred = sc_label
                accepted = True
                sc_accept += 1

            if SC_DEBUG and used_sc <= SC_DEBUG_MAX:
                print(f"[SC] idx={idx} id={row.get('id')} base={pred_base} "
                      f"m={margin:.3f} H={ent:.3f} p={top1p:.3f} "
                      f"sc={sc_label} gap={sc_gap:.3f} accept={int(accepted)}")

        pred = str(pred)
        if pred not in [str(x) for x in allow_labels]:
            pred = "1"

        cnt[pred] += 1
        rows_out.append({"id": row.get("id"), "answer": int(pred)})

        pbar.set_postfix({
            "pred": pred,
            "sc": f"{used_sc}/{sc_budget}",
            "acc": f"{(sc_accept/used_sc):.2f}" if used_sc > 0 else "n/a",
        })
        pbar.update(1)

    pbar.close()
    pd.DataFrame(rows_out).to_csv(OUT_PATH, index=False)

    print(f"[DONE] wrote: {OUT_PATH}")
    if used_sc > 0:
        print(f"[SC] used={used_sc}/{sc_budget} accept={sc_accept} accept_rate={(sc_accept/used_sc):.3f}")
    else:
        print(f"[SC] used=0/{sc_budget} accept=0 accept_rate=0.000")
    print("[CNT] " + " ".join([f"{k}:{v}" for k, v in cnt.items()]))

if __name__ == "__main__":
    main()
