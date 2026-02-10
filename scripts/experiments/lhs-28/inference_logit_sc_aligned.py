#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aligned inference (logit + SC) for your train_cot ChatML format.

Key points:
- Prompt is built as *messages* and rendered via tokenizer.apply_chat_template(..., add_generation_prompt=True)
- ChatML structure 유지:
    <|im_start|>user ... <|im_end|>
    <|im_start|>assistant ...
- truncation_side="left" so that tail (choices + "정답:") is preserved when max length is exceeded
- Existing logit-based classification + SC logic 유지

Env vars:
  BASE_MODEL, ADAPTER_MODEL (optional)
  TEST_PATH, OUT_PATH
  SEED
  MAX_SEQ_LENGTH (default 4096)
  MAX_SC_RATE
  Gate thresholds: MARGIN_THRESHOLD, ENTROPY_THRESHOLD, TOP1P_THRESHOLD
  v3 guard: SC_TRIGGER_K, CONF_MARGIN, CONF_TOP1P, CONF_ENTROPY, HARD_MARGIN
  SC params: SC_SAMPLES_BASE, SC_TEMPERATURE_BASE, SC_SAMPLES_HARD, SC_TEMPERATURE_HARD
  SC accept: SC_ACCEPT_MARGIN, SC_ACCEPT_MARGIN_STRICT
  Debug:
    FIRST_SAMPLE_DEBUG=1
    SC_DEBUG=1
    SC_DEBUG_MAX=200
"""

import os
import math
import random
from ast import literal_eval
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except Exception:
    _HAS_UNSLOTH = False

from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
from peft import PeftModel




def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RestrictTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allow_token_ids: set[int], mask_value: float = -1e9):
        self.allow_token_ids = allow_token_ids
        self.mask_value = mask_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, self.mask_value)
        idx = torch.tensor(list(self.allow_token_ids), device=scores.device, dtype=torch.long)
        mask[:, idx] = 0.0
        return scores + mask


def extract_choices(row: dict, p0: dict) -> list[str]:
    # Most reliable: inside problems JSON
    ch = p0.get("choices")
    if isinstance(ch, list) and len(ch) > 0:
        return ch
    # Some variants might have top-level column "choices"
    ch2 = row.get("choices")
    if isinstance(ch2, str):
        try:
            parsed = literal_eval(ch2)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except Exception:
            pass
    if isinstance(ch2, list) and len(ch2) > 0:
        return ch2
    return []


def build_user_content(paragraph: str, question: str, question_plus: str | None, choices: list[str]) -> str:
    qps = f"\n<보기>:\n{question_plus}" if question_plus else ""
    ch = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    nums = ", ".join(str(i) for i in range(1, len(choices) + 1)) # 수정 ver
    return (
        "지문을 읽고 질문의 답을 구하세요.\n\n"
        f"지문:\n{paragraph}\n\n"
        f"질문:\n{question}{qps}\n\n"
        f"선택지:\n{ch}\n\n"
        f"{nums} 중에 하나를 정답으로 고르세요.\n"
        "정답:"
    )


def build_chatml_prompt(tokenizer, paragraph: str, question: str, question_plus: str | None, choices: list[str]) -> str:
    messages = [
        {"role": "user", "content": build_user_content(paragraph, question, question_plus, choices)},
    ]
    # add_generation_prompt=True => "<|im_start|>assistant\n" 까지 붙여줌 (모델이 바로 다음 토큰으로 답을 내도록)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _label_token_candidates(tokenizer, label: int) -> list[int]:
    """
    Some tokenizers score '1' differently vs ' 1' depending on preceding whitespace/newline.
    We take a small candidate set and later use max logit over candidates.
    """
    s = str(label)
    cands = []
    for variant in (s, " " + s, "\n" + s):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            cands.append(ids[-1])
    # unique preserve order
    seen = set()
    out = []
    for i in cands:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


@torch.inference_mode()
def logits_label_stats(model, tokenizer, prompt: str, allow_labels: list[int], max_len: int, logit_temperature):
    # preserve tail (choices/정답) when truncating
    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    ).to(model.device)

    out = model(**inputs)
    scores = out.logits[0, -1]  # [vocab]

    # Build candidate token ids per label and take max score among candidates
    label_to_score = []
    for l in allow_labels:
        cand_ids = _label_token_candidates(tokenizer, l)
        if not cand_ids:
            label_to_score.append(float("-inf"))
            continue
        v = torch.tensor(cand_ids, device=scores.device, dtype=torch.long)
        label_to_score.append(float(torch.max(scores[v]).item()))
    sub = torch.tensor(label_to_score, device=scores.device)  # [K]
    # prior correction
    if 'label_prior' in globals() and label_prior is not None:
        for i, l in enumerate(allow_labels):
            p = label_prior.get(str(l))
            if p:
                sub[i] -= PRIOR_ALPHA * math.log(p)
    probs = torch.softmax(sub / max(logit_temperature, 1e-6), dim=-1)

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


@torch.inference_mode()
def self_consistency_logprob_sum(
    model, tokenizer, prompt: str, allow_labels: list[int],
    n_samples: int, temperature: float, max_len: int,
):
    # Restrict to candidate tokens for labels
    allow_token_ids = set()
    id_to_label = {}
    for l in allow_labels:
        for tid in _label_token_candidates(tokenizer, l):
            allow_token_ids.add(tid)
            id_to_label[tid] = str(l)

    lp = LogitsProcessorList([RestrictTokensLogitsProcessor(allow_token_ids)])

    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    ).to(model.device)

    label_scores = defaultdict(float)
    any_valid = False

    for _ in range(n_samples):
        gen = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            logits_processor=lp,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        scores = gen.scores[0][0]
        new_token_id = int(gen.sequences[0, inputs["input_ids"].shape[1]].item())
        label = id_to_label.get(new_token_id)
        if label is None:
            continue

        any_valid = True
        lpv = float(torch.log_softmax(scores, dim=-1)[new_token_id].item())
        label_scores[label] += lpv

    if not any_valid:
        return str(allow_labels[0]), 0.0

    items = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = items[0]
    second_score = items[1][1] if len(items) > 1 else -1e9
    gap = float(best_score - second_score)
    return best_label, gap


def main():
    base_model = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
    adapter_model = os.environ.get("ADAPTER_MODEL", "").strip()
    test_path = os.environ.get("TEST_PATH", "data/test.csv")
    out_path = os.environ.get("OUT_PATH", "submission_sc_aligned.csv")
    seed = int(os.environ.get("SEED", "42"))
    max_len = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))

    # gating thresholds
    MARGIN_THRESHOLD = float(os.environ.get("MARGIN_THRESHOLD", "0.75"))
    ENTROPY_THRESHOLD = float(os.environ.get("ENTROPY_THRESHOLD", "1.40"))
    TOP1P_THRESHOLD = float(os.environ.get("TOP1P_THRESHOLD", "0.40"))
    LOGIT_TEMPERATURE = float(os.environ.get("LOGIT_TEMPERATURE", "1.0"))


    # v3 guard
    SC_TRIGGER_K = int(os.environ.get("SC_TRIGGER_K", "2"))
    CONF_MARGIN = float(os.environ.get("CONF_MARGIN", "2.0"))
    CONF_TOP1P = float(os.environ.get("CONF_TOP1P", "0.80"))
    CONF_ENTROPY = float(os.environ.get("CONF_ENTROPY", "0.60"))
    HARD_MARGIN = float(os.environ.get("HARD_MARGIN", "0.45"))

    # SC params
    SC_SAMPLES_BASE = int(os.environ.get("SC_SAMPLES_BASE", "9"))
    SC_TEMPERATURE_BASE = float(os.environ.get("SC_TEMPERATURE_BASE", "0.65"))
    SC_SAMPLES_HARD = int(os.environ.get("SC_SAMPLES_HARD", "13"))
    SC_TEMPERATURE_HARD = float(os.environ.get("SC_TEMPERATURE_HARD", "0.80"))

    # SC accept
    SC_ACCEPT_MARGIN = float(os.environ.get("SC_ACCEPT_MARGIN", "1.0"))
    SC_ACCEPT_MARGIN_STRICT = float(os.environ.get("SC_ACCEPT_MARGIN_STRICT", "1.6"))

    # budget
    MAX_SC_RATE = float(os.environ.get("MAX_SC_RATE", "0.10"))
    # prior correction (Macro F1 friendly)
    PRIOR_ALPHA = float(os.environ.get("PRIOR_ALPHA", "0.0"))
    PRIOR_TRAIN_PATH = os.environ.get("PRIOR_TRAIN_PATH", "").strip()


    # debug
    FIRST_SAMPLE_DEBUG = int(os.environ.get("FIRST_SAMPLE_DEBUG", "1"))
    SC_DEBUG = int(os.environ.get("SC_DEBUG", "0"))
    SC_DEBUG_MAX = int(os.environ.get("SC_DEBUG_MAX", "200"))

    set_seed(seed)

    # load model/tokenizer
    if _HAS_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_len,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # Ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_model:
        model = PeftModel.from_pretrained(model, adapter_model)

    model.eval()

    df = pd.read_csv(test_path)

    # ---- build label prior (train set) ----
    label_prior = None
    if PRIOR_ALPHA > 0 and PRIOR_TRAIN_PATH:
        try:
            tdf = pd.read_csv(PRIOR_TRAIN_PATH)
            counts = {}
            for r in tdf.to_dict(orient="records"):
                probs = r.get("problems")
                if isinstance(probs, str):
                    p0 = literal_eval(probs)
                    p0 = p0[0] if isinstance(p0, list) else p0
                    ans = str(p0.get("answer", "")).strip()
                    if ans.isdigit():
                        counts[ans] = counts.get(ans, 0) + 1
            total = sum(counts.values())
            label_prior = {k: v / total for k, v in counts.items() if v > 0}
            print("[PRIOR] loaded:", label_prior)
        except Exception as e:
            print("[PRIOR] failed to load:", e)
            label_prior = None

    total = len(df)
    sc_budget = int(math.floor(total * MAX_SC_RATE))
    used_sc = 0
    accept_sc = 0
    cnt = defaultdict(int)

    out_rows = []

    pbar = tqdm(
        df.to_dict(orient="records"),
        total=total,
        desc="Batch Inference (logit+SC)",
        dynamic_ncols=True,
    )

    for i, row in enumerate(pbar, start=1):
        qid = str(row["id"])
        paragraph = row["paragraph"]

        problems = row.get("problems")
        if isinstance(problems, str):
            p0 = literal_eval(problems)
            p0 = p0[0] if isinstance(p0, list) else p0
        else:
            # extremely defensive
            out_rows.append({"id": qid, "answer": "1"})
            cnt["1"] += 1
            continue

        question = p0["question"]
        question_plus = row.get("question_plus")
        if isinstance(question_plus, float) and np.isnan(question_plus):
            question_plus = None
        if question_plus is None:
            question_plus = p0.get("question_plus")

        choices = extract_choices(row, p0)
        if not choices:
            out_rows.append({"id": qid, "answer": "1"})
            cnt["1"] += 1
            continue

        allow_labels = list(range(1, len(choices) + 1))
        prompt = build_chatml_prompt(tokenizer, paragraph, question, question_plus, choices)

        pred_base, margin, ent, top1p = logits_label_stats(model, tokenizer, prompt, allow_labels, max_len, LOGIT_TEMPERATURE)
        pred = pred_base

        # gating
        t_margin = (margin < MARGIN_THRESHOLD)
        t_ent = (ent > ENTROPY_THRESHOLD)
        t_top1p = (top1p < TOP1P_THRESHOLD)
        triggers = int(t_margin) + int(t_ent) + int(t_top1p)

        base_confident = (margin >= CONF_MARGIN) and (top1p >= CONF_TOP1P) and (ent <= CONF_ENTROPY)
        need_sc = (not base_confident) and ((triggers >= SC_TRIGGER_K) or (margin < HARD_MARGIN))

        did_sc = False
        accepted = False
        sc_label = None
        sc_gap = None
        accept_thr = None

        if need_sc and used_sc < sc_budget:
            is_hard = (margin < HARD_MARGIN)
            n_samples = SC_SAMPLES_HARD if is_hard else SC_SAMPLES_BASE
            temp = SC_TEMPERATURE_HARD if is_hard else SC_TEMPERATURE_BASE

            sc_label, sc_gap = self_consistency_logprob_sum(
                model, tokenizer, prompt, allow_labels, n_samples, temp, max_len
            )
            used_sc += 1
            did_sc = True

            accept_thr = SC_ACCEPT_MARGIN_STRICT if is_hard else SC_ACCEPT_MARGIN
            accepted = (sc_gap >= accept_thr)
            if accepted:
                pred = sc_label
                accept_sc += 1

            if SC_DEBUG and used_sc <= SC_DEBUG_MAX:
                print(
                    f"[SC] i={i}/{total} id={qid} allow={allow_labels} "
                    f"pred_base={pred_base} m={margin:.4f} H={ent:.4f} p1={top1p:.4f} "
                    f"flags=(m:{int(t_margin)},H:{int(t_ent)},p1:{int(t_top1p)}) trig={triggers}/3 "
                    f"conf={int(base_confident)} "
                    f"sc=({sc_label},{sc_gap:.4f}) thr={accept_thr:.4f} accept={int(accepted)} pred_final={pred}"
                )

        if i == 1 and FIRST_SAMPLE_DEBUG:
            print("\n[DEBUG] First sample prompt (rendered with apply_chat_template):\n")
            print(prompt)
            print("\n[DEBUG] Base:", f"pred={pred_base} m={margin:.4f} H={ent:.4f} p1={top1p:.4f}")
            print("[DEBUG] Gate:", f"trig={triggers}/3 need_sc={int(need_sc)} budget={sc_budget} used_sc={used_sc}")
            if did_sc:
                print("[DEBUG] SC:", f"sc=({sc_label},{sc_gap:.4f}) thr={accept_thr:.4f} accept={int(accepted)}")
            print("[DEBUG] Final pred:", pred, "\n")

        out_rows.append({"id": qid, "answer": str(pred)})
        cnt[str(pred)] += 1

        # live progress postfix
        pbar.set_postfix({
            "acc": "n/a",
            "pred": str(pred),
            "sc": f"{used_sc}/{sc_budget}",
            "cnt1": cnt.get("1", 0),
            "cnt2": cnt.get("2", 0),
            "cnt3": cnt.get("3", 0),
            "cnt4": cnt.get("4", 0),
            "cnt5": cnt.get("5", 0),
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False)

    print(f"\n[DONE] wrote: {out_path}")
    print(f"[SC] used={used_sc}/{sc_budget} accept={accept_sc} accept_rate={(accept_sc/used_sc if used_sc else 0):.3f}")
    print("[CNT]", ", ".join([f"{k}:{v}" for k, v in sorted(cnt.items())]))


if __name__ == "__main__":
    main()