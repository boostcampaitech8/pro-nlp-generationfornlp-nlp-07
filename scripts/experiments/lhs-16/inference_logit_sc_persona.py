\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Standalone inference script (logit + SC budget) with real-time progress logging.

- Base prediction: logit argmax over label tokens (1..K)
- Optional SC: 1-token sampling restricted to label tokens, only on uncertain samples
- SC is budgeted by MAX_SC_RATE (e.g., 0.10 -> at most 10% of samples)

Env vars:
  BASE_MODEL, ADAPTER_MODEL (optional)
  TEST_PATH, OUT_PATH
  SEED
  MAX_SC_RATE
  Gate thresholds: MARGIN_THRESHOLD, ENTROPY_THRESHOLD, TOP1P_THRESHOLD
  v3 guard: SC_TRIGGER_K, CONF_MARGIN, CONF_TOP1P, CONF_ENTROPY, HARD_MARGIN
  SC params: SC_SAMPLES_BASE, SC_TEMPERATURE_BASE, SC_SAMPLES_HARD, SC_TEMPERATURE_HARD
  SC accept: SC_ACCEPT_MARGIN, SC_ACCEPT_MARGIN_STRICT
  Debug:
    FIRST_SAMPLE_DEBUG=1  (prints first prompt + base/sc stats)
    SC_DEBUG=1            (prints per-SC decision line up to SC_DEBUG_MAX)
    SC_DEBUG_MAX=200
'''

import os
import math
import random
from ast import literal_eval
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
from peft import PeftModel

try:
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except Exception:
    _HAS_UNSLOTH = False


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



# ===== Persona & silent reasoning (KoNET-inspired) =====
PERSONA_PRESETS = {
    "teacher":   "너는 수능 국어/사회 영역 문제를 정확히 푸는 베테랑 교사다.",
    "professor": "너는 논리적 엄밀성을 중시하는 교수다.",
    "academic":  "너는 근거 기반으로 오류를 배제하는 연구자다.",
    "student":   "너는 함정을 피하며 실전적으로 푸는 수험생이다.",
}

def get_persona_prefix(name: str | None) -> str:
    if not name:
        return ""
    key = name.strip().lower()
    text = PERSONA_PRESETS.get(key, "")
    return (text + "\n\n") if text else ""


def build_prompt(
    paragraph: str,
    question: str,
    question_plus: str | None,
    choices: list[str],
    persona: str | None = None,
    silent_cot: bool | None = None,
) -> str:
    # persona: teacher/professor/academic/student (or None)
    # silent_cot: if True, nudge internal step-by-step checking but keep output as 1 token.
    if silent_cot is None:
        silent_cot = (os.environ.get("SILENT_COT", "1") == "1")

    persona_prefix = get_persona_prefix(persona)
    qps = f"\n<보기>:\n{question_plus}" if question_plus else ""
    ch = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])

    silent_hint = ""
    if silent_cot:
        silent_hint = "정답을 고르기 전에 (출력하지 말고) 핵심 근거를 단계적으로 점검하세요.\n"

    return (
        f"{persona_prefix}지문을 읽고 질문의 답을 구하세요.\n\n"
        f"지문:\n{paragraph}\n\n"
        f"질문:\n{question}{qps}\n\n"
        f"선택지:\n{ch}\n\n"
        f"{silent_hint} 1~{len(choices)} 중에 하나를 정답으로 고르세요.\n"
        "정답:"
    )


@torch.inference_mode()
def logits_label_stats(model, tokenizer, prompt: str, allow_labels: list[int]):
    # prompt can be a single string or a list of strings (persona-diverse SC)
    if isinstance(prompt, list):
        prompts = [p for p in prompt if p]
        if len(prompts) == 0:
            prompts = [""]
    else:
        prompts = [prompt]

    # Aggregate last-token logits across prompts by sum
    ids = [tokenizer.encode(str(l), add_special_tokens=False)[-1] for l in allow_labels]

    sub_sum = None
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        out = model(**inputs)
        scores = out.logits[0, -1]  # [vocab]
        sub = scores[torch.tensor(ids, device=scores.device)]  # [K]
        sub_sum = sub if sub_sum is None else (sub_sum + sub)

    sub = sub_sum
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



@torch.inference_mode()
def self_consistency_logprob_sum(
    model, tokenizer, prompt: str | list[str], allow_labels: list[int],
    n_samples: int, temperature: float,
):
    allow_token_ids = set(tokenizer.encode(str(l), add_special_tokens=False)[-1] for l in allow_labels)
    lp = LogitsProcessorList([RestrictTokensLogitsProcessor(allow_token_ids)])

    max_len = int(os.getenv("MAX_SEQ_LENGTH", "4096"))
    # prompt can be str or list[str]
    if isinstance(prompt, list):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,          
            truncation=True,      
            max_length=max_len,
        ).to(model.device)
    else:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(model.device)
    label_scores = defaultdict(float)
    any_valid = False

    id_to_label = {tokenizer.encode(str(l), add_special_tokens=False)[-1]: str(l) for l in allow_labels}

    for _ in range(n_samples):
        #_inputs = tokenized[_ % len(tokenized)]
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
    base_model = os.environ.get("BASE_MODEL", "unsloth/Qwen3-32B-bnb-4bit")
    adapter_model = os.environ.get("ADAPTER_MODEL", "").strip()
    test_path = os.environ.get("TEST_PATH", "data/test.csv")
    out_path = os.environ.get("OUT_PATH", "submission_sc_budget.csv")
    seed = int(os.environ.get("SEED", "42"))

    # KoNET-inspired: persona for diversity + silent reasoning hint (output still 1 token)
    BASE_PERSONA = os.environ.get("BASE_PERSONA", "teacher").strip() or None
    SC_PERSONAS = [s.strip() for s in os.environ.get("SC_PERSONAS", "teacher,student,professor,academic").split(",") if s.strip()]
    # If 0, SC uses the same base prompt (no persona diversity)
    SC_USE_PERSONA = (os.environ.get("SC_USE_PERSONA", "1") == "1")


    # gating thresholds
    MARGIN_THRESHOLD = float(os.environ.get("MARGIN_THRESHOLD", "0.75"))
    ENTROPY_THRESHOLD = float(os.environ.get("ENTROPY_THRESHOLD", "1.40"))
    TOP1P_THRESHOLD = float(os.environ.get("TOP1P_THRESHOLD", "0.40"))

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

    # debug
    FIRST_SAMPLE_DEBUG = int(os.environ.get("FIRST_SAMPLE_DEBUG", "1"))
    SC_DEBUG = int(os.environ.get("SC_DEBUG", "0"))
    SC_DEBUG_MAX = int(os.environ.get("SC_DEBUG_MAX", "200"))

    set_seed(seed)

    # load model
    if _HAS_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=4096,
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

    if adapter_model:
        model = PeftModel.from_pretrained(model, adapter_model)

    model.eval()

    df = pd.read_csv(test_path)
    total = len(df)
    sc_budget = int(math.floor(total * MAX_SC_RATE))
    used_sc = 0
    accept_sc = 0

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
        problems = literal_eval(row["problems"])
        p0 = problems[0] if isinstance(problems, list) else problems

        question = p0["question"]
        question_plus = row.get("question_plus")
        if isinstance(question_plus, float) and np.isnan(question_plus):
            question_plus = None
        if question_plus is None:
            question_plus = p0.get("question_plus")

        choices = p0.get("choices") or []
        if not choices:
            out_rows.append({"id": qid, "answer": "1"})
            continue

        allow_labels = list(range(1, len(choices) + 1))
        prompt = build_prompt(paragraph, question, question_plus, choices, persona=BASE_PERSONA)

        pred_base, margin, ent, top1p = logits_label_stats(model, tokenizer, prompt, allow_labels)
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

            sc_prompt = prompt
            if SC_USE_PERSONA and len(SC_PERSONAS) > 0:
                sc_prompt = [build_prompt(paragraph, question, question_plus, choices, persona=p) for p in SC_PERSONAS]

            sc_label, sc_gap = self_consistency_logprob_sum(
                model, tokenizer, sc_prompt, allow_labels, n_samples, temp
            )
            used_sc += 1
            did_sc = True

            accept_thr = SC_ACCEPT_MARGIN if is_hard else SC_ACCEPT_MARGIN_STRICT
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
            print("\n[DEBUG] First sample prompt:\n")
            print(prompt)
            print("\n[DEBUG] Base:", f"pred={pred_base} m={margin:.4f} H={ent:.4f} p1={top1p:.4f}")
            print("[DEBUG] Gate:", f"trig={triggers}/3 need_sc={int(need_sc)} budget={sc_budget} used_sc={used_sc}")
            if did_sc:
                print("[DEBUG] SC:", f"sc=({sc_label},{sc_gap:.4f}) thr={accept_thr:.4f} accept={int(accepted)}")
            print("[DEBUG] Final pred:", pred, "\n")

        out_rows.append({"id": qid, "answer": str(pred)})

        pbar.set_postfix(
            pred=pred,
            sc=f"{used_sc}/{sc_budget}",
            acc=f"{(accept_sc/used_sc):.2f}" if used_sc else "n/a",
        )

    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"\n[DONE] wrote {out_path}")
    print(f"SC used: {used_sc}/{total} (budget={sc_budget}, MAX_SC_RATE={MAX_SC_RATE})")
    if used_sc:
        print(f"SC accepted: {accept_sc}/{used_sc} ({accept_sc/used_sc:.3f})")


if __name__ == "__main__":
    main()
