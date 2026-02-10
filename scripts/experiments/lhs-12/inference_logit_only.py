#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Standalone inference (logit-only, no SC) with progress logging.
'''

import os
import random
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer
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


def build_prompt(paragraph: str, question: str, question_plus: str | None, choices: list[str]) -> str:
    qps = f"\n<보기>:\n{question_plus}" if question_plus else ""
    ch = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    return (
        "지문을 읽고 질문의 답을 구하세요.\n\n"
        f"지문:\n{paragraph}\n\n"
        f"질문:\n{question}{qps}\n\n"
        f"선택지:\n{ch}\n\n"
        f"1~{len(choices)} 중에 하나를 정답으로 고르세요.\n"
        "정답:"
    )


@torch.inference_mode()
def predict_one(model, tokenizer, prompt: str, allow_labels: list[int]) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[0, -1]  # [vocab]

    label_token_ids = [
        tokenizer.encode(str(l), add_special_tokens=False)[-1]
        for l in allow_labels
    ]
    sub = logits[torch.tensor(label_token_ids, device=logits.device)]
    pred_idx = int(torch.argmax(sub).item())
    return str(allow_labels[pred_idx])


def main():
    base_model = os.environ.get("BASE_MODEL", "unsloth/Qwen3-32B-bnb-4bit")
    adapter_model = os.environ.get("ADAPTER_MODEL", "").strip()
    test_path = os.environ.get("TEST_PATH", "data/test.csv")
    out_path = os.environ.get("OUT_PATH", "submission_logit_only.csv")
    seed = int(os.environ.get("SEED", "42"))

    set_seed(seed)

    # Load model
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
    out_rows = []

    pbar = tqdm(df.to_dict(orient="records"),
                total=total,
                desc="Batch Inference (logit-only)",
                dynamic_ncols=True)

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
            pred = "1"
        else:
            allow_labels = list(range(1, len(choices) + 1))
            prompt = build_prompt(paragraph, question, question_plus, choices)
            pred = predict_one(model, tokenizer, prompt, allow_labels)

        # 🔎 첫 샘플 디버깅 출력
        if i == 1:
            print("\n[DEBUG] First sample prompt:\n")
            print(prompt)
            print("\n[DEBUG] Model prediction:", pred, "\n")

        out_rows.append({"id": qid, "answer": pred})
        pbar.set_postfix(pred=pred)

    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"\n[DONE] wrote {out_path} (logit-only)")


if __name__ == "__main__":
    main()
