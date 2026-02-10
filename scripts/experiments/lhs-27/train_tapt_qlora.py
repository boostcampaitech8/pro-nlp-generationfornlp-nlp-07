#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import argparse
import pandas as pd
from datasets import Dataset

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Optional: QLoRA
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import bitsandbytes as bnb  # noqa: F401
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_train_paragraphs(train_csv: str) -> list[str]:
    df = pd.read_csv(train_csv)
    if "paragraph" not in df.columns:
        raise ValueError(f"`paragraph` column not found. columns={list(df.columns)}")

    texts = [normalize_text(x) for x in df["paragraph"].tolist()]
    # 너무 짧은 텍스트 제거(노이즈 방지)
    texts = [t for t in texts if len(t) >= 80]
    return texts


def chunked_group_texts(examples, block_size: int):
    # HF 공식 방식: 토큰을 이어붙인 뒤 block_size로 자르기
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {}
    for k, t in concatenated.items():
        result[k] = [t[i : i + block_size] for i in range(0, total_length, block_size)]
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    # QLoRA toggle
    ap.add_argument("--use_qlora", type=int, default=1)  # 1=QLoRA TAPT(추천), 0=full FT
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # k-bit config
    ap.add_argument("--load_in_4bit", type=int, default=1)
    ap.add_argument("--bnb_compute_dtype", type=str, default="bfloat16")  # bfloat16/float16

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    texts = load_train_paragraphs(args.train_csv)
    print(f"[TAPT] paragraphs: {len(texts)}")

    ds = Dataset.from_dict({"text": texts})

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        return tok(batch["text"], add_special_tokens=True, truncation=False)

    tok_ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    block_size = int(args.seq_len)
    lm_ds = tok_ds.map(
        lambda x: chunked_group_texts(x, block_size=block_size),
        batched=True,
    )

    # ---- model load ----
    use_qlora = int(args.use_qlora) == 1
    if use_qlora:
        if not _HAS_PEFT:
            raise RuntimeError("peft/bitsandbytes not available. set --use_qlora 0 or install peft,bitsandbytes.")

        compute_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.bnb_compute_dtype]

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_4bit=bool(int(args.load_in_4bit)),
            torch_dtype=compute_dtype,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)

        lconf = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=None,  # 자동 탐색(모델에 따라 peft가 적절히 적용)
        )
        model = get_peft_model(model, lconf)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # ---- IMPORTANT: save only once ----
    train_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=20,
        save_strategy="no",          # ✅ 중간 저장 금지
        eval_strategy="no", 
        report_to="none",
        bf16=torch.cuda.is_available(),  # 가능하면 bf16
        fp16=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=lm_ds,
        data_collator=data_collator,
    )

    trainer.train()

    # ✅ 딱 1번만 저장
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)

    print(f"[TAPT] saved final model to: {args.out_dir}")


if __name__ == "__main__":
    main()
