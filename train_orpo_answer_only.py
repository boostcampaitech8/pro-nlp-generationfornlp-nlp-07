#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from datasets import load_dataset

import torch
from peft import PeftModel

try:
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except Exception:
    _HAS_UNSLOTH = False

# TRL ORPO
from trl import ORPOConfig, ORPOTrainer


def _env_str(name, default=""):
    v = os.environ.get(name, "").strip()
    return v if v else default

def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except:
        return default

def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except:
        return default


def main():
    BASE_MODEL = _env_str("BASE_MODEL", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
    START_ADAPTER_MODEL = _env_str("START_ADAPTER_MODEL", "")  # 기존 SFT/CoT 어댑터
    PAIRS_JSONL = _env_str("PAIRS_JSONL", "pairs.jsonl")

    OUTPUT_DIR = Path(_env_str("OUTPUT_DIR", "./outputs_orpo"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MAX_SEQ_LENGTH = _env_int("MAX_SEQ_LENGTH", 4096)
    LR = _env_float("ORPO_LR", 5e-6)
    EPOCHS = _env_float("ORPO_EPOCHS", 1)
    GRAD_ACCUM = _env_int("ORPO_GRAD_ACCUM", 8)
    BATCH = _env_int("ORPO_BSZ", 1)
    BETA = _env_float("ORPO_BETA", 0.1)  # 보통 0.05~0.2 사이 탐색

    # Load dataset
    ds = load_dataset("json", data_files={"train": PAIRS_JSONL})["train"]

    # Load base model
    if not _HAS_UNSLOTH:
        raise RuntimeError("Unsloth가 없는 환경이면 별도 transformers 로더를 붙여야 합니다.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    # ---- Attach trainable adapters (REQUIRED for 4bit) ----
    if START_ADAPTER_MODEL:
        # continue from an existing adapter
        model = PeftModel.from_pretrained(model, START_ADAPTER_MODEL, is_trainable=True)
    else:
        # create a fresh LoRA adapter for ORPO
        lora_r = int(os.environ.get("LORA_R", "64"))
        lora_alpha = int(os.environ.get("LORA_ALPHA", "64"))
        lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.0"))

        # Unsloth helper to attach LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load existing adapter (continue training)
    if START_ADAPTER_MODEL:
        model = PeftModel.from_pretrained(model, START_ADAPTER_MODEL, is_trainable=True)

    # ORPOConfig
    cfg = ORPOConfig(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        beta=BETA,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_SEQ_LENGTH,  # prompt가 길면 여기서 자르는데, 너는 left-trunc을 선호하면 tokenizer 설정으로 조정 가능
    )

    trainer = ORPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(str(OUTPUT_DIR / "orpo_adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "orpo_adapter"))
    print(f"[DONE] saved -> {OUTPUT_DIR/'orpo_adapter'}")


if __name__ == "__main__":
    main()
