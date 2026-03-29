# scripts/experiments/psj_10/train_distill_qwen32_sft_v3.py

import os
import sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)

import ast
import random
import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

import psutil, builtins
builtins.psutil = psutil

from src.utils.hf_utils import upload_model_to_hf
from src.config.config import HF_MODEL_NAME, HF_ORG

@dataclass
class Config:
    model_name: str = "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"
    max_seq_len: int = 4096
    train_csv: str = "data/train/train_augmented_concat.csv"

    output_dir: str = "scripts/experiments/psj_10/qwen32_sft_ckpt_v3"

    seed: int = 42
    num_epochs: int = 3
    lr: float = 1e-4
    per_device_train_batch_size: int = 1
    grad_accum_steps: int = 4
    logging_steps: int = 10
    max_new_tokens: int = 16 
    use_greedy: bool = True

cfg = Config()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_and_parse_train_csv():
    if not os.path.exists(cfg.train_csv):
        raise FileNotFoundError(f"train csv를 찾을 수 없습니다: {cfg.train_csv}")
    
    print(f"[DATA] loading train csv: {cfg.train_csv}")
    df_raw = pd.read_csv(cfg.train_csv)
    id_col = df_raw.columns[0]
    passage_col = df_raw.columns[1]
    qa_col = df_raw.columns[2]

    rows = []
    for _, row in df_raw.iterrows():
        passage = str(row[passage_col])
        qa_str = str(row[qa_col])
        try:
            qa_dict = ast.literal_eval(qa_str)
        except Exception:
            continue

        question = qa_dict.get("question", "")
        choices = qa_dict.get("choices", [])
        answer = qa_dict.get("answer", None)

        if answer is None:
            continue

        choices_str_lines = []
        for idx, c in enumerate(choices, start=1):
            choices_str_lines.append(f"{idx}. {c}")
        choices_str = "\n".join(choices_str_lines)

        rows.append(dict(
            passage=passage,
            question=str(question),
            choices=choices_str,
            answer=int(answer),
        ))

    df = pd.DataFrame(rows)
    print(f"[DATA] 유효 샘플 수: {len(df)}")
    return df

PROMPT_TEMPLATE = """너는 한국 수능 한국사/국사/사회/국어 문제를 푸는 전문가야.
아래의 지문과 문제, 선택지를 읽고 가장 알맞은 하나의 정답 번호만 골라라.

[지문]
{passage}

[문제]
{question}

[선택지]
{choices}

정답:"""

def build_prompt(passage: str, question: str, choices: str) -> str:
    return PROMPT_TEMPLATE.format(passage=passage, question=question, choices=choices)

def build_sft_text(row):
    prompt = build_prompt(row["passage"], row["question"], row["choices"])
    return f"{prompt} {row['answer']}"

def load_model_and_tokenizer():
    print(f"[LOAD] model: {cfg.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg.model_name,
        max_seq_length = cfg.max_seq_len,
        load_in_4bit = True,
        device_map = "auto",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )
    return model, tokenizer

def main():
    set_seed(cfg.seed)
    df = load_and_parse_train_csv()
    train_texts = [build_sft_text(r) for _, r in df.iterrows()]
    train_dataset = Dataset.from_dict({"text": train_texts})
    model, tokenizer = load_model_and_tokenizer()

    training_args = TrainingArguments(
        output_dir = cfg.output_dir,
        per_device_train_batch_size = cfg.per_device_train_batch_size,
        gradient_accumulation_steps = cfg.grad_accum_steps,
        num_train_epochs = cfg.num_epochs,
        learning_rate = cfg.lr,
        logging_steps = cfg.logging_steps,
        save_strategy = "epoch",
        fp16 = True,
        bf16 = False,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        args = training_args,
        max_seq_length = cfg.max_seq_len,
    )

    print("[TRAIN] Start training (v3 - Answer Only)...")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"[SAVE] Saved to {cfg.output_dir}")

    print("[MEMORY] Cleaning up GPU memory before upload...")
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("[MEMORY] GPU memory cleared.")

    try:
        base_name = HF_MODEL_NAME.split("-v")[0]
        hf_experiment_name = f"{base_name}-v3"
        model_name_hf = f"{HF_ORG}/{hf_experiment_name}"
        print(f"[HF] Uploading: {model_name_hf}")
        upload_model_to_hf(
            checkpoint_path=cfg.output_dir,
            model_name=model_name_hf,
            experiment_name=hf_experiment_name,
        )
    except Exception as e:
        print(f"[HF] Upload Warning: {e}")

if __name__ == "__main__":
    main()