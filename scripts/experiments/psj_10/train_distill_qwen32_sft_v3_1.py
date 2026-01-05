import os
import sys


sys.path.append(os.getcwd())

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

# src 모듈 로드 시도
try:
    from src.utils.hf_utils import upload_model_to_hf
    from src.config.config import HF_MODEL_NAME, HF_ORG
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    sys.path.append(project_root)
    from src.utils.hf_utils import upload_model_to_hf
    from src.config.config import HF_MODEL_NAME, HF_ORG

@dataclass
class Config:
    model_name: str = "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"
    max_seq_len: int = 4096
    
    train_csv: str = "data/train/train_augmented_concat.csv"
    
    output_dir: str = "scripts/experiments/psj_10/qwen32_sft_ckpt_v3_1"

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
        abs_path = os.path.join(os.getcwd(), cfg.train_csv)
        if os.path.exists(abs_path):
            cfg.train_csv = abs_path
        else:
            raise FileNotFoundError(f"train csv 없음: {cfg.train_csv}")
    
    print(f"[DATA] loading: {cfg.train_csv}")
    df_raw = pd.read_csv(cfg.train_csv)
    
    passage_col = df_raw.columns[1]
    qa_col = df_raw.columns[2]

    rows = []
    for _, row in df_raw.iterrows():
        try:
            qa_dict = ast.literal_eval(str(row[qa_col]))
            question = qa_dict.get("question", "")
            choices = qa_dict.get("choices", [])
            answer = qa_dict.get("answer", None)
        except:
            continue

        if answer is None: continue

        choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        rows.append({
            "passage": str(row[passage_col]),
            "question": str(question),
            "choices": choices_str,
            "answer": int(answer)
        })

    return pd.DataFrame(rows)

PROMPT_TEMPLATE = """너는 한국 수능 한국사/국사/사회/국어 문제를 푸는 전문가야.
아래의 지문과 문제, 선택지를 읽고 가장 알맞은 하나의 정답 번호만 골라라.

[지문]
{passage}

[문제]
{question}

[선택지]
{choices}

정답:"""

def build_sft_text(row):
    prompt = PROMPT_TEMPLATE.format(
        passage=row["passage"], 
        question=row["question"], 
        choices=row["choices"]
    )
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
        r = 32,             # 🔥 [수정] Rank 32로 상향
        lora_alpha = 64,    # 🔥 [수정] Alpha 64로 상향
        lora_dropout = 0.05,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )
    return model, tokenizer

def main():
    set_seed(cfg.seed)
    df = load_and_parse_train_csv()
    train_dataset = Dataset.from_dict({"text": [build_sft_text(r) for _, r in df.iterrows()]})
    
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

    print(f"[TRAIN] Start v3.1 (r={32}) training on psj_10...")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"[SAVE] Saved to {cfg.output_dir}")

    print("[MEMORY] Cleaning up...")
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

    try:
        base_name = HF_MODEL_NAME.split("-v")[0]
        hf_name = f"{base_name}-v3-1" 
        print(f"[HF] Uploading: {hf_name}")
        upload_model_to_hf(cfg.output_dir, f"{HF_ORG}/{hf_name}", hf_name)
    except Exception as e:
        print(f"[HF] Warning: {e}")

if __name__ == "__main__":
    main()