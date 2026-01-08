import os
import sys

# 프로젝트 루트 경로 설정
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

# src 모듈 로드 (HF 업로드용 - 없으면 패스)
try:
    from src.utils.hf_utils import upload_model_to_hf
    from src.config.config import HF_MODEL_NAME, HF_ORG
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    sys.path.append(project_root)
    # 임시 더미 함수
    def upload_model_to_hf(path, repo, name): print(f"[Mock] Upload {name} to {repo}")
    HF_MODEL_NAME = "DeepSeek_Distill_qwen2.5_32B_cot"
    HF_ORG = "my-org"

@dataclass
class Config:
    # 1. 모델 선택 
    model_name: str = "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"
    max_seq_len: int = 4096  # 해설이 기니까 길게
    
    # 2. 데이터 경로 (Gemini가 만든 CoT 데이터)
    train_csv: str = "data/train/train_cot_test_all.csv"
    
    output_dir: str = "scripts/experiments/psj_11/qwen32_cot_finetuned"

    seed: int = 42
    num_epochs: int = 1        
    lr: float = 2e-4           
    per_device_train_batch_size: int = 1 
    grad_accum_steps: int = 4
    logging_steps: int = 5
    max_new_tokens: int = 1024  

cfg = Config()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_and_parse_train_csv():
    """Gemini CoT 데이터 파싱"""
    if not os.path.exists(cfg.train_csv):
        raise FileNotFoundError(f"데이터 파일 없음: {cfg.train_csv} (Gemini 스크립트 완료 확인하세요!)")
    
    print(f"[DATA] Loading CoT Data: {cfg.train_csv}")
    df_raw = pd.read_csv(cfg.train_csv)
    
    # 해설(reasoning) 있는 것만 필터링
    df_clean = df_raw[df_raw['reasoning'].notna() & (df_raw['reasoning'] != "")].copy()
    print(f"[DATA] 원본 {len(df_raw)}개 -> 학습 가능 {len(df_clean)}개")

    rows = []
    for idx, row in df_clean.iterrows():
        try:
            # problems 컬럼 우선 파싱
            qa_src = row['problems'] if pd.notna(row['problems']) else row.get('question_plus')
            qa_dict = ast.literal_eval(str(qa_src))
            
            question = qa_dict.get("question", "")
            choices = qa_dict.get("choices", [])
            answer = qa_dict.get("answer", None)
            passage = str(row['paragraph']) if pd.notna(row['paragraph']) else ""
            reasoning = str(row['reasoning'])

            if not question or answer is None or not reasoning:
                continue

            choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
            
            rows.append({
                "passage": passage,
                "question": question,
                "choices": choices_str,
                "answer": int(answer),
                "reasoning": reasoning
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

# CoT용 프롬프트 (해설 유도)
PROMPT_TEMPLATE = """당신은 수능 및 공무원 시험 한국사/국어/사회 과목의 1타 강사입니다.
주어진 지문과 문제를 읽고, 정답을 도출하는 논리적인 과정을 [해설]로 상세히 설명한 뒤, 마지막에 정답 번호를 맞히세요.

[지문]
{passage}

[문제]
{question}

[선택지]
{choices}

[해설]"""

def build_sft_text(row):
    prompt = PROMPT_TEMPLATE.format(
        passage=row["passage"], 
        question=row["question"], 
        choices=row["choices"]
    )
    # 프롬프트 + 해설 + 정답
    return f"{prompt}\n{row['reasoning']}\n\n정답: {row['answer']}"

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
        r = 32,             
        lora_alpha = 64,    
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
    
    print(f"\n[Sample] {train_dataset[0]['text'][:300]}...\n")

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
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.03,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        args = training_args,
        max_seq_length = cfg.max_seq_len,
        packing = False,
    )

    print(f"[TRAIN] Start CoT Training (psj_11) on {len(df)} samples...")
    trainer.train()
    
    print(f"[SAVE] Saving to {cfg.output_dir}")
    trainer.save_model(cfg.output_dir)

    print("[MEMORY] Cleanup...")
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()