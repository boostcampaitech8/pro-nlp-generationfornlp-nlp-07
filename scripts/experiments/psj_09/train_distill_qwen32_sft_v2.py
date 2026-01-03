# scripts/experiments/psj_09/train_distill_qwen32_sft_v2.py

'''
기존 train 파일에서 변경 사항
- 데이터 증강한 거 239개 추가 (공무원 시험 한국사, 국어)
- eval 데이터로 쓰는거 아까워서 eval 없이 all data train
'''

import os
import ast
import re
import time
import random
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

# [HF] Hugging Face 업로드 유틸 & 설정
from src.utils.hf_utils import upload_model_to_hf
from src.config.config import HF_MODEL_NAME, HF_ORG


# ======================
# 1. 설정
# ======================
@dataclass
class Config:
    model_name: str = "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"
    max_seq_len: int = 4096

    # 🔥 증강된 train 데이터 사용
    train_csv: str = "data/train/train_augmented_concat.csv"

    # 🔥 v2 체크포인트 디렉토리 (기존이랑 구분)
    output_dir: str = "scripts/experiments/psj_09/qwen32_sft_ckpt_v2"

    seed: int = 42

    # 기존과 동일한 하이퍼파라미터
    num_epochs: int = 3
    lr: float = 1e-4
    per_device_train_batch_size: int = 1
    grad_accum_steps: int = 4
    logging_steps: int = 10
    max_new_tokens: int = 256
    use_greedy: bool = True


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================
# 2. 데이터 로드 & 파싱
# ======================
def load_and_parse_train_csv():
    if not os.path.exists(cfg.train_csv):
        raise FileNotFoundError(f"train csv를 찾을 수 없습니다: {cfg.train_csv}")

    print(f"[DATA] loading train csv: {cfg.train_csv}")
    df_raw = pd.read_csv(cfg.train_csv)

    if len(df_raw.columns) < 3:
        raise ValueError(
            f"열이 최소 3개(id, passage, qa_json)는 있어야 합니다. "
            f"현재 열: {df_raw.columns.tolist()}"
        )

    id_col = df_raw.columns[0]
    passage_col = df_raw.columns[1]
    qa_col = df_raw.columns[2]

    print(f"[DATA] 인식된 컬럼:")
    print(f"  ID:      {id_col}")
    print(f"  passage: {passage_col}")
    print(f"  qa_json: {qa_col}")

    rows = []
    for _, row in df_raw.iterrows():
        passage = str(row[passage_col])
        qa_str = str(row[qa_col])

        try:
            qa_dict = ast.literal_eval(qa_str)
        except Exception as e:
            print(f"[WARN] qa_json 파싱 실패 → 스킵: {qa_str[:50]}..., error={e}")
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

        rows.append(
            dict(
                id=row[id_col],
                passage=passage,
                question=str(question),
                choices=choices_str,
                answer=int(answer),
            )
        )

    df = pd.DataFrame(rows)
    print(f"[DATA] 유효 샘플 수 (증강 포함): {len(df)}")
    return df


# ======================
# 3. 프롬프트 템플릿 (기존과 동일)
# ======================
PROMPT_TEMPLATE = """너는 한국 수능 한국사/국사/사회/국어 문제를 푸는 전문가야.
아래의 지문과 문제, 선택지를 읽고 가장 알맞은 하나의 정답 번호만 골라라.

- 먼저 간단하게 생각 과정을 한국어로 설명하고,
- 마지막 줄에 다음 형식으로 정답을 출력해라.

정답: N

여기서 N은 1, 2, 3, 4, 5 중 하나이다.

[지문]
{passage}

[문제]
{question}

[선택지]
{choices}

생각 과정:"""


def build_prompt(passage: str, question: str, choices: str) -> str:
    return PROMPT_TEMPLATE.format(
        passage=passage,
        question=question,
        choices=choices,
    )


def build_sft_text(row):
    prompt = build_prompt(row["passage"], row["question"], row["choices"])
    return f"{prompt}\n\n정답: {row['answer']}"


# ======================
# 4. 모델 로드 & LoRA 세팅
# ======================
def load_model_and_tokenizer():
    print(f"[LOAD] model: {cfg.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = cfg.model_name,
        max_seq_length = cfg.max_seq_len,
        load_in_4bit   = True,
        device_map     = "auto",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0.05,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    return model, tokenizer


# ======================
# 5. train full data + HF 업로드 (v2 이름)
# ======================
def main():
    set_seed(cfg.seed)

    # 🔥 증강 포함 전체 train 사용 (val split X)
    df = load_and_parse_train_csv()
    if len(df) == 0:
        raise ValueError("유효한 train 샘플이 없습니다.")

    print(f"[TRAIN DATA] total train samples (no val split): {len(df)}")

    # SFT용 텍스트 만들기
    train_texts = [build_sft_text(r) for _, r in df.iterrows()]
    train_dataset = Dataset.from_dict({"text": train_texts})

    # 모델 로드
    model, tokenizer = load_model_and_tokenizer()

    # 학습 설정 (이전과 동일)
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

    print("[TRAIN] start training on full augmented data (no eval)...")
    t0 = time.time()
    trainer.train()
    t1 = time.time()
    print(f"[TRAIN] done. elapsed = {(t1 - t0)/60:.1f} min")

    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    print(f"[SAVE] LoRA checkpoint saved to: {cfg.output_dir}")

    # ======================
    # Hugging Face 업로드 (v2 이름)
    # ======================
    try:
        # HF_MODEL_NAME이 ...-v1 형태라면 -v2로 바꾸고,
        # 아니면 그냥 뒤에 -v2 붙이기
        if HF_MODEL_NAME.endswith("-v1"):
            base = HF_MODEL_NAME[:-3]  # '-v1' 자르기
            hf_experiment_name = base + "-v2"
        else:
            hf_experiment_name = HF_MODEL_NAME + "-v2"

        model_name = f"{HF_ORG}/{hf_experiment_name}"
        print(f"[HF] Uploading model to Hugging Face: {model_name}")

        upload_model_to_hf(
            checkpoint_path=cfg.output_dir,
            model_name=model_name,
            experiment_name=hf_experiment_name,
        )
        print("[HF] Upload completed successfully.")
    except Exception as e:
        print(f"[HF][WARN] Hugging Face upload failed: {e}")


if __name__ == "__main__":
    main()
