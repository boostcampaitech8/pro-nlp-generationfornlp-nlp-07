# scripts/experiments/psj_09/train_distill_qwen32_sft_eval.py
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score  # 🔥 Macro F1 계산용

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

    train_csv: str = "data/train/train.csv"   # 필요하면 경로만 바꿔줘
    output_dir: str = "scripts/experiments/psj_09/qwen32_sft_ckpt"

    seed: int = 42
    test_size: float = 0.1   # train/val 비율

    num_epochs: int = 3      # ← SOTA 시도할 땐 2~3으로 올려도 됨
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
        raise FileNotFoundError(f"train.csv를 찾을 수 없습니다: {cfg.train_csv}")

    print(f"[DATA] loading train csv: {cfg.train_csv}")
    df_raw = pd.read_csv(cfg.train_csv)

    if len(df_raw.columns) < 3:
        raise ValueError(f"열이 최소 3개(id, passage, qa_json)는 있어야 합니다. 현재 열: {df_raw.columns.tolist()}")

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
            # 파싱 실패한 경우 스킵
            print(f"[WARN] qa_json 파싱 실패 → 스킵: {qa_str[:50]}..., error={e}")
            continue

        # {'question': ..., 'choices': [...], 'answer': 4} 전제
        question = qa_dict.get("question", "")
        choices = qa_dict.get("choices", [])
        answer = qa_dict.get("answer", None)

        if answer is None:
            continue

        # choices를 보기 좋게 합쳐놓기
        # 예: "1. ...\n2. ...\n3. ..."
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
    print(f"[DATA] 유효 샘플 수: {len(df)}")
    return df


# ======================
# 3. 프롬프트 템플릿
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
    # 정답 숫자만 supervision으로 줌 (풀이 텍스트가 없다 보니)
    # 나중에 해설 텍스트 생기면 "정답: {answer}\n풀이: ..." 식으로 확장 가능
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

    # Unsloth + Qwen/DeepSeek 추천 패턴
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
        use_gradient_checkpointing = "unsloth",  # or True
    )

    return model, tokenizer


# ======================
# 5. 정답 파싱 (eval용)
# ======================
def parse_answer_from_output(text: str):
    """
    모델 출력에서 '정답: N' 패턴을 찾아 N(정답 번호)를 int로 반환.
    못 찾으면 None.
    """
    pattern = r"정답\s*[:：]\s*([0-9①②③④⑤])"
    m = re.search(pattern, text)
    if not m:
        return None

    ans = m.group(1)
    circled_map = {
        "①": 1,
        "②": 2,
        "③": 3,
        "④": 4,
        "⑤": 5,
    }
    if ans in circled_map:
        return circled_map[ans]
    try:
        return int(ans)
    except ValueError:
        return None


def generate_one(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
    )
    if cfg.use_greedy:
        gen_kwargs.update(dict(do_sample=False, temperature=0.0))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=0.7, top_p=0.9))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text


# ======================
# 6. train/val split + SFT 학습 + eval + HF 업로드
# ======================
def main():
    set_seed(cfg.seed)

    df = load_and_parse_train_csv()
    if len(df) == 0:
        raise ValueError("유효한 train 샘플이 없습니다.")

    # stratify = answer (가능하면)
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            stratify=df["answer"],
        )
    except Exception as e:
        print(f"[WARN] stratify split 실패 → 일반 split 사용. error={e}")
        train_df, val_df = train_test_split(
            df,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            shuffle=True,
        )

    print(f"[SPLIT] train: {len(train_df)}, val: {len(val_df)} (seed={cfg.seed})")

    # SFT용 텍스트 만들기
    train_texts = [build_sft_text(r) for _, r in train_df.iterrows()]
    train_dataset = Dataset.from_dict({"text": train_texts})

    # 모델 로드
    model, tokenizer = load_model_and_tokenizer()

    # 학습 설정
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

    print("[TRAIN] start training...")
    t0 = time.time()
    trainer.train()
    t1 = time.time()
    print(f"[TRAIN] done. elapsed = {(t1 - t0)/60:.1f} min")

    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    print(f"[SAVE] LoRA checkpoint saved to: {cfg.output_dir}")

    # ======================
    # Hugging Face 업로드
    # ======================
    try:
        model_name = f"{HF_ORG}/{HF_MODEL_NAME}"
        print(f"[HF] Uploading model to Hugging Face: {model_name}")
        upload_model_to_hf(
            checkpoint_path=cfg.output_dir,
            model_name=model_name,
            experiment_name=HF_MODEL_NAME,
        )
        print("[HF] Upload completed successfully.")
    except Exception as e:
        print(f"[HF][WARN] Hugging Face upload failed: {e}")

    # ======================
    # Eval on val_df
    # ======================
    print("[EVAL] evaluating on validation split...")
    model.eval()
    FastLanguageModel.for_inference(model)

    preds = []
    correct_flags = []
    raw_outputs = []

    t0 = time.time()
    for idx, row in val_df.iterrows():
        prompt = build_prompt(row["passage"], row["question"], row["choices"])
        out_text = generate_one(model, tokenizer, prompt)
        raw_outputs.append(out_text)
        pred = parse_answer_from_output(out_text)
        preds.append(pred)

        gt = int(row["answer"])
        is_correct = (pred is not None) and (pred == gt)
        correct_flags.append(is_correct)

        if (len(correct_flags) % 10) == 0:
            acc_so_far = sum(correct_flags) / len(correct_flags) * 100
            print(f"[EVAL] {len(correct_flags)}/{len(val_df)} acc_so_far = {acc_so_far:.2f}%")

    total_acc = sum(correct_flags) / len(correct_flags) * 100
    t1 = time.time()

    # ======================
    # Macro F1 계산
    # ======================
    y_true = []
    y_pred = []
    for gt, p in zip(val_df["answer"].tolist(), preds):
        if p is None:
            continue
        try:
            gt = int(gt)
            p = int(p)
        except ValueError:
            continue

        # 정답/예측은 1~5 범위만 사용
        if gt not in [1, 2, 3, 4, 5]:
            continue
        if p not in [1, 2, 3, 4, 5]:
            continue

        y_true.append(gt)
        y_pred.append(p)

    if len(y_true) > 0:
        macro_f1 = f1_score(
            y_true,
            y_pred,
            labels=[1, 2, 3, 4, 5],  # 🔥 클래스 1~5로 고정
            average="macro",
            zero_division=0,
        )
    else:
        macro_f1 = 0.0


    print(f"[EVAL DONE] val samples: {len(val_df)}")
    print(f"[EVAL DONE] val accuracy (seed={cfg.seed}): {total_acc:.2f}%")
    print(f"[EVAL DONE] val macro F1 (seed={cfg.seed}): {macro_f1:.4f}")
    print(f"[EVAL TIME] {(t1 - t0):.1f} sec")

    # 결과를 csv로도 남겨두기
    val_df = val_df.copy()
    val_df["pred"] = preds
    val_df["correct"] = correct_flags
    val_df["raw_output"] = raw_outputs

    eval_out = os.path.join(cfg.output_dir, "val_predictions_seed42.csv")
    val_df.to_csv(eval_out, index=False, encoding="utf-8-sig")
    print(f"[SAVE] eval results → {eval_out}")


if __name__ == "__main__":
    main()
