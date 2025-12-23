import os
import ast
import re
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from tqdm import tqdm

# 1. 환경 설정
DATA_PATH = "data/train/train.csv"
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"  # ✅ 변경
OUTPUT_DIR = "./exaone_3p5_7p8b_trl_unsloth_sft_baseline"
MAX_SEQ_LENGTH = 4096  # 길어서 OOM 나면 2048로 낮춰봐

# 2. 데이터 로드 및 분할 (Train/Eval)
df = pd.read_csv(DATA_PATH)
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

# 3. 모델 및 LoRA 설정
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,          # V100 32GB면 4bit 추천
)

# eos 토큰 없을 때 대비
if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.sep_token if tokenizer.sep_token is not None else "</s>"

model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # ✅ 128 -> 64 (baseline/안정성)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 4. 프롬프트 및 전처리
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
당신은 수능 국어 및 한국사 문제 풀이 전문가입니다.
주어진 [지문]을 주의 깊게 분석하고, [질문]의 의도를 파악하여 정답을 논리적으로 추론하세요.
[보기] 중에서 가장 적절한 답을 찾아 번호로만 답하세요.

### Input:
[지문]
{paragraph}

[질문]
{question}

[보기]
{choices_str}

### Response:
{answer}"""

def formatting_prompts_func(examples):
    paragraphs = examples["paragraph"]
    problems = [ast.literal_eval(p) if isinstance(p, str) else p for p in examples["problems"]]

    texts = []
    for para, prob in zip(paragraphs, problems):
        choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(prob["choices"])])
        text = alpaca_prompt.format(
            paragraph=para,
            question=prob["question"],
            choices_str=choices_str,
            answer=prob["answer"],
        ) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

train_dataset = Dataset.from_pandas(train_df).map(formatting_prompts_func, batched=True)

# 5. 학습 (SFTTrainer)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        per_device_train_batch_size=2,       # 7.8B 4bit면 보통 OK
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=1e-4,                  # baseline 유지
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=OUTPUT_DIR,
        save_strategy="no",
    ),
)
trainer.neftune_noise_alpha = 5
trainer.train()

# 6. Evaluation 및 Macro F1 측정
print("\n--- Evaluating on Eval Split ---")
FastLanguageModel.for_inference(model)

y_true, y_pred = [], []

for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
    prob = ast.literal_eval(row["problems"]) if isinstance(row["problems"], str) else row["problems"]
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(prob["choices"])])

    input_text = alpaca_prompt.format(
        paragraph=row["paragraph"],
        question=prob["question"],
        choices_str=choices_str,
        answer=""
    ).strip()

    inputs = tokenizer([input_text], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=8,
        use_cache=True,
        do_sample=False,   # ✅ 선택지 문제는 deterministic이 대체로 유리
        temperature=0.0,
    )
    prediction = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    # 숫자 1개만 뽑기 (1~5 등)
    match = re.search(r"\b([1-9])\b", prediction)
    pred_val = int(match.group(1)) if match else 0

    y_true.append(int(prob["answer"]))
    y_pred.append(pred_val)

macro_f1 = f1_score(y_true, y_pred, average="macro")
print(f"\nFinal Eval Macro F1 Score: {macro_f1:.4f}")

# 7. 모델 저장
model.save_pretrained(f"{OUTPUT_DIR}_lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}_lora")
