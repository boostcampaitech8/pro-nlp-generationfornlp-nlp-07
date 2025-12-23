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
MODEL_NAME = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
OUTPUT_DIR = "./qwen3_14b_4bit_trl024_unsloth_sft_v3"
MAX_SEQ_LENGTH = 4096

# 2. 데이터 로드 및 분할 (Train/Eval)
df = pd.read_csv(DATA_PATH)
# 데이터를 9:1 비율로 분할
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

# 3. 모델 및 LoRA 설정
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # LoRA 표현력 32 -> 128로 강화
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 128,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

# 4. 프롬프트 및 전처리 (CoT 및 페르소나 강화 버전)

# 변경점:
# 1. System Prompt에 '전문가' 페르소나 부여
# 2. 단순히 '골라라'가 아니라 '논리적으로 추론하여 도출하라'는 지시 추가
# 3. Input 섹션을 [지문], [질문], [보기]로 명확히 구조화하여 가독성 높임

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
    # problems 컬럼이 문자열로 되어있으면 딕셔너리로 변환
    problems = [ast.literal_eval(p) if isinstance(p, str) else p for p in examples["problems"]]
    
    texts = []
    for para, prob in zip(paragraphs, problems):
        # 보기 포맷팅 (1. 보기내용 \n 2. 보기내용 ...)
        choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(prob['choices'])])
        
        text = alpaca_prompt.format(
            paragraph=para,
            question=prob['question'],
            choices_str=choices_str,
            # 학습 시에는 정답 번호를 넣고, 추론 시에는 비워두거나 처리
            answer=prob['answer'] 
        ) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts }

# 데이터셋 매핑 적용
train_dataset = Dataset.from_pandas(train_df).map(formatting_prompts_func, batched = True)
# 5. 학습 (SFTTrainer)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 3,
        learning_rate = 1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.05,
        #warmup_ratio = 0.1, # 초반 안정화
        lr_scheduler_type = "cosine", # 기존 linear에서 cosine으로 변경
        seed = 42,
        output_dir = OUTPUT_DIR,
        save_strategy = "no",
    ),
)
trainer.neftune_noise_alpha = 5
trainer.train()


# 6. Evaluation 및 Macro F1 측정
print("\n--- Evaluating on Eval Split ---")
FastLanguageModel.for_inference(model) # 추론 모드 활성화

y_true = []
y_pred = []

# 평가 데이터에 대해 루프
for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
    prob = ast.literal_eval(row['problems']) if isinstance(row['problems'], str) else row['problems']
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(prob['choices'])])
    
    # 정답을 제외한 입력 구성
    input_text = alpaca_prompt.format(
        paragraph=row['paragraph'],
        question=prob['question'],
        choices_str=choices_str,
        answer="" # 출력 부분은 비움
    ).strip()

    inputs = tokenizer([input_text], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 10, use_cache = True)
    prediction = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    # 결과에서 숫자만 추출 (예: "정답은 2번입니다" -> 2)
    match = re.search(r'\d', prediction)
    pred_val = int(match.group()) if match else 0 # 추출 실패 시 0 처리
    
    y_true.append(int(prob['answer']))
    y_pred.append(pred_val)

# Macro F1 점수 계산
macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"\nFinal Eval Macro F1 Score: {macro_f1:.4f}")

# 7. 모델 저장
model.save_pretrained(f"{OUTPUT_DIR}_lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}_lora")