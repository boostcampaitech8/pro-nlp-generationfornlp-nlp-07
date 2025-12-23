import ast
import re
import pandas as pd
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm

# 1. 설정
TEST_DATA_PATH = "data/test/test.csv"  # 실제 테스트 파일 경로로 수정하세요
LORA_PATH = "./qwen3_14b_4bit_trl024_unsloth_sft_v5_lora"
OUTPUT_CSV = "submission.csv"
MAX_SEQ_LENGTH = 4096

# 2. 모델 및 토크나이저 로드 (학습된 LoRA 적용)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = LORA_PATH,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # 추론 최적화

# 3. 데이터 로드
test_df = pd.read_csv(TEST_DATA_PATH)

# 4. 프롬프트 템플릿 (학습 때와 동일하게 유지)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
주어진 지문을 읽고 질문의 정답을 보기에서 골라 번호로만 답하세요.

### Input:
지문: {paragraph}
질문: {question}
보기:
{choices_str}

### Response:
"""

# 5. 추론 시작
results = []

print(f"--- Starting Inference on {len(test_df)} samples ---")
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    # problems 컬럼 파싱 (answer는 없을 것이므로 dict 안전하게 처리)
    prob = ast.literal_eval(row['problems']) if isinstance(row['problems'], str) else row['problems']
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(prob.get('choices', []))])
    
    input_text = alpaca_prompt.format(
        paragraph=row['paragraph'],
        question=prob.get('question', ''),
        choices_str=choices_str
    )

    inputs = tokenizer([input_text], return_tensors = "pt").to("cuda")
    
    # max_new_tokens를 짧게 설정하여 속도 향상 (정답 번호만 필요하므로)
    outputs = model.generate(**inputs, 
        max_new_tokens = 5, 
        use_cache = True
        )
    prediction = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    # 숫자만 추출
    match = re.search(r'\d', prediction)
    pred_val = int(match.group()) if match else 1 # 추출 실패 시 기본값 1
    
    results.append({
        "id": row["id"],
        "answer": pred_val
    })

# 6. 결과 저장
submission_df = pd.DataFrame(results)
submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n[Done] Submission file saved to {OUTPUT_CSV}")