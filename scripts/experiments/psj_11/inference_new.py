import os
import ast
import re
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm

import psutil, builtins
builtins.psutil = psutil

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
@dataclass
class Config:
    lora_ckpt_dir: str = "NLP-07-ODQA/qwen32-cot-finetuned-psj11"
    test_csv: str = "data/test/test.csv"
    output_dir: str = "scripts/experiments/psj_11/submission"
    
    # 처음부터 다시 하므로 파일명을 명확하게 지정 (필요시 변경)
    submission_name: str = "submission_cot_final.csv"
    
    max_seq_len: int = 3072 
    max_new_tokens: int = 1100
    use_greedy: bool = True     
    seed: int = 42

cfg = Config()

# ==========================================
# 2. 프롬프트 템플릿 (Lite Version)
# ==========================================
PROMPT_TEMPLATE = """
다음 지문을 읽고 문제의 정답을 맞히세요.

[지시사항]
1. 문제의 유형(긍정 질문/부정 질문)을 먼저 파악하세요.
2. 각 선택지의 내용이 지문에 있는지 '사실 관계'만 건조하게 따지세요. (추측 금지)
3. 정답이 도출되는 논리적 근거만 짧게 서술하세요.

[지문]
{passage}

[문제]
{question}

[선택지]
{choices}

[풀이 과정]
1. 질문 분석:
2. 선지 검증:
3. 최종 결론:

정답:
"""

# ==========================================
# 3. 유틸리티 함수
# ==========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_prompt(passage, question, choices):
    return PROMPT_TEMPLATE.format(passage=passage, question=question, choices=choices)

def parse_answer_from_cot(text: str):
    if not text: return None
    
    # "정답: N" 패턴 찾기 (마지막 것 채택)
    patterns = re.findall(r"정답\s*[:：]\s*([1-5①②③④⑤])", text)
    if patterns:
        val = patterns[-1]
        mapper = {"①":1, "②":2, "③":3, "④":4, "⑤":5}
        return mapper.get(val, int(val))
    
    # 패턴 없으면 끝부분 숫자 찾기
    matches = re.findall(r"([1-5])", text[-50:])
    if matches: 
        return int(matches[-1])
    return None

def load_and_parse_test_csv():
    if not os.path.exists(cfg.test_csv):
        raise FileNotFoundError(f"test csv 없음: {cfg.test_csv}")
    
    df_raw = pd.read_csv(cfg.test_csv)
    if "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Unnamed: 0"])

    rows = []
    for _, row in df_raw.iterrows():
        try:
            if isinstance(row["problems"], str):
                qa_dict = ast.literal_eval(row["problems"])
            else:
                qa_dict = row["problems"]
            question = qa_dict.get("question", "")
            choices = qa_dict.get("choices", [])
        except:
            question, choices = "", []

        choices_str_lines = [f"{i+1}. {c}" for i, c in enumerate(choices)]
        rows.append({
            "id": row["id"],
            "passage": row["paragraph"] if pd.notna(row["paragraph"]) else "",
            "question": str(question),
            "choices": "\n".join(choices_str_lines),
        })
    return pd.DataFrame(rows)

def load_model_and_tokenizer():
    print(f"[LOAD] Model from: {cfg.lora_ckpt_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg.lora_ckpt_dir,
        max_seq_length = cfg.max_seq_len, 
        load_in_4bit = True,
        device_map = "auto",
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_one(model, tokenizer, prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    gen_kwargs = {
        "max_new_tokens": cfg.max_new_tokens, 
        "use_cache": True,
        "repetition_penalty": 1.1  # 페널티 완화
    }
    
    if cfg.use_greedy:
        gen_kwargs.update({"do_sample": False, "temperature": 0.0})
    else:
        gen_kwargs.update({"do_sample": True, "temperature": 0.3})

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

# ==========================================
# 4. 메인 실행 (Main)
# ==========================================
def main():
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # 데이터 로드
    test_df = load_and_parse_test_csv()
    
    # 경로 설정
    sub_path = os.path.join(cfg.output_dir, cfg.submission_name)
    raw_path = sub_path.replace(".csv", "_raw.csv")
    
    # ▼▼▼ [Clean Start] 기존 파일 삭제 (처음부터 하니까요) ▼▼▼
    if os.path.exists(sub_path):
        os.remove(sub_path)
        print(f"🧹 기존 제출 파일 삭제됨 (새로 시작): {sub_path}")
        
    if os.path.exists(raw_path):
        os.remove(raw_path)
        print(f"🧹 기존 디버그 파일 삭제됨: {raw_path}")
    
    # 전체 데이터 사용
    remaining_df = test_df
    
    print(f"[INFER] 전체 {len(test_df)}개 데이터에 대해 처음부터 추론을 시작합니다!")
    
    # 모델 로드
    model, tokenizer = load_model_and_tokenizer()
    
    # 추론 루프
    for idx, row in tqdm(remaining_df.iterrows(), total=len(remaining_df)):
        try:
            prompt = build_prompt(row["passage"], row["question"], row["choices"])
            
            # 생성
            cot_output = generate_one(model, tokenizer, prompt)
            
            # 정답 파싱
            pred = parse_answer_from_cot(cot_output)
            if pred is None: pred = 1 
            
            # 저장용 데이터프레임 생성
            new_row_sub = pd.DataFrame({"id": [row["id"]], "answer": [pred]})
            new_row_raw = pd.DataFrame({
                "id": [row["id"]],
                "answer": [pred],
                "reasoning": [cot_output],
                "passage": [row["passage"]],
                "question": [row["question"]]
            })
            
            # 파일 이어쓰기 (mode='a')
            # 파일이 없으면 헤더 포함, 있으면 내용만 추가
            hdr_sub = not os.path.exists(sub_path)
            new_row_sub.to_csv(sub_path, mode='a', header=hdr_sub, index=False)
            
            hdr_raw = not os.path.exists(raw_path)
            new_row_raw.to_csv(raw_path, mode='a', header=hdr_raw, index=False, encoding="utf-8-sig")
            
            # 메모리 정리
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error on {row['id']}: {e}")
            continue

    print(f"[DONE] 모든 작업 완료! 파일 저장 경로: {sub_path}")

if __name__ == "__main__":
    main()