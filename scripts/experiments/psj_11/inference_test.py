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
from tqdm import tqdm

import psutil, builtins
builtins.psutil = psutil

@dataclass
class Config:
    lora_ckpt_dir: str = "scripts/experiments/psj_11/qwen32_cot_finetuned"
    test_csv: str = "data/test/test.csv"
    output_dir: str = "scripts/experiments/psj_11/submission"
    submission_name: str = "submission_cot_v1.csv"
    
    # 속도를 위해 512 토큰 유지
    max_seq_len: int = 3072 
    max_new_tokens: int = 512
    use_greedy: bool = True     
    seed: int = 42

cfg = Config()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

PROMPT_TEMPLATE = """당신은 수능 및 공무원 시험 한국사/국어/사회 과목의 1타 강사입니다.
주어진 지문과 문제를 읽고, 정답을 도출하는 과정을 [해설]에 작성하세요.

[주의사항]
1. 각 선택지의 근거를 **한 문장으로 핵심만 간결하게** 요약하세요.
2. 불필요한 서론이나 반복을 피하고, 300자 이내로 답변을 마치세요.
3. 마지막에 반드시 '정답 : X' 형식으로 결론을 내리세요.

[지문]
{passage}

[문제]
{question}

[선택지]
{choices}

[해설]"""

def build_prompt(passage, question, choices):
    return PROMPT_TEMPLATE.format(passage=passage, question=question, choices=choices)

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

def parse_answer_from_cot(text: str):
    if not text: return None
    match = re.search(r"정답\s*[:：]\s*([1-5①②③④⑤])", text)
    if match:
        val = match.group(1)
        mapper = {"①":1, "②":2, "③":3, "④":4, "⑤":5}
        return mapper.get(val, int(val))
    matches = re.findall(r"([1-5])", text[-50:])
    if matches: return int(matches[-1])
    return None

def generate_one(model, tokenizer, prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    # 앵무새 방지 (속도 향상)
    gen_kwargs = {
        "max_new_tokens": cfg.max_new_tokens, 
        "use_cache": True,
        "repetition_penalty": 1.2 
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

def main():
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    test_df = load_and_parse_test_csv()
    
    sub_path = os.path.join(cfg.output_dir, cfg.submission_name)
    raw_path = sub_path.replace(".csv", "_raw.csv")
    
    if os.path.exists(sub_path):
        os.remove(sub_path)
        print(f"🧹 기존 제출 파일 삭제됨: {sub_path}")
        
    if os.path.exists(raw_path):
        os.remove(raw_path)
        print(f"🧹 기존 디버그 파일 삭제됨: {raw_path}")

    # 건너뛰기 로직 삭제됨 -> 전체 데이터 실행
    remaining_df = test_df 
    
    print(f"[INFER] 전체 {len(remaining_df)}개 작업을 처음부터 시작합니다!")
    
    model, tokenizer = load_model_and_tokenizer()
    
    for idx, row in tqdm(remaining_df.iterrows(), total=len(remaining_df)):
        try:
            prompt = build_prompt(row["passage"], row["question"], row["choices"])
            
            # 생성
            cot_output = generate_one(model, tokenizer, prompt)
            
            # 파싱
            pred = parse_answer_from_cot(cot_output)
            if pred is None: pred = 1 
            
            # 데이터프레임 한 줄 생성
            new_row_sub = pd.DataFrame({"id": [row["id"]], "answer": [pred]})
            new_row_raw = pd.DataFrame({
                "id": [row["id"]],
                "answer": [pred],
                "reasoning": [cot_output],
                "passage": [row["passage"]],
                "question": [row["question"]]
            })
            
            # 파일이 없으면 헤더 포함 저장, 있으면 내용만 추가 (Append)
            hdr_sub = not os.path.exists(sub_path)
            new_row_sub.to_csv(sub_path, mode='a', header=hdr_sub, index=False)
            
            hdr_raw = not os.path.exists(raw_path)
            new_row_raw.to_csv(raw_path, mode='a', header=hdr_raw, index=False, encoding="utf-8-sig")
            
            # 메모리 청소
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error on {row['id']}: {e}")
            continue

    print(f"[DONE] 모든 작업 완료! 파일 저장됨: {sub_path}")

if __name__ == "__main__":
    main()