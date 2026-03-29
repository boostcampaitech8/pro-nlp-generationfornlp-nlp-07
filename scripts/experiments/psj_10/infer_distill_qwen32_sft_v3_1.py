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
    lora_ckpt_dir: str = "scripts/experiments/psj_10/qwen32_sft_ckpt_v3_1"
    
    test_csv: str = "data/test/test.csv"
    
    submission_path: str = "scripts/experiments/psj_10/submission/submission_deepseek_distill_qwen32_v3_1.csv"
    
    max_seq_len: int = 4096
    max_new_tokens: int = 16
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
        raise FileNotFoundError(f"csv 없음: {cfg.test_csv}")
    df_raw = pd.read_csv(cfg.test_csv)
    if "Unnamed: 0" in df_raw.columns: df_raw = df_raw.drop(columns=["Unnamed: 0"])

    rows = []
    for _, row in df_raw.iterrows():
        try:
            qa_dict = ast.literal_eval(row["problems"])
            q = qa_dict.get("question", "")
            c = qa_dict.get("choices", [])
        except:
            q, c = "", []
        
        c_str = "\n".join([f"{i+1}. {x}" for i, x in enumerate(c)])
        rows.append({"id": row["id"], "passage": row["paragraph"], "question": str(q), "choices": c_str})
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

def generate_one(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = {"max_new_tokens": cfg.max_new_tokens, "do_sample": False, "temperature": 0.0} if cfg.use_greedy else {"max_new_tokens": cfg.max_new_tokens, "do_sample": True, "temperature": 0.7}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

def parse_answer(text):
    if not text: return 1
    m = re.search(r"([1-5①②③④⑤])", text)
    if m:
        val = m.group(1)
        return {"①":1,"②":2,"③":3,"④":4,"⑤":5}.get(val, int(val))
    return 1

def main():
    set_seed(cfg.seed)
    test_df = load_and_parse_test_csv()
    
    print(f"[LOAD] LoRA: {cfg.lora_ckpt_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.lora_ckpt_dir,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=True,
        device_map="auto"
    )
    FastLanguageModel.for_inference(model)

    preds, raw_outputs = [], []
    print("[INFER] Start...")
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prompt = PROMPT_TEMPLATE.format(passage=row["passage"], question=row["question"], choices=row["choices"])
        out = generate_one(model, tokenizer, prompt)
        raw_outputs.append(out)
        preds.append(parse_answer(out))

   
    os.makedirs(os.path.dirname(cfg.submission_path), exist_ok=True)
    
    sub_df = pd.DataFrame({"id": test_df["id"], "answer": preds})
    sub_df.to_csv(cfg.submission_path, index=False)
    print(f"[SAVE] {cfg.submission_path}")

    # Debug
    debug_path = cfg.submission_path.replace(".csv", "_raw.csv")
    test_df["raw_output"] = raw_outputs
    test_df["pred"] = preds
    test_df.to_csv(debug_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()