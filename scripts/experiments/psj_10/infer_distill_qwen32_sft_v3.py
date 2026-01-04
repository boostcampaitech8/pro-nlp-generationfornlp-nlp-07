# scripts/experiments/psj_10/infer_distill_qwen32_sft_v3.py

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
    lora_ckpt_dir: str = "scripts/experiments/psj_10/qwen32_sft_ckpt_v3"

    test_csv: str = "data/test/test.csv"
    submission_path: str = "scripts/experiments/psj_10/submission/submission_deepseek_distill_qwen32_v3.csv"
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
        raise FileNotFoundError(f"test csv 없음: {cfg.test_csv}")
    df_raw = pd.read_csv(cfg.test_csv)
    if "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Unnamed: 0"])

    rows = []
    for _, row in df_raw.iterrows():
        try:
            qa_dict = ast.literal_eval(row["problems"])
            question = qa_dict.get("question", "")
            choices = qa_dict.get("choices", [])
        except:
            question, choices = "", []

        choices_str_lines = [f"{i+1}. {c}" for i, c in enumerate(choices)]
        rows.append({
            "id": row["id"],
            "passage": row["paragraph"],
            "question": str(question),
            "choices": "\n".join(choices_str_lines),
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

def build_prompt(passage, question, choices):
    return PROMPT_TEMPLATE.format(passage=passage, question=question, choices=choices)

def load_model_and_tokenizer():
    print(f"[LOAD] LoRA: {cfg.lora_ckpt_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg.lora_ckpt_dir,
        max_seq_length = cfg.max_seq_len,
        load_in_4bit = True,
        device_map = "auto",
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def parse_answer_from_output(text: str):
    if not text:
        return None
    match = re.search(r"([1-5①②③④⑤])", text)
    if match:
        val = match.group(1)
        mapper = {"①":1, "②":2, "③":3, "④":4, "⑤":5}
        return mapper.get(val, int(val))
    return None

def generate_one(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = {"max_new_tokens": cfg.max_new_tokens}
    if cfg.use_greedy:
        gen_kwargs.update({"do_sample": False, "temperature": 0.0})
    else:
        gen_kwargs.update({"do_sample": True, "temperature": 0.7})

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

def main():
    set_seed(cfg.seed)
    test_df = load_and_parse_test_csv()
    model, tokenizer = load_model_and_tokenizer()
    preds, raw_outputs = [], []

    print(f"[INFER] Start inference on {len(test_df)} samples...")
    t0 = time.time()
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prompt = build_prompt(row["passage"], row["question"], row["choices"])
        out_text = generate_one(model, tokenizer, prompt)
        raw_outputs.append(out_text)
        pred = parse_answer_from_output(out_text)
        if pred is None: pred = 1
        preds.append(pred)

    print(f"[DONE] Time: {(time.time()-t0)/60:.1f} min")
    sub_df = pd.DataFrame({"id": test_df["id"], "answer": preds})
    sub_df.to_csv(cfg.submission_path, index=False)
    print(f"[SAVE] Submission: {cfg.submission_path}")

    debug_path = cfg.submission_path.replace(".csv", "_raw.csv")
    test_df["raw_output"] = raw_outputs
    test_df["pred"] = preds
    test_df.to_csv(debug_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()