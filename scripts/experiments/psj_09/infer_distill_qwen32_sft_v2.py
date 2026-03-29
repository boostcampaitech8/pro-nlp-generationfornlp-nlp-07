# scripts/experiments/psj_09/infer_distill_qwen32_sft.py
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


# ======================
# 1. 설정
# ======================
@dataclass
class Config:
    # 🔹 학습 때 쓴 LoRA 체크포인트 디렉토리 (train_distill_qwen32_sft_eval.py 의 cfg.output_dir 와 동일하게!)
    lora_ckpt_dir: str = "scripts/experiments/psj_09/qwen32_sft_ckpt_v2"

    # 🔹 test.csv 경로
    test_csv: str = "data/test/test.csv"

    # 🔹 결과 파일 이름
    submission_path: str = "submission_deepseek_distill_qwen32_v2.csv"

    max_seq_len: int = 4096
    max_new_tokens: int = 256
    use_greedy: bool = True  # True면 greedy decoding
    seed: int = 42


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================
# 2. test.csv 로드 & 파싱
# ======================
def load_and_parse_test_csv():
    if not os.path.exists(cfg.test_csv):
        raise FileNotFoundError(f"test.csv를 찾을 수 없습니다: {cfg.test_csv}")

    print(f"[DATA] loading test csv: {cfg.test_csv}")
    df_raw = pd.read_csv(cfg.test_csv)

    # 🔹 인덱스 컬럼 자동으로 생긴 거면 날려버리기
    if "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Unnamed: 0"])

    # 🔹 네 파일 구조에 맞게 컬럼 고정
    #    id        : id
    #    passage   : paragraph
    #    qa_json   : problems  ( {'question': ..., 'choices': [...]} )
    id_col      = "id"
    passage_col = "paragraph"
    qa_col      = "problems"

    print(f"[DATA] 인식된 컬럼:")
    print(f"  ID:      {id_col}")
    print(f"  passage: {passage_col}")
    print(f"  qa_json: {qa_col}")

    rows = []
    for _, row in df_raw.iterrows():
        passage = str(row[passage_col])
        qa_str  = str(row[qa_col])

        try:
            qa_dict = ast.literal_eval(qa_str)
        except Exception as e:
            print(f"[WARN] qa_json 파싱 실패 → 빈 question/choices로 대체: {qa_str[:50]}..., error={e}")
            question = ""
            choices = []
        else:
            question = qa_dict.get("question", "")
            choices  = qa_dict.get("choices", [])

        # 선택지 예쁘게 붙이기
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
            )
        )

    df = pd.DataFrame(rows)
    print(f"[DATA] test 샘플 수: {len(df)}")
    return df


# ======================
# 3. 프롬프트 템플릿 (train 스크립트와 동일)
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


# ======================
# 4. 모델 로드 (LoRA 체크포인트)
# ======================
def load_model_and_tokenizer():
    """
    학습 끝나고 trainer.save_model(cfg.output_dir)으로 저장한
    LoRA 체크포인트를 그대로 불러와서 사용.
    """
    if not os.path.isdir(cfg.lora_ckpt_dir):
        raise FileNotFoundError(
            f"LoRA 체크포인트 디렉토리를 찾을 수 없습니다: {cfg.lora_ckpt_dir}"
        )

    print(f"[LOAD] LoRA model from: {cfg.lora_ckpt_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = cfg.lora_ckpt_dir,   # 🔥 base가 아니라 LoRA 디렉토리
        max_seq_length = cfg.max_seq_len,
        load_in_4bit   = True,
        device_map     = "auto",
    )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ======================
# 5. 정답 파싱 (개선 버전)
# ======================
def parse_answer_from_output(text: str):
    """
    모델 출력에서 정답 번호를 최대한 뽑는다.
    우선순위:
      1) '정답: N' / '정답 N' / '정답은 N' / '답안: N' 패턴
      2) 출력 마지막 부분에서 1~5 한 번 더 스캔
    그래도 못 찾으면 None 리턴.
    """
    if not isinstance(text, str):
        return None

    circled_map = {
        "①": 1,
        "②": 2,
        "③": 3,
        "④": 4,
        "⑤": 5,
    }

    # 1) '정답' / '답안' 근처에서 먼저 찾기
    # 콜론 있어도 되고 없어도 되고, '정답은 3번입니다' 형태도 허용
    patterns = [
        r"정답\s*[:：]?\s*([1-5①②③④⑤])",       # 정답: 3 / 정답 3
        r"정답은\s*([1-5①②③④⑤])",              # 정답은 3
        r"답안\s*[:：]?\s*([1-5①②③④⑤])",       # 답안: 4
        r"답은\s*([1-5①②③④⑤])",               # 답은 5
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            ans = m.group(1)
            if ans in circled_map:
                return circled_map[ans]
            try:
                return int(ans)
            except ValueError:
                pass

    # 2) 그래도 못 찾으면, 출력의 마지막 부분에서 숫자 한 번 더 시도
    tail = text[-200:]  # 끝 200자만 보기
    m2 = re.search(r"([1-5①②③④⑤])", tail)
    if m2:
        ans = m2.group(1)
        if ans in circled_map:
            return circled_map[ans]
        try:
            return int(ans)
        except ValueError:
            pass

    # 3) 진짜 끝까지 못 찾으면
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
# 6. main: test 인퍼런스 + submission.csv 저장
# ======================
def main():
    set_seed(cfg.seed)

    # 1) test.csv 파싱
    test_df = load_and_parse_test_csv()

    # 2) 모델 로드
    model, tokenizer = load_model_and_tokenizer()
    model.eval()

    preds = []
    raw_outputs = []

    print("[INFER] start generation on test set...")
    t0 = time.time()

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prompt = build_prompt(row["passage"], row["question"], row["choices"])
        out_text = generate_one(model, tokenizer, prompt)
        raw_outputs.append(out_text)

        pred = parse_answer_from_output(out_text)

        # 🔸 혹시 파싱 실패하거나 1~5 범위가 아니면, 일단 1로 fallback
        if pred is None or pred not in [1, 2, 3, 4, 5]:
            # 디버깅 원하면 아래 주석을 잠깐 풀어서 어떤 출력이 실패했는지 확인 가능
            # print("[WARN] 정답 파싱 실패, 기본값 1로 대체. 출력:", out_text)
            pred = 1

        preds.append(pred)

    t1 = time.time()
    print(f"[INFER DONE] test samples: {len(test_df)}")
    print(f"[INFER TIME] {(t1 - t0):.1f} sec")

    # 3) submission.csv 저장
    sub_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "answer": preds,  # 대회 포맷에 맞춰서 컬럼명 조정
        }
    )
    sub_df.to_csv(cfg.submission_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] submission → {cfg.submission_path}")

    # (선택) raw 모델 출력도 따로 저장
    debug_out = cfg.submission_path.replace(".csv", "_raw_outputs.csv")
    debug_df = test_df.copy()
    debug_df["raw_output"] = raw_outputs
    debug_df.to_csv(debug_out, index=False, encoding="utf-8-sig")
    print(f"[SAVE] raw outputs → {debug_out}")


if __name__ == "__main__":
    main()
