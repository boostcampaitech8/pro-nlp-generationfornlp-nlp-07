import os
import ast
import random
import numpy as np
import pandas as pd
import torch
import wikipedia
from unsloth import FastLanguageModel

# ======================
# 1. 환경 & 경로 설정
# ======================

MODEL_NAME = "NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_15"

TEST_PATH = "data/test/test.csv"
CLASS_PATH = "data/train/classification_results_psj_test.csv"

# submission.csv는 프로젝트 루트에 저장
OUTPUT_CSV = "submission.csv"

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 8  # 정답 번호만 뽑을 거라 짧게

SEED = 42


# ======================
# 2. 유틸: 시드 고정
# ======================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================
# 3. 모델 로딩 (Unsloth)
# ======================

def load_unsloth_model():
    print("[MODEL] Unsloth 모델 로딩 시작...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,       # auto
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[MODEL] Unsloth 모델 로딩 완료!")
    return model, tokenizer


# ======================
# 4. 위키피디아 설정 + RAG 검색 함수
# ======================

wikipedia.set_lang("ko")

def wiki_retrieve(query: str, max_pages: int = 3, max_chars: int = 800) -> str:
    """주어진 query로 위키 검색해서 컨텍스트 텍스트 반환."""
    if not isinstance(query, str) or not query.strip():
        return ""

    try:
        titles = wikipedia.search(query)[:max_pages]
    except Exception as e:
        print(f"[WIKI] 검색 실패: {query} | {e}")
        return ""

    ctx_list = []
    for t in titles:
        try:
            page = wikipedia.page(t, auto_suggest=False)
            text = page.content[:max_chars]
            ctx_list.append(f"[제목] {page.title}\n{text}")
        except Exception:
            continue

    return "\n\n".join(ctx_list)


# ======================
# 5. 프롬프트 & 답 파싱 함수
# ======================

NUM_MAP = {
    "①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
}


def build_prompt(passage: str, question: str, choices, wiki_ctx: str = "") -> str:
    """수능형 문제 + (옵션) 위키 컨텍스트를 포함한 프롬프트 생성."""
    choices_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
    wiki_block = wiki_ctx.strip() if wiki_ctx and wiki_ctx.strip() else "없음"

    prompt = f"""너는 한국어 수능형 독해 문제를 푸는 AI 모델이다.
지문과 문항, 보기를 읽고 정답 번호를 고른다.
정답은 반드시 1, 2, 3, 4, 5 중 하나의 숫자만 출력한다.

[위키 컨텍스트]
{wiki_block}

[지문]
{passage}

[문항]
{question}

[보기]
{choices_str}

정답 번호만 '1', '2', '3', '4', '5' 중 하나로 출력해.
정답 번호:"""
    return prompt


def parse_label(raw) -> int | None:
    """모델 출력 문자열에서 1~5 정수로 변환."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        raw = str(int(raw))
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    # 동그라미 숫자 / 그냥 숫자 매핑
    if text in NUM_MAP:
        return NUM_MAP[text]

    # 안에 숫자만 골라내기
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits and int(digits) in [1, 2, 3, 4, 5]:
        return int(digits)

    return None


def infer_one(model, tokenizer, passage, question, choices, wiki_ctx: str = "") -> int | None:
    """한 문제에 대해 (옵션) wiki_ctx를 넣어 정답 번호 예측."""
    prompt = build_prompt(passage, question, choices, wiki_ctx=wiki_ctx)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # 첫 토큰에서 숫자만 뽑기
    gen_first = gen.split()[0] if gen.split() else gen
    pred = parse_label(gen_first)
    return pred


# ======================
# 6. 메인 로직 (TEST → submission.csv)
# ======================

def main():
    set_seed(SEED)

    # ---- 모델 로딩 ----
    model, tokenizer = load_unsloth_model()

    # ---- 데이터 로딩 ----
    print(f"[DATA] test.csv 로드: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)

    print(f"[DATA] classification_results_psj_test.csv 로드: {CLASS_PATH}")
    cls_df = pd.read_csv(CLASS_PATH)

    # use_rag / query 컬럼 이름 가정:
    #   id, category, use_rag, query
    print("[DATA] test + classification_results_psj_test 조인(id 기준)")
    df = test_df.merge(cls_df, on="id", how="left")

    # 시드 42 기준으로 순서 고정 (재현성)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"[DATA] 최종 df shape = {df.shape}")
    print(f"[DATA] 컬럼: {list(df.columns)}")

    submission_rows = []

    for idx, row in df.iterrows():
        paragraph = row["paragraph"]
        problems_str = row["problems"]

        try:
            problems = ast.literal_eval(problems_str)
        except Exception as e:
            print(f"[WARN] problems 파싱 실패 (idx={idx}, id={row.get('id')}): {e}")
            continue

        question = problems.get("question", "")
        choices = problems.get("choices", [])

        # --- RAG 여부 & 쿼리 ---
        use_rag_flag = row.get("use_rag", False)
        # csv에 따라 'wiki_query' 혹은 'query'일 수 있으니 둘 다 지원
        wiki_query = row.get("wiki_query", None) or row.get("query", None)

        # bool로 정리
        if isinstance(use_rag_flag, str):
            use_rag_flag_norm = use_rag_flag.lower() in ["true", "1", "yes", "y"]
        else:
            use_rag_flag_norm = bool(use_rag_flag)

        # ===== 위키 컨텍스트 구성 =====
        wiki_ctx = ""
        if use_rag_flag_norm and isinstance(wiki_query, str) and wiki_query.strip():
            print(f"[RAG] idx={idx}, id={row['id']} | query='{wiki_query}'")
            wiki_ctx = wiki_retrieve(wiki_query)

        # ===== 최종 예측 (RAG 컨텍스트 포함/미포함) =====
        pred = infer_one(
            model=model,
            tokenizer=tokenizer,
            passage=paragraph,
            question=question,
            choices=choices,
            wiki_ctx=wiki_ctx,
        )

        # 혹시 None 나오면 안전하게 1번으로 fallback
        if pred is None:
            print(f"[WARN] id={row['id']} | 예측 실패 → 1로 대체")
            pred = 1

        submission_rows.append({
            "id": row["id"],
            "answer": int(pred),
        })

        if (idx + 1) % 10 == 0:
            print(f"[PROGRESS] {idx+1}/{len(df)} 완료")

    # ======================
    # 7. submission.csv 저장
    # ======================
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[SAVE] submission 저장 완료: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
