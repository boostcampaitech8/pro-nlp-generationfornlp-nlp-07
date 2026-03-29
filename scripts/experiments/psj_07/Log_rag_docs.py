# scripts/experiments/psj_07/log_rag_docs.py

import os
import ast
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from unsloth import FastLanguageModel
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder

# ✅ Kiwi + BM25
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi


# ======================
# 1. 설정
# ======================

MODEL_NAME = "NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_15"

TEST_PATH = "data/test/test.csv"

MAX_SEQ_LENGTH = 4096
SEED = 42

# 🔼 base 예측의 confidence threshold (이거보다 낮으면 RAG 사용)
CONF_THRESHOLD = 0.7

# vector store / docs
PERSIST_DIR = "scripts/experiments/psj_07/vectorstores/kowiki_e5_large"
DOCS_PKL = "scripts/experiments/psj_07/vectorstores/kowiki_docs.pkl"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

# hybrid 검색
TOP_K_DENSE = 6      # e5 dense에서 k개
TOP_K_BM25 = 6       # Kiwi BM25에서 k개
TOP_K_FINAL = 4      # reranker 이후 최종 k개

# reranker (bge-reranker-large)
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"

# RAG 사용 문항 + 불러온 문서들 로그 저장 경로
LOG_CSV_PATH = "scripts/experiments/psj_07/rag_used_docs_log_v2.csv"

# 최종 제출용 submission
OUTPUT_CSV = "submission.csv"


# ======================
# 2. 유틸
# ======================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================
# 3. 모델 로딩 (Unsloth)
# ======================

def load_unsloth_model():
    print("[MODEL] loading unsloth model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[MODEL] loaded.")
    return model, tokenizer


# ======================
# 4. Kiwi 기반 BM25 Retriever
# ======================

class KiwiBM25Retriever:
    def __init__(self, docs, k: int = 6):
        self.docs = docs
        self.k = k
        self.kiwi = Kiwi()

        print("[BM25-KIWI] building corpus tokens...")
        self.corpus_tokens = [self._tokenize(d.page_content) for d in tqdm(docs)]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print("[BM25-KIWI] ready.")

    def _tokenize(self, text: str):
        if not isinstance(text, str):
            return []
        tokens = []
        for t in self.kiwi.tokenize(text):
            # 명사/동사/형용사/수사/어근 정도만 사용
            if t.tag.startswith(("NN", "VV", "VA", "MM", "XR")):
                tokens.append(t.form)
        # 위 품사 필터로 전부 날아가면 백업으로 전체 사용
        if not tokens:
            tokens = [t.form for t in self.kiwi.tokenize(text)]
        return tokens

    def get_relevant_documents(self, query: str):
        q_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][: self.k]
        return [self.docs[i] for i in top_idx]

    # LangChain 스타일과 맞추기 위해 invoke도 구현
    def invoke(self, query: str):
        return self.get_relevant_documents(query)


# ======================
# 5. Hybrid Retriever + Reranker 세팅
# ======================

def load_dense_retriever():
    print("[RAG] loading e5 embeddings + Chroma")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vs = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    retriever = vs.as_retriever(search_kwargs={"k": TOP_K_DENSE})
    return retriever


def load_bm25_retriever():
    print(f"[RAG] loading docs for Kiwi BM25: {DOCS_PKL}")
    with open(DOCS_PKL, "rb") as f:
        docs = pickle.load(f)
    print(f"[RAG] docs loaded: {len(docs)}")

    bm25 = KiwiBM25Retriever(docs=docs, k=TOP_K_BM25)
    print("[RAG] Kiwi BM25 retriever ready.")
    return bm25


def load_reranker():
    print(f"[RERANK] loading CrossEncoder: {RERANKER_MODEL_NAME}")
    reranker = CrossEncoder(
        RERANKER_MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("[RERANK] ready.")
    return reranker


def hybrid_retrieve_with_rerank(query, dense_retriever, bm25_retriever, reranker):
    """
    dense + BM25 하이브리드 검색 후
    bge-reranker로 rerank 해서 상위 TOP_K_FINAL 문서 반환
    """

    # 1) dense / sparse 각각 검색
    if hasattr(dense_retriever, "invoke"):
        dense_docs = dense_retriever.invoke(query)
    else:
        dense_docs = dense_retriever.get_relevant_documents(query)

    if hasattr(bm25_retriever, "invoke"):
        sparse_docs = bm25_retriever.invoke(query)
    else:
        sparse_docs = bm25_retriever.get_relevant_documents(query)

    # 2) 합치고 중복 제거 (page_content + title 기준)
    def doc_key(d: Document):
        return (d.page_content, d.metadata.get("title", None))

    merged = {}
    for d in dense_docs + sparse_docs:
        merged[doc_key(d)] = d
    candidates = list(merged.values())

    if not candidates:
        return []

    # 3) reranker 점수 계산
    pairs = [[query, d.page_content] for d in candidates]
    scores = reranker.predict(pairs)  # numpy array 또는 list

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:TOP_K_FINAL]]
    return top_docs


# ======================
# 6. 프롬프트 & confidence 추론
# ======================

def build_prompt(passage: str, question: str, choices, wiki_ctx: str = "") -> str:
    """
    question + context summary 형태를 포함한 프롬프트
    wiki_ctx에는 RAG로 가져온 요약/컨텍스트가 들어감.
    """
    choices_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
    wiki_block = wiki_ctx.strip() if wiki_ctx else "없음"

    prompt = f"""너는 한국어 수능형 독해 문제를 푸는 AI 모델이다.
지문과 문항, 보기를 읽고 정답 번호를 고른다.
정답은 반드시 1, 2, 3, 4, 5 중 하나의 숫자만 출력한다.

question: {question}
context summary: 이 문제는 다음 내용을 이해해야 풀 수 있다.
{wiki_block}

[지문]
{passage}

[보기]
{choices_str}

정답 번호:"""
    return prompt


def infer_confidence(model, tokenizer, prompt: str):
    """
    마지막 토큰 logits에서 "1"~"5"에 해당하는 토큰 확률만 뽑아서
    softmax → 가장 큰 확률과 그 인덱스를 리턴.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # 마지막 토큰의 logits

    # "1"~"5" 토큰만 뽑아서 softmax
    token_ids = [tokenizer(str(i))["input_ids"][-1] for i in ["1", "2", "3", "4", "5"]]
    probs = F.softmax(logits[:, token_ids], dim=-1).squeeze(0)  # shape: (5,)

    best_idx = int(torch.argmax(probs))
    best_prob = float(probs[best_idx])
    pred_number = best_idx + 1

    return pred_number, best_prob


# ======================
# 7. RAG 쿼리 + 컨텍스트 구성
# ======================

def build_rag_query(paragraph: str, question: str) -> str:
    # 지문에서 앞부분 600자 정도만 쿼리에 붙이기 (EDA 기준 안정권)
    snippet = (paragraph or "")[:600]
    query = f"query: {question}\n\n{snippet}"
    return query


def build_wiki_context_from_docs(docs, max_chars: int = 1500) -> str:
    """
    rerank된 top docs를 받아서 위키 컨텍스트 문자열로 합치기
    """
    if not docs:
        return ""

    ctx_pieces = []
    total_len = 0

    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title", "")
        header = f"[문서{i}] {title}" if title else f"[문서{i}]"
        text = d.page_content.strip()
        chunk = f"{header}\n{text}"

        if total_len + len(chunk) > max_chars:
            break
        ctx_pieces.append(chunk)
        total_len += len(chunk)

    return "\n\n".join(ctx_pieces)


# ======================
# 8. 로그 출력 helper
# ======================

def log_case_header(idx, row_id):
    print("\n" + "=" * 80)
    print(f"[CASE] idx={idx} | id={row_id}")
    print("=" * 80)


def print_base_result(base_pred, base_conf):
    print(f"[BASE] pred={base_pred} | conf={base_conf:.4f}")


def print_rag_summary(docs, rag_pred, rag_conf):
    print(f"[RAG]  pred={rag_pred} | conf={rag_conf:.4f}")
    if not docs:
        print("[RAG]  ⚠️ retrieved_docs = 0 (검색 실패)")
        return

    titles = [d.metadata.get("title", "") for d in docs]
    print(f"[RAG]  retrieved_docs={len(docs)} | top_titles={titles[:3]}")


def print_decision(final_pred, base_conf, rag_conf):
    if rag_conf is None:
        print("[FINAL] KEEP BASE (RAG 실행 실패 또는 미사용)")
    elif rag_conf < base_conf:
        print(f"[FINAL] KEEP BASE (rag_conf < base_conf → {rag_conf:.4f} < {base_conf:.4f})")
    else:
        print(f"[FINAL] USE  RAG  (rag_conf >= base_conf → {rag_conf:.4f} ≥ {base_conf:.4f})")

    print(f"[FINAL] answer = {final_pred}")
    print("-" * 80)


# ======================
# 9. 메인: RAG 사용 문항들 로그 + 최종 submission 생성
# ======================

def main():
    set_seed(SEED)

    # 1) 모델 / 리트리버 / 리랭커 로딩
    model, tokenizer = load_unsloth_model()
    dense_retriever = load_dense_retriever()
    bm25_retriever = load_bm25_retriever()
    reranker = load_reranker()

    # 2) 데이터 로딩
    df = pd.read_csv(TEST_PATH)
    df = df.reset_index(drop=True)
    print(f"[DATA] test shape: {df.shape}")
    print(f"[DATA] columns: {list(df.columns)}")

    log_rows = []          # RAG 사용 케이스 + 위키 문서 로그
    submission_rows = []   # 모든 row에 대한 최종 예측

    # 3) 각 row마다 base confidence 확인 → 낮으면 RAG 검색 & 로그
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="log_rag_docs"):
        row_id = row["id"]
        log_case_header(idx, row_id)

        paragraph = row["paragraph"]
        problems_str = row["problems"]

        try:
            problems = ast.literal_eval(problems_str)
        except Exception as e:
            print(f"[WARN] problems parse failed (idx={idx}, id={row_id}): {e}")
            continue

        question = problems.get("question", "")
        choices = problems.get("choices", [])

        # (1) RAG 없이 base 추론
        prompt_base = build_prompt(
            passage=paragraph,
            question=question,
            choices=choices,
            wiki_ctx=""
        )
        base_pred, base_conf = infer_confidence(model, tokenizer, prompt_base)
        print_base_result(base_pred, base_conf)

        # 기본값: RAG 안 쓴다고 가정하고 시작
        rag_pred, rag_conf = None, None
        final_pred = base_pred

        # (2) confidence가 threshold 이상이면 RAG 안 씀
        if base_conf >= CONF_THRESHOLD:
            print(f"[SKIP RAG] base_conf >= threshold({CONF_THRESHOLD})")
            print_decision(final_pred, base_conf, rag_conf)

        else:
            # (3) RAG용 쿼리 만들고 위키 문서 가져오기
            print(f"[RAG TRIGGER] base_conf < threshold({CONF_THRESHOLD})")
            query = build_rag_query(paragraph, question)
            docs = hybrid_retrieve_with_rerank(
                query,
                dense_retriever,
                bm25_retriever,
                reranker,
            )

            # (4) RAG 컨텍스트로 다시 추론
            wiki_ctx = build_wiki_context_from_docs(docs)
            prompt_rag = build_prompt(
                passage=paragraph,
                question=question,
                choices=choices,
                wiki_ctx=wiki_ctx,
            )
            rag_pred, rag_conf = infer_confidence(model, tokenizer, prompt_rag)
            print_rag_summary(docs, rag_pred, rag_conf)

            # (5) 🔥 RAG confidence가 base보다 낮으면 base 답 유지
            if (rag_conf is None) or (rag_conf < base_conf):
                final_pred = base_pred
            else:
                final_pred = rag_pred

            print_decision(final_pred, base_conf, rag_conf)

            # (6) RAG 사용 케이스만 위키 문서 로그 남기기
            if not docs:
                # 문서 못 찾은 경우도 로그에 남김
                log_rows.append({
                    "id": row_id,
                    "row_idx": idx,
                    "base_pred": base_pred,
                    "base_conf": base_conf,
                    "rag_pred": rag_pred,
                    "rag_conf": rag_conf,
                    "final_pred": final_pred,
                    "doc_rank": None,
                    "doc_title": None,
                    "doc_source": None,
                    "doc_content_snippet": None,
                })
            else:
                for rank, d in enumerate(docs, start=1):
                    title = d.metadata.get("title", "")
                    source = d.metadata.get("source", "")
                    snippet = d.page_content[:300].replace("\n", " ")

                    log_rows.append({
                        "id": row_id,
                        "row_idx": idx,
                        "base_pred": base_pred,
                        "base_conf": base_conf,
                        "rag_pred": rag_pred,
                        "rag_conf": rag_conf,
                        "final_pred": final_pred,
                        "doc_rank": rank,
                        "doc_title": title,
                        "doc_source": source,
                        "doc_content_snippet": snippet,
                    })

        # 🔚 어떤 경우든 최종 예측은 submission_rows에 추가
        submission_rows.append({
            "id": row_id,
            "answer": int(final_pred),
        })

    # 4) RAG 로그 CSV 저장 (RAG 사용 케이스만)
    log_df = pd.DataFrame(log_rows)
    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)
    log_df.to_csv(LOG_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"[SAVE] RAG 사용 문항 로그 저장 완료: {LOG_CSV_PATH}")
    print(f"[SAVE] rows: {len(log_df)}")

    # 5) 최종 submission 저장 (전체 test row 대상)
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[SAVE] submission saved: {OUTPUT_CSV}")
    print(f"[SAVE] submission rows: {len(sub_df)}")


if __name__ == "__main__":
    main()
