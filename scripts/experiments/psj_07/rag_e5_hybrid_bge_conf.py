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
from langchain_core.retrievers import BaseRetriever

from sentence_transformers import CrossEncoder

# ✅ Kiwi BM25 추가
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi


# ======================
# 1. 설정
# ======================

MODEL_NAME = "NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_15"

TEST_PATH = "data/test/test.csv"
OUTPUT_CSV = "submission_rag_e5_hybrid_bge_conf.csv"

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 8
SEED = 42

# confidence threshold
CONF_THRESHOLD = 0.5

# vector store / docs
PERSIST_DIR = "scripts/experiments/psj_07/vectorstores/kowiki_e5_large"
DOCS_PKL = "scripts/experiments/psj_07/vectorstores/kowiki_docs.pkl"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

# hybrid 검색
TOP_K_DENSE = 6      # e5 dense에서 k개
TOP_K_BM25 = 6       # BM25에서 k개
TOP_K_FINAL = 4      # reranker 이후 최종 k개

# reranker (bge-reranker-large)
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"

NUM_MAP = {
    "①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
}


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
# 4. Kiwi 기반 BM25 Retriever 정의
# ======================

class KiwiBM25Retriever(BaseRetriever):
    """
    - kiwi로 문서/쿼리 모두 형태소 분석해서 토큰 리스트 생성
    - rank_bm25.BM25Okapi로 스코어 계산
    - BaseRetriever를 상속해서 retriever.invoke(query)도 동작
    """
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
        # 필요하면 품사 필터링(명사/동사/형용사 등) 추가 가능
        tokens = []
        for t in self.kiwi.tokenize(text):
            # 예시: 조사, 기호 등은 제외
            if t.tag.startswith(("NN", "VV", "VA", "MM", "XR")):
                tokens.append(t.form)
        # 혹시 너무 많이 걸러지면 그냥 tokens = [t.form for t in self.kiwi.tokenize(text)]
        if not tokens:  # 너무 비어버리면 fallback
            tokens = [t.form for t in self.kiwi.tokenize(text)]
        return tokens

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        q_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        # 상위 k개 인덱스
        top_idx = np.argsort(scores)[::-1][: self.k]
        return [self.docs[i] for i in top_idx]

    async def _aget_relevant_documents(self, query: str, *, run_manager=None):
        # async 버전은 그냥 sync 래핑
        return self._get_relevant_documents(query, run_manager=run_manager)


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
    LangChain 0.2+ 스타일에 맞게 retriever 호출:
    - VectorStoreRetriever: invoke(query)
    - KiwiBM25Retriever(BaseRetriever 상속): invoke(query) 가능
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
    scores = reranker.predict(pairs)  # numpy array

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:TOP_K_FINAL]]
    return top_docs


# ======================
# 6. 프롬프트 & confidence 추론
# ======================

def build_prompt(passage: str, question: str, choices, wiki_ctx: str = "") -> str:
    choices_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
    wiki_block = wiki_ctx.strip() if wiki_ctx else "없음"

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

정답 번호:"""
    return prompt


def infer_confidence(model, tokenizer, prompt: str):
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
# 7. RAG 컨텍스트 구성
# ======================

def build_rag_query(paragraph: str, question: str) -> str:
    snippet = (paragraph or "")[:200]
    query = f"query: {question}\n\n{snippet}"
    return query


def build_wiki_context(
    dense_retriever,
    bm25_retriever,
    reranker,
    paragraph: str,
    question: str,
    max_chars: int = 1500,
) -> str:
    query = build_rag_query(paragraph, question)
    docs = hybrid_retrieve_with_rerank(query, dense_retriever, bm25_retriever, reranker)

    if not docs:
        return ""

    ctx_pieces = []
    total_len = 0
    for i, d in enumerate(docs):
        title = d.metadata.get("title", "")
        header = f"[문서{i+1}] {title}" if title else f"[문서{i+1}]"
        text = d.page_content.strip()
        chunk = f"{header}\n{text}"

        if total_len + len(chunk) > max_chars:
            break
        ctx_pieces.append(chunk)
        total_len += len(chunk)

    return "\n\n".join(ctx_pieces)


# ======================
# 8. 한 문제 예측 (Adaptive RAG)
# ======================

def predict_one(
    model,
    tokenizer,
    dense_retriever,
    bm25_retriever,
    reranker,
    passage,
    question,
    choices,
) -> int:
    # 1) 먼저 RAG 없이 추론
    prompt_base = build_prompt(passage, question, choices, wiki_ctx="")
    base_pred, base_conf = infer_confidence(model, tokenizer, prompt_base)
    print(f"[BASE] pred={base_pred}, conf={base_conf:.3f}")

    # 2) confidence 낮으면 RAG 켜고 재추론
    if base_conf < CONF_THRESHOLD:
        wiki_ctx = build_wiki_context(
            dense_retriever, bm25_retriever, reranker, passage, question
        )
        prompt_rag = build_prompt(passage, question, choices, wiki_ctx=wiki_ctx)
        rag_pred, rag_conf = infer_confidence(model, tokenizer, prompt_rag)
        print(f"[RAG] used. pred={rag_pred}, conf={rag_conf:.3f}")
        return rag_pred

    # 3) 충분히 확신 있으면 그냥 base 사용
    return base_pred


# ======================
# 9. 메인 루프
# ======================

def main():
    set_seed(SEED)

    # 모델 / 리트리버 / 리랭커 로딩
    model, tokenizer = load_unsloth_model()
    dense_retriever = load_dense_retriever()
    bm25_retriever = load_bm25_retriever()   # ✅ Kiwi BM25
    reranker = load_reranker()

    # 데이터 로딩
    df = pd.read_csv(TEST_PATH)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"[DATA] test shape: {df.shape}")
    print(f"[DATA] columns: {list(df.columns)}")

    rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="inference"):
        paragraph = row["paragraph"]
        problems_str = row["problems"]

        try:
            problems = ast.literal_eval(problems_str)
        except Exception as e:
            print(f"[WARN] problems parse failed (idx={idx}, id={row.get('id')}): {e}")
            continue

        question = problems.get("question", "")
        choices = problems.get("choices", [])

        pred = predict_one(
            model,
            tokenizer,
            dense_retriever,
            bm25_retriever,
            reranker,
            passage=paragraph,
            question=question,
            choices=choices,
        )

        rows.append({
            "id": row["id"],
            "answer": int(pred),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[SAVE] submission saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
