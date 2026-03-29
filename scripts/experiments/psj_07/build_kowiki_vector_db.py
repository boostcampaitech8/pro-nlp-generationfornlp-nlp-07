import os
import pickle
from datasets import load_dataset
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ======================
# 설정
# ======================

DATASET_NAME = "NLP-07-ODQA/kowiki-cleaned"
SPLIT = "train"

# psj_07 폴더 안에 저장
PERSIST_DIR = "scripts/experiments/psj_07/vectorstores/kowiki_e5_large"
DOCS_PKL    = "scripts/experiments/psj_07/vectorstores/kowiki_docs.pkl"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

# chunk 크기 (조금 크게 잡아서 chunk 개수 줄이기)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# 🔥 전체 위키 중에서 text_length 기준 상위 문서만 사용
MAX_ROWS   = 80_000      # 사용할 최대 문서 수
MAX_CHUNKS = 500_000     # 생성할 최대 chunk 수 (안전 장치)


def load_kowiki():
    """
    HF에서 위키 데이터셋 로드 후,
    text_length 기준 내림차순 정렬 → 상위 MAX_ROWS 개만 사용.
    """
    print(f"[DATA] loading {DATASET_NAME} ({SPLIT})")
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    print("[DATA] original dataset:", ds)

    # text_length 기준으로 길이 긴 문서부터 정렬
    if "text_length" in ds.column_names:
        print("[DATA] sorting by text_length (desc)")
        ds = ds.sort("text_length", reverse=True)
    else:
        print("[WARN] 'text_length' column not found. Using original order.")

    # 상위 MAX_ROWS 개만 사용
    if len(ds) > MAX_ROWS:
        print(f"[DATA] selecting top {MAX_ROWS} rows (out of {len(ds)})")
        ds = ds.select(range(MAX_ROWS))

    print("[DATA] final dataset after selection:", ds)
    return ds


def build_documents(ds):
    """
    content를 CHUNK_SIZE 단위로 잘라서 LangChain Document 리스트로 변환.
    전체 chunk 개수가 MAX_CHUNKS를 넘으면 중단.
    """
    print("[DOC] building Document list + chunking ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    docs = []
    for row in tqdm(ds, desc="chunking"):
        content = row.get("content", "")
        title = row.get("title", "")
        page_id = row.get("page_id", None)
        text_len = row.get("text_length", None)

        if not isinstance(content, str) or not content.strip():
            continue

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=f"passage: {chunk.strip()}",
                    metadata={
                        "title": title,
                        "page_id": page_id,
                        "text_length": text_len,
                        "chunk_id": i,
                    },
                )
            )

            # 너무 많이 쌓이면 중단
            if len(docs) >= MAX_CHUNKS:
                print(f"[DOC] reached MAX_CHUNKS={MAX_CHUNKS}, stop chunking.")
                print(f"[DOC] total chunks = {len(docs)}")
                return docs

    print(f"[DOC] total chunks = {len(docs)}")
    return docs


def build_vectorstore(docs):
    """
    e5-large-instruct로 임베딩해서 Chroma 벡터DB에 저장.
    """
    print(f"[EMB] loading embedding model: {EMBED_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[VS] building & persisting Chroma at {PERSIST_DIR}")
    os.makedirs(PERSIST_DIR, exist_ok=True)

    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vs.persist()
    print("[VS] Chroma saved.")


def save_docs_for_bm25(docs):
    """
    BM25Retriever용으로 동일한 Document 리스트를 pickle로 저장.
    """
    os.makedirs(os.path.dirname(DOCS_PKL), exist_ok=True)
    with open(DOCS_PKL, "wb") as f:
        pickle.dump(docs, f)
    print(f"[DOC] Documents saved for BM25: {DOCS_PKL}")


def main():
    ds = load_kowiki()
    docs = build_documents(ds)
    print(f"[DOC] final docs count = {len(docs)}")

    build_vectorstore(docs)
    save_docs_for_bm25(docs)

    print("[DONE] vectorstore + docs build complete.")


if __name__ == "__main__":
    main()
