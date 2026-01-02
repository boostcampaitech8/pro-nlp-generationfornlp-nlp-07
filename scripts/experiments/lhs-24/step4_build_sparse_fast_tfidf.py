#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast TF-IDF builder for large Korean wiki chunks.

Goal: keep retrieval quality stable while making TF-IDF build fast.
- No Kiwi (too slow at 1M+ docs)
- Regex tokenization
- Truncate document text for TF-IDF only
- Limit max_features
"""

import argparse, gzip, json, re
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import joblib
import numpy as np


# fast token pattern: Korean/English/number sequences
TOKEN_PATTERN = r"(?u)[가-힣A-Za-z0-9]{2,}"


def pick_text(j: dict) -> str:
    # support multiple keys (kowiki chunks often use chunk/content)
    return (j.get("text") or j.get("chunk") or j.get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="jsonl.gz of chunks")
    ap.add_argument("--docstore", default="docstore.pt")
    ap.add_argument("--vec_out", default="tfidf_vectorizer.joblib")
    ap.add_argument("--mat_out", default="tfidf_matrix.npz")

    # speed/quality knobs (safe defaults)
    ap.add_argument("--max_chars", type=int, default=2000, help="truncate each doc for TF-IDF")
    ap.add_argument("--max_features", type=int, default=500000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_df", type=float, default=0.98)
    ap.add_argument("--ngram_max", type=int, default=1)  # 1 keeps it fast; 2 increases cost a lot
    ap.add_argument("--max_docs", type=int, default=0, help="0 = all docs")
    args = ap.parse_args()

    texts, metas = [], []

    with gzip.open(args.chunks, "rt", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Loading chunks"):
            j = json.loads(line)
            text = pick_text(j)
            if not text:
                continue

            # truncate only for TF-IDF speed (keep full text in docstore for prompting)
            text_tfidf = text[: args.max_chars]

            metas.append({
                "title": j.get("title", ""),
                "page_id": j.get("page_id"),
                "chunk_id": j.get("chunk_id"),
                "text": text,                 # full text stored
            })
            texts.append(text_tfidf)          # truncated text used for tf-idf

            if args.max_docs and len(texts) >= args.max_docs:
                break

    # fast TF-IDF
    vectorizer = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN,   # fast regex tokenizer
        lowercase=False,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    X = vectorizer.fit_transform(tqdm(texts, desc="TF-IDF fitting"))

    joblib.dump(vectorizer, args.vec_out)
    save_npz(args.mat_out, X)
    torch.save(metas, args.docstore)

    print(f"[OK] docstore: {args.docstore} (n={len(metas)})")
    print(f"[OK] tfidf vec: {args.vec_out}")
    print(f"[OK] tfidf mat: {args.mat_out} shape={X.shape} nnz={X.nnz}")
    print(f"[OK] max_chars={args.max_chars} max_features={args.max_features} min_df={args.min_df} ngram_max={args.ngram_max}")


if __name__ == "__main__":
    main()
