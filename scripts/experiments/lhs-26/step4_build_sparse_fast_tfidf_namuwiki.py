#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build TF-IDF vectorizer + sparse matrix from docstore.pt.

Output files match your retriever:
- tfidf_vectorizer.joblib
- tfidf_matrix.npz  (scipy sparse CSR; keys will be indices/indptr/data/shape/format)
- docstore.pt is reused (not re-written unless --docstore_out is provided)

Example:
  python3 step4_build_sparse_fast_tfidf_namuwiki.py \
    --docstore docstore_namuwiki.pt \
    --vec_out tfidf_vectorizer.joblib \
    --mat_out tfidf_matrix.npz
"""

import argparse
import os
import re

import joblib
import numpy as np
import torch
from tqdm import tqdm
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


_tok_re = re.compile(r"[가-힣A-Za-z0-9]+")

def _tokenize(s: str):
    return [t.lower() for t in _tok_re.findall(s or "") if len(t) >= 2]

def _pick_text(m):
    if isinstance(m, dict):
        return (m.get("text") or m.get("chunk") or "")
    return str(m or "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docstore", default="docstore.pt")
    ap.add_argument("--docstore_out", default="", help="optional: rewrite docstore to this path")
    ap.add_argument("--vec_out", default="tfidf_vectorizer.joblib")
    ap.add_argument("--mat_out", default="tfidf_matrix.npz")

    ap.add_argument("--max_chars", type=int, default=900, help="truncate for TF-IDF only")
    ap.add_argument("--max_docs", type=int, default=0, help="0=all docs")
    ap.add_argument("--max_features", type=int, default=250000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_df", type=float, default=0.98)
    ap.add_argument("--ngram_max", type=int, default=1)

    args = ap.parse_args()

    metas = torch.load(args.docstore, map_location="cpu")
    if isinstance(metas, dict) and "texts" in metas:
        metas = metas["texts"]
    if isinstance(metas, dict):
        # id->item mapping
        metas = [metas[k] for k in sorted(metas.keys())]

    if not isinstance(metas, list) or not metas:
        raise SystemExit("docstore is empty or not a list. Expected torch.save(list[dict]).")

    texts = []
    N = len(metas)
    cap = int(args.max_docs) if int(args.max_docs) > 0 else N
    cap = min(cap, N)

    for m in tqdm(metas[:cap], desc="Loading doc texts"):
        t = _pick_text(m)
        if not t:
            texts.append("")
            continue
        if args.max_chars > 0 and len(t) > args.max_chars:
            t = t[: args.max_chars]
        texts.append(t)

    vectorizer = TfidfVectorizer(
        tokenizer=_tokenize,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        strip_accents=None,
        max_features=int(args.max_features),
        min_df=int(args.min_df),
        max_df=float(args.max_df),
        ngram_range=(1, int(args.ngram_max)),
        dtype=np.float32,
    )

    X = vectorizer.fit_transform(tqdm(texts, desc="TF-IDF fitting"))

    os.makedirs(os.path.dirname(args.vec_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.mat_out) or ".", exist_ok=True)

    joblib.dump(vectorizer, args.vec_out)
    save_npz(args.mat_out, X)

    if args.docstore_out:
        os.makedirs(os.path.dirname(args.docstore_out) or ".", exist_ok=True)
        torch.save(metas[:cap], args.docstore_out)
        print(f"[OK] docstore_out: {args.docstore_out} (n={cap})")
    else:
        print(f"[OK] docstore: {args.docstore} (n={cap})")

    print(f"[OK] tfidf vec: {args.vec_out}")
    print(f"[OK] tfidf mat: {args.mat_out} shape={X.shape} nnz={X.nnz}")
    print(f"[OK] max_chars={args.max_chars} max_features={args.max_features} min_df={args.min_df} ngram_max={args.ngram_max}")


if __name__ == "__main__":
    main()
