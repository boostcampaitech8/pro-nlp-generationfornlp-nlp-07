#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_build_bm25_sparse_fast.py  (Kiwi tokenizer version)

Build BM25 sparse artifacts from docstore.pt:
- bm25_vocab.joblib : vocabulary dict (token -> col_id)
- bm25_tf.npz       : term-frequency CSR matrix (N_docs, V)
- bm25_meta.npz     : df, doc_len, avgdl, N

This version uses Kiwi morphological tokenizer.
"""

import argparse
import numpy as np
import torch
import joblib
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

# ---- Kiwi tokenizer (module-level; not local closure) ----
try:
    from kiwipiepy import Kiwi
except Exception as e:
    Kiwi = None
    _KIWI_IMPORT_ERR = e

_KIWI = None

def _get_kiwi():
    global _KIWI
    if _KIWI is None:
        if Kiwi is None:
            raise RuntimeError(
                f"kiwipiepy is not installed or failed to import: {_KIWI_IMPORT_ERR}\n"
                "Install: pip install kiwipiepy"
            )
        _KIWI = Kiwi()
    return _KIWI

def kiwi_tokenize(text: str):
    """
    Return list[str] tokens for CountVectorizer.
    - Uses token.form (surface form)
    - Drops spaces / empty
    """
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    kiwi = _get_kiwi()
    toks = kiwi.tokenize(s)
    out = []
    for t in toks:
        # kiwipiepy token fields: form, tag, start, len ...
        w = getattr(t, "form", "")
        if not w:
            continue
        w = w.strip()
        if not w:
            continue
        out.append(w)
    return out
# ---------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docstore", required=True)
    ap.add_argument("--out_vocab", default="bm25_vocab.joblib")
    ap.add_argument("--out_tf", default="bm25_tf.npz")
    ap.add_argument("--out_meta", default="bm25_meta.npz")
    ap.add_argument("--max_features", type=int, default=500000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--ngram_max", type=int, default=2)
    args = ap.parse_args()

    ds = torch.load(args.docstore, map_location="cpu")
    texts = ds["texts"]

    # IMPORTANT:
    # - token_pattern must be None when tokenizer is provided
    # - analyzer stays 'word' and uses tokenizer output
    vectorizer = CountVectorizer(
        tokenizer=kiwi_tokenize,
        token_pattern=None,
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_range=(1, args.ngram_max),
        lowercase=False,  # Korean: keep as-is; Kiwi outputs normalized forms anyway
    )

    tf = vectorizer.fit_transform(texts).tocsr()  # (N, V)
    df = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.int32)
    doc_len = np.asarray(tf.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(doc_len.mean()) if doc_len.size else 0.0
    N = tf.shape[0]

    # Save vocabulary dict only (stable / pickle-safe)
    joblib.dump(vectorizer.vocabulary_, args.out_vocab)
    sparse.save_npz(args.out_tf, tf)
    np.savez(args.out_meta, df=df, doc_len=doc_len, avgdl=avgdl, N=N)

    print("[DONE] bm25 vocab/tf/meta saved (Kiwi tokenizer)")
    print("  N:", N, "V:", tf.shape[1], "avgdl:", avgdl)
    print("  vocab:", args.out_vocab)
    print("  tf:   ", args.out_tf)
    print("  meta: ", args.out_meta)


if __name__ == "__main__":
    main()
