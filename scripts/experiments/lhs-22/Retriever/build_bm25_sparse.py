#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import joblib
import torch
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer

from tokenizers_ko import KoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docstore", default="docstore.pt")
    ap.add_argument("--out_vocab", default="bm25_vocab.joblib")         # ✅ vectorizer 대신 vocab 저장
    ap.add_argument("--out_tf", default="bm25_tf_matrix.npz")
    ap.add_argument("--out_meta", default="bm25_meta.npz")
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_df", type=float, default=0.9)
    ap.add_argument("--ngram_max", type=int, default=1)
    args = ap.parse_args()

    docstore = torch.load(args.docstore, map_location="cpu")
    texts = [docstore[i]["text"].replace("\n", " ") for i in range(len(docstore))]

    tok = KoTokenizer()
    def analyzer(x: str):
        return tok(x)  # ✅ KoTokenizer는 callable

    vectorizer = CountVectorizer(
        analyzer=analyzer,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, args.ngram_max),
        lowercase=False,
    )

    tf = vectorizer.fit_transform(texts).tocsr()
    save_npz(args.out_tf, tf)

    # ✅ pickle 안전: vocabulary만 저장
    joblib.dump(vectorizer.vocabulary_, args.out_vocab)

    # BM25 meta
    doc_len = np.asarray(tf.sum(axis=1)).reshape(-1).astype(np.float32)
    avgdl = float(doc_len.mean())

    df = np.asarray((tf > 0).sum(axis=0)).reshape(-1).astype(np.float32)
    N = float(tf.shape[0])
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

    np.savez(
        args.out_meta,
        idf=idf,
        doc_len=doc_len,
        avgdl=np.array([avgdl], dtype=np.float32),
    )

    print(f"[OK] saved: {args.out_vocab}, {args.out_tf}, {args.out_meta}")
    print(f"docs={tf.shape[0]}, vocab={tf.shape[1]}, avgdl={avgdl:.2f}")

if __name__ == "__main__":
    main()
