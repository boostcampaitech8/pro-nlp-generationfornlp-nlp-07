#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, gzip, json, re
import numpy as np
import torch
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import joblib
from tokenizers_ko import KoTokenizer

# --------- tokenizer (Kiwi 우선, 없으면 fallback) ---------
class KoTokenizer:
    def __init__(self, mode="kiwi"):
        self.mode = mode
        self._kiwi = None
        if mode == "kiwi":
            try:
                from kiwipiepy import Kiwi
                self._kiwi = Kiwi()
            except Exception:
                self.mode = "fallback"

    def __call__(self, text: str):
        text = text.strip()
        if not text:
            return []

        if self.mode == "kiwi" and self._kiwi is not None:
            # 수능형: 핵심 명사/동사/형용사 중심
            toks = []
            for t in self._kiwi.tokenize(text):
                # NNG/NNP: 일반/고유명사, VV/VA: 동사/형용사, XR: 어근
                if t.tag in ("NNG", "NNP", "VV", "VA", "XR"):
                    toks.append(t.form)
            return toks

        # fallback: 한글/영문/숫자 토큰 + 2-gram 보강(표현 변형 보완)
        words = re.findall(r"[가-힣A-Za-z0-9]+", text)
        bigrams = []
        for w in words:
            if len(w) >= 4:  # 너무 짧은 건 제외
                bigrams.extend([w[i:i+2] for i in range(len(w)-1)])
        return words + bigrams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="chunks.jsonl.gz")
    ap.add_argument("--docstore", default="docstore.pt")
    ap.add_argument("--vec_out", default="tfidf_vectorizer.joblib")
    ap.add_argument("--mat_out", default="tfidf_matrix.npz")
    ap.add_argument("--tokenizer", default="kiwi", choices=["kiwi", "fallback"])
    ap.add_argument("--max_docs", type=int, default=0)
    args = ap.parse_args()

    tok = KoTokenizer(mode=args.tokenizer)

    texts = []
    metas = []  # 반드시 text 포함 (rerank/hybrid에 필요)

    with gzip.open(args.chunks, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(tqdm(f, desc="Loading chunks")):
            j = json.loads(line)
            text = j.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            metas.append({
                "title": j.get("title", ""),
                "page_id": j.get("page_id", None),
                "chunk_id": j.get("chunk_id", None),
                "text": text,
            })
            texts.append(text)
            if args.max_docs and len(texts) >= args.max_docs:
                break

    print(f"[INFO] loaded chunks: {len(texts)}")

    # TF-IDF (BM25보다 구현/속도/안정성이 좋고, top-k 뽑기 쉬움)
    vectorizer = TfidfVectorizer(
        tokenizer=tok,
        lowercase=False,
        min_df=2,
        max_df=0.98,
        ngram_range=(1, 1),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(tqdm(texts, desc="Fitting TF-IDF"))

    # save
    joblib.dump(vectorizer, args.vec_out)
    save_npz(args.mat_out, X)
    torch.save(metas, args.docstore)

    print(f"[OK] docstore -> {args.docstore} (n={len(metas)})")
    print(f"[OK] vectorizer -> {args.vec_out}")
    print(f"[OK] matrix -> {args.mat_out}  shape={X.shape}")

if __name__ == "__main__":
    main()
