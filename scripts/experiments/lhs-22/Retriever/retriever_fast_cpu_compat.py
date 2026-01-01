#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch
import faiss
import joblib

# ---- Compatibility shim for legacy TF-IDF joblib pickles ----
try:
    from tokenizers_ko import KoTokenizer
    import __main__
    setattr(__main__, 'KoTokenizer', KoTokenizer)
except Exception:
    pass
# --------------------------------------------------------------
from scipy.sparse import load_npz

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


def rrf_fusion(rank_lists, k=60, rrf_k=60, weights=None):
    """
    rank_lists: list of (doc_ids list)
    weights: list of float, same length as rank_lists
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)

    scores = {}
    for w, ids in zip(weights, rank_lists):
        for r, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (rrf_k + r + 1))

    # top-k
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [doc_id for doc_id, _ in items]


class HybridWikiRetriever:
    def __init__(
        self,
        faiss_index_path="wiki.faiss",
        docstore_path="docstore.pt",
        tfidf_vec_path="tfidf_vectorizer.joblib",
        tfidf_mat_path="tfidf_matrix.npz",
        dense_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-v2-m3",
        dense_device="cpu",
        rerank_device="cpu",
    ):
        self.index = faiss.read_index(faiss_index_path)

        # docstore: list[{"title","text",...}]  index id == list index
        self.docstore = torch.load(docstore_path, weights_only=False)

        # Keep retriever models off GPU by default (avoid VRAM contention with the LLM)
        self.dense_device = dense_device
        self.rerank_device = rerank_device
        self.dense = SentenceTransformer(dense_model, device=dense_device)
        self.dense.eval()

        self.vectorizer = joblib.load(tfidf_vec_path)
        self.X = load_npz(tfidf_mat_path)  # (N, V)


        self.reranker = None
        if reranker_model and CrossEncoder is not None:
            # CrossEncoder는 느리므로 topN에만 적용
            self.reranker = CrossEncoder(reranker_model, device=self.rerank_device)

    def retrieve(
        self,
        query: str,
        k_final=5,
        k_dense=30,
        k_sparse=50,
        k_fuse=80,
        use_rerank=True,
        rrf_k=60,
        fusion_weights=(1.0, 1.0),  # (dense, sparse)
    ):
        # --- Dense ---
        q_dense = self.dense.encode(
            "query: " + query,
            normalize_embeddings=True,
        ).astype("float32")
        D, I = self.index.search(q_dense[None, :], k_dense)
        dense_ids = I[0].tolist()

        # --- Sparse (TF-IDF cosine via dot product; vectorizer returns L2 normed by default? not always)
        # We'll use dot product on tfidf; that's fine as proxy ranking.
        q_vec = self.vectorizer.transform([query])
        scores = (self.X @ q_vec.T).toarray().ravel()  # (N,)
        if k_sparse < len(scores):
            top_idx = np.argpartition(-scores, k_sparse)[:k_sparse]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        else:
            top_idx = np.argsort(-scores)
        sparse_ids = top_idx.tolist()

        # --- Fusion (RRF) ---
        fused = rrf_fusion(
            [dense_ids, sparse_ids],
            k=k_fuse,
            rrf_k=rrf_k,
            weights=list(fusion_weights),
        )

        # --- Rerank (optional) ---
        if use_rerank and self.reranker_model:
            if self.reranker is None:
                self.reranker = CrossEncoder(self.reranker_model, device=self.rerank_device)
        if use_rerank and self.reranker is not None:
            pairs = []
            for doc_id in fused:
                txt = self.docstore[doc_id]["text"]
                pairs.append((query, txt))
            rr_scores = self.reranker.predict(pairs)  # higher is better
            order = np.argsort(-np.asarray(rr_scores))[:k_final]
            final_ids = [fused[i] for i in order]
        else:
            final_ids = fused[:k_final]

        return [self.docstore[i] for i in final_ids]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="wiki.faiss")
    ap.add_argument("--docstore", default="docstore.pt")
    ap.add_argument("--tfidf_vec", default="tfidf_vectorizer.joblib")
    ap.add_argument("--tfidf_mat", default="tfidf_matrix.npz")
    ap.add_argument("--dense_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--reranker_model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--k_final", type=int, default=5)
    args = ap.parse_args()

    r = HybridWikiRetriever(
        faiss_index_path=args.index,
        docstore_path=args.docstore,
        tfidf_vec_path=args.tfidf_vec,
        tfidf_mat_path=args.tfidf_mat,
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
    )

    q = "고려 시대의 토지 제도 특징은?"
    res = r.retrieve(q, k_final=args.k_final, use_rerank=True)
    for i, x in enumerate(res, 1):
        print(f"[{i}] {x['title']}")
        print(x["text"][:200].replace("\n", " "), "...\n")