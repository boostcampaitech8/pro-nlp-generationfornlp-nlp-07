#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid retriever (FAISS dense + TF-IDF sparse) with **proper fusion** and **fast sparse top-k**.

Fixes vs previous version:
- No set() + slice (which made final docs effectively random).
- Uses RRF (Reciprocal Rank Fusion) to combine sparse/dense rankings.
- Sparse scoring stays sparse (no .toarray() over all docs).
- Optional: return stable, deterministic ordering.
"""

from __future__ import annotations

import os
import joblib
import torch
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import faiss


def _topk_from_sparse_colvec(colvec, k: int):
    """
    colvec: scipy sparse shape (N, 1) or (N,)
    Return (idx, scores) sorted by score desc.
    """
    if colvec is None:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    v = colvec
    if hasattr(v, "tocoo"):
        v = v.tocoo()
        if v.nnz == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        rows = v.row
        data = v.data.astype(np.float32, copy=False)
    else:
        # dense fallback (shouldn't happen)
        data = np.asarray(v, dtype=np.float32).reshape(-1)
        rows = np.arange(len(data), dtype=np.int64)

    k = int(k)
    if k <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    if len(data) <= k:
        order = np.argsort(-data, kind="mergesort")
        return rows[order].astype(np.int64), data[order]

    part = np.argpartition(-data, k-1)[:k]
    part_scores = data[part]
    order = np.argsort(-part_scores, kind="mergesort")
    return rows[part][order].astype(np.int64), part_scores[order]


class HybridWikiRetriever:
    """
    Expected docstore format:
      - either a list[dict] where each dict has {"title","text",...}
      - or a dict with key "metas" or "docs"
    """

    def __init__(
        self,
        faiss_path: str,
        docstore_path: str,
        tfidf_vec_path: str,
        tfidf_mat_path: str,
        *,
        use_ip: int = 1,
        rrf_k0: int = 60,
        w_sparse: float = 1.0,
        w_dense: float = 1.0,
    ):
        self.index = faiss.read_index(faiss_path)
        self._use_ip = bool(use_ip)
        self.rrf_k0 = int(rrf_k0)
        self.w_sparse = float(w_sparse)
        self.w_dense = float(w_dense)

        ds = torch.load(docstore_path, map_location="cpu")
        if isinstance(ds, dict):
            self.metas = ds.get("metas") or ds.get("docs") or ds.get("items") or ds
        else:
            self.metas = ds

        self.vectorizer = joblib.load(tfidf_vec_path)
        self.tfidf = load_npz(tfidf_mat_path).tocsr()  # (N, V)

        if not isinstance(self.tfidf, csr_matrix):
            self.tfidf = self.tfidf.tocsr()

        self.N = int(self.tfidf.shape[0])

    def _dense_search(self, q_emb: np.ndarray, k: int):
        if q_emb is None:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        q = np.asarray(q_emb, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        D, I = self.index.search(q, int(k))
        return I.reshape(-1).astype(np.int64), D.reshape(-1).astype(np.float32)

    def _sparse_search(self, query: str, k: int):
        qv = self.vectorizer.transform([query])  # (1, V) sparse
        # result: (N, 1) sparse
        scores = (self.tfidf @ qv.T)
        idx, sc = _topk_from_sparse_colvec(scores, int(k))
        return idx, sc

    def _rrf_fuse(self, s_idx, d_idx, k_final: int):
        """
        RRF score = sum( w / (k0 + rank) )
        rank starts at 1
        """
        k0 = max(1, int(self.rrf_k0))
        score = {}

        # stable ranks
        for r, doc_id in enumerate(s_idx, start=1):
            score[int(doc_id)] = score.get(int(doc_id), 0.0) + self.w_sparse * (1.0 / (k0 + r))
        for r, doc_id in enumerate(d_idx, start=1):
            score[int(doc_id)] = score.get(int(doc_id), 0.0) + self.w_dense * (1.0 / (k0 + r))

        items = list(score.items())
        items.sort(key=lambda x: (-x[1], x[0]))  # deterministic
        return [doc_id for doc_id, _ in items[: int(k_final)]]

    def retrieve(
        self,
        query: str,
        *,
        k_sparse: int = 50,
        k_dense: int = 50,
        k_final: int = 5,
        q_emb: np.ndarray | None = None,
    ):
        # guard
        query = (query or "").strip()
        if not query:
            return []

        k_sparse = int(k_sparse)
        k_dense = int(k_dense)
        k_final = int(k_final)

        s_idx, _ = self._sparse_search(query, k_sparse) if k_sparse > 0 else (np.empty((0,), dtype=np.int64), None)
        d_idx, _ = self._dense_search(q_emb, k_dense) if k_dense > 0 else (np.empty((0,), dtype=np.int64), None)

        cand = self._rrf_fuse(s_idx.tolist(), d_idx.tolist(), k_final=k_final)

        docs = []
        for i in cand:
            if i < 0 or i >= self.N:
                continue
            m = self.metas[i]
            if isinstance(m, dict):
                title = m.get("title", "")
                text = m.get("text", m.get("chunk", ""))
            else:
                title = ""
                text = str(m)
            docs.append({"id": int(i), "title": title, "text": text})
        return docs


if __name__ == "__main__":
    # quick smoke (optional)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss", required=True)
    ap.add_argument("--docstore", required=True)
    ap.add_argument("--vec", required=True)
    ap.add_argument("--mat", required=True)
    ap.add_argument("--q", default="광합성은 무엇인가")
    args = ap.parse_args()
    r = HybridWikiRetriever(args.faiss, args.docstore, args.vec, args.mat)
    out = r.retrieve(args.q, k_sparse=50, k_dense=50, k_final=5, q_emb=None)
    print("hits:", len(out))
    for d in out[:3]:
        print("-", d["id"], d["title"], d["text"][:80].replace("\n", " "))
