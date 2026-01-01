# retriever_hybrid_faiss_bm25.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import joblib
import torch
import faiss
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

# =========================
# Kiwi tokenizer (BM25)
# =========================
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
                f"kiwipiepy import failed: {_KIWI_IMPORT_ERR}\n"
                "Install with: pip install kiwipiepy"
            )
        _KIWI = Kiwi()
    return _KIWI

def kiwi_tokenize(text: str):
    if not text:
        return []
    kiwi = _get_kiwi()
    toks = kiwi.tokenize(str(text))
    out = []
    for t in toks:
        w = getattr(t, "form", "")
        if w:
            w = w.strip()
            if w:
                out.append(w)
    return out


# =========================
# Hybrid Retriever
# =========================
class HybridWikiRetriever:
    def __init__(
        self,
        faiss_path: str,
        docstore_path: str,
        embed_model_name: str,
        device: str = "cpu",

        # BM25 artifacts (Kiwi-based)
        bm25_vocab_path: str = "bm25_vocab.joblib",
        bm25_tf_path: str = "bm25_tf.npz",
        bm25_meta_path: str = "bm25_meta.npz",

        reranker_name: str | None = None,
    ):
        self.device = device

        # ---------- Dense (FAISS) ----------
        self.index = faiss.read_index(faiss_path)
        self.docstore = torch.load(docstore_path, map_location="cpu")

        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(
            embed_model_name,
            device=device,
        )

        # dim sanity check (fail fast)
        test_emb = self.embedder.encode(
            ["dim check"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if test_emb.shape[1] != self.index.d:
            raise ValueError(
                f"FAISS dim mismatch: index.d={self.index.d} embed_dim={test_emb.shape[1]}"
            )

        # ---------- BM25 Sparse ----------
        self.bm25_vocab = joblib.load(bm25_vocab_path)
        self.bm25_tf = sparse.load_npz(bm25_tf_path).tocsr()
        meta = np.load(bm25_meta_path)

        self.bm25_df = meta["df"]
        self.bm25_doc_len = meta["doc_len"]
        self.bm25_avgdl = float(meta["avgdl"])
        self.bm25_N = int(meta["N"])

        # Vectorizer for QUERY ONLY (no fit)
        self.bm25_vectorizer = CountVectorizer(
            tokenizer=kiwi_tokenize,
            token_pattern=None,
            vocabulary=self.bm25_vocab,
            lowercase=False,
        )

        # ---------- Reranker (optional) ----------
        self.reranker = None
        if reranker_name:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_name, device=device)

    # =========================
    # Dense search
    # =========================
    def _dense_search(self, query: str, k: int):
        q_emb = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores, idxs = self.index.search(q_emb, k)
        return idxs[0], scores[0]

    # =========================
    # BM25 search (Kiwi)
    # =========================
    def _bm25_search(self, query: str, k: int):
        q_vec = self.bm25_vectorizer.transform([query])  # (1, V)
        q_vec = q_vec.tocsr()

        # BM25 score
        # score = sum_i idf_i * tf_{d,i} * (k1+1)/(tf_{d,i}+k1*(1-b+b*dl/avgdl))
        k1 = 1.5
        b = 0.75

        df = self.bm25_df
        N = self.bm25_N
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

        # sparse dot
        tf = self.bm25_tf[:, q_vec.indices]
        dl = self.bm25_doc_len

        numer = tf.multiply(idf[q_vec.indices])
        denom = tf + k1 * (1 - b + b * dl[:, None] / self.bm25_avgdl)
        score = (numer * (k1 + 1)) / denom

        scores = np.asarray(score.sum(axis=1)).ravel()
        topk = np.argsort(-scores)[:k]
        return topk, scores[topk]

    # =========================
    # Hybrid retrieve
    # =========================
    def retrieve(
        self,
        query: str,
        k_final: int = 5,
        k_dense: int = 30,
        k_sparse: int = 50,
        k_fuse: int = 80,
        use_rerank: bool = False,
    ):
        dense_ids, dense_scores = self._dense_search(query, k_dense)
        sparse_ids, sparse_scores = self._bm25_search(query, k_sparse)

        # union
        cand_ids = list(dict.fromkeys(
            list(dense_ids[:k_dense]) + list(sparse_ids[:k_sparse])
        ))[:k_fuse]

        # rerank (optional)
        if self.reranker:
            pairs = [
                (query, self.docstore["texts"][i])
                for i in cand_ids
            ]
            rr_scores = self.reranker.predict(pairs)
            order = np.argsort(-rr_scores)
            cand_ids = [cand_ids[i] for i in order]

        final_ids = cand_ids[:k_final]

        docs = []
        for i in final_ids:
            meta = self.docstore["metas"][i]
            docs.append({
                "id": int(i),
                "title": meta.get("title", ""),
                "text": self.docstore["texts"][i],
                "doc_id": meta.get("doc_id"),
                "chunk_id": meta.get("chunk_id"),
            })
        return docs


        return docs
