# retriever_cpu_bm25.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
import joblib
import torch
import faiss
from scipy.sparse import load_npz
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import CountVectorizer

from tokenizers_ko import KoTokenizer


def _rrf_fuse(ranked_lists, weights, k0=60, k_fuse=200):
    scores = {}
    for lst, w in zip(ranked_lists, weights):
        for r, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + float(w) / float(k0 + r + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in fused[:k_fuse]]


class HybridWikiRetriever:
    def __init__(
        self,
        faiss_path: str = "wiki.faiss",
        docstore_path: str = "docstore.pt",
        embed_model_name: str = "intfloat/multilingual-e5-base",
        dense_batch_size: int = 64,

        # ✅ vocab 저장 포맷
        bm25_vocab_path: str = "bm25_vocab.joblib",
        bm25_tf_path: str = "bm25_tf_matrix.npz",
        bm25_meta_path: str = "bm25_meta.npz",
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,

        reranker_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ):
        self.device = device
        self.dense_batch_size = dense_batch_size

        self.docstore = torch.load(docstore_path, map_location="cpu")
        self.N = len(self.docstore)

        self.index = faiss.read_index(faiss_path)
        self.embedder = SentenceTransformer(embed_model_name, device=device)

        # ---- BM25 ----
        vocab = joblib.load(bm25_vocab_path)  # dict term->id

        self._tok = KoTokenizer()

        # ✅ analyzer는 코드에만 존재 (저장 안 함)
        def analyzer(x: str):
            return self._tok(x)

        self.bm25_vectorizer = CountVectorizer(
            vocabulary=vocab,
            analyzer=analyzer,
            lowercase=False,
        )

        tf_csr = load_npz(bm25_tf_path).tocsr()
        self.bm25_tf = tf_csr.tocsc()

        meta = np.load(bm25_meta_path)
        self.bm25_idf = meta["idf"].astype(np.float32)
        self.bm25_doclen = meta["doc_len"].astype(np.float32)
        self.bm25_avgdl = float(meta["avgdl"][0])
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)

        # reranker lazy
        self.reranker_name = reranker_name
        self.reranker_model = None

    def _ensure_reranker(self):
        if self.reranker_model is None:
            self.reranker_model = CrossEncoder(self.reranker_name, device=self.device)

    def _dense_search(self, query: str, k_dense: int) -> List[int]:
        q_emb = self.embedder.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        _, I = self.index.search(q_emb, k_dense)
        return I[0].tolist()

    def _bm25_search(self, query: str, k_sparse: int) -> List[int]:
        q_tf = self.bm25_vectorizer.transform([query]).tocsr()
        q_indices = q_tf.indices
        q_counts = q_tf.data

        scores = np.zeros(self.N, dtype=np.float32)
        k1 = self.bm25_k1
        b = self.bm25_b
        avgdl = self.bm25_avgdl
        dl = self.bm25_doclen

        for term_id, qcnt in zip(q_indices, q_counts):
            col = self.bm25_tf.getcol(term_id)
            if col.nnz == 0:
                continue
            doc_ids = col.indices
            tf_vals = col.data.astype(np.float32)

            denom = tf_vals + k1 * (1.0 - b + b * (dl[doc_ids] / avgdl))
            term_score = self.bm25_idf[term_id] * (tf_vals * (k1 + 1.0) / denom) * float(qcnt)
            scores[doc_ids] += term_score

        if k_sparse >= self.N:
            return np.argsort(-scores).tolist()
        return np.argpartition(-scores, k_sparse)[:k_sparse].tolist()

    def retrieve(
        self,
        query: str,
        k_final: int = 20,
        k_dense: int = 120,
        k_sparse: int = 300,
        k_fuse: int = 200,
        use_rerank: bool = False,
        w_sparse: float = 0.6,
        w_dense: float = 0.4,
        rerank_topn: int = 60,
        rrf_k0: int = 60,
    ) -> List[Dict[str, Any]]:
        dense_ids = self._dense_search(query, k_dense=k_dense)
        sparse_ids = self._bm25_search(query, k_sparse=k_sparse)

        fused_ids = _rrf_fuse(
            ranked_lists=[sparse_ids, dense_ids],
            weights=[w_sparse, w_dense],
            k0=rrf_k0,
            k_fuse=k_fuse,
        )

        if use_rerank:
            self._ensure_reranker()
            top_ids = fused_ids[:max(1, min(rerank_topn, len(fused_ids)))]
            pairs = [[query, self.docstore[did]["text"]] for did in top_ids]
            rr_scores = np.asarray(self.reranker_model.predict(pairs), dtype=np.float32)
            order = np.argsort(-rr_scores).tolist()
            reranked = [top_ids[i] for i in order]
            seen = set(reranked)
            fused_ids = reranked + [x for x in fused_ids if x not in seen]

        final_ids = fused_ids[:k_final]
        return [self.docstore[i] for i in final_ids]
