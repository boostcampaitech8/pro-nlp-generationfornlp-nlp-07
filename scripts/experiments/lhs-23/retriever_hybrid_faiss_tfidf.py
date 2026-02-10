import joblib, torch
import numpy as np
from scipy.sparse import load_npz
import faiss

class HybridWikiRetriever:
    def __init__(
        self,
        faiss_path,
        docstore_path,
        tfidf_vec_path,
        tfidf_mat_path,
    ):
        self.index = faiss.read_index(faiss_path)
        self.metas = torch.load(docstore_path)
        self.vectorizer = joblib.load(tfidf_vec_path)
        self.tfidf = load_npz(tfidf_mat_path)

    def _sparse_search(self, query, k):
        qv = self.vectorizer.transform([query])
        scores = (self.tfidf @ qv.T).toarray().squeeze(1)
        idx = np.argsort(-scores)[:k]
        return idx, scores[idx]

    def _dense_search(self, q_emb, k):
        D, I = self.index.search(q_emb, k)
        return I[0], D[0]

    def retrieve(
        self,
        query,
        q_emb=None,
        k_sparse=50,
        k_dense=30,
        k_final=5,
    ):
        s_idx, s_score = self._sparse_search(query, k_sparse)

        if q_emb is not None:
            d_idx, d_score = self._dense_search(q_emb, k_dense)
            cand = list(set(s_idx.tolist()) | set(d_idx.tolist()))
        else:
            cand = s_idx.tolist()

        # 간단한 late-fusion: sparse score 우선
        cand = cand[:k_final]

        docs = []
        for i in cand:
            m = self.metas[i]
            docs.append({
                "id": int(i),
                "title": m.get("title", ""),
                "text": m.get("text", ""),
            })
        return docs
