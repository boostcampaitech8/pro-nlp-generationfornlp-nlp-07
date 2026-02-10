# build_faiss_with_text.py
import argparse
import numpy as np
import torch
import faiss

def to_float32(emb):
    if isinstance(emb, torch.Tensor):
        return emb.detach().cpu().float().numpy()
    return np.asarray(emb, dtype="float32")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="chunk_embeddings.pt")
    ap.add_argument("--out_index", default="wiki.faiss")
    ap.add_argument("--out_meta", default="wiki_meta.pt")
    args = ap.parse_args()

    # PyTorch 2.6 대응
    data = torch.load(args.emb, weights_only=False)

    vecs = to_float32(data["embeddings"])
    metas = data["metas"]  # ★ text 포함 meta

    assert len(vecs) == len(metas), "embeddings/meta length mismatch"

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine (normalized)
    index.add(vecs)

    faiss.write_index(index, args.out_index)
    torch.save(metas, args.out_meta)

    print(f"[OK] FAISS index: {args.out_index} ntotal={index.ntotal}")
    print(f"[OK] Meta saved:  {args.out_meta} (with text)")
