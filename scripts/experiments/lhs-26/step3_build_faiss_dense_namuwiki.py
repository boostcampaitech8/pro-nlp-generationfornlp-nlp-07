#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build FAISS dense index from docstore.pt texts using SentenceTransformer.

Output:
- faiss_index.index

Example:
  python3 step3_build_faiss_dense_namuwiki.py \
    --docstore docstore_namuwiki.pt \
    --out_faiss faiss_index.index \
    --model intfloat/multilingual-e5-base \
    --batch_size 128 --device cuda
"""

import argparse
import os
import numpy as np
import torch
import faiss
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise SystemExit(
        "Missing dependency: sentence-transformers. Install with: pip install sentence-transformers\n"
        f"Original error: {e}"
    )


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def pick_text(m):
    if isinstance(m, dict):
        return (m.get("text") or m.get("chunk") or "")
    return str(m or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docstore", default="docstore.pt")
    ap.add_argument("--out_faiss", default="faiss_index.index")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_ip", type=int, default=1)
    ap.add_argument("--use_e5_prefix", type=int, default=1, help="1: add 'passage:' prefix for e5 models")

    ap.add_argument("--max_docs", type=int, default=0, help="0=all docs")
    ap.add_argument("--max_chars", type=int, default=1500, help="truncate for embedding only")

    args = ap.parse_args()

    # device fallback
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    metas = torch.load(args.docstore, map_location="cpu")
    if isinstance(metas, dict) and "texts" in metas:
        metas = metas["texts"]
    if isinstance(metas, dict):
        metas = [metas[k] for k in sorted(metas.keys())]

    if not isinstance(metas, list) or not metas:
        raise SystemExit("docstore is empty or not a list. Expected torch.save(list[dict]).")

    N = len(metas)
    cap = int(args.max_docs) if int(args.max_docs) > 0 else N
    cap = min(cap, N)

    model = SentenceTransformer(args.model, device=device)

    use_prefix = int(args.use_e5_prefix) == 1 and ("e5" in args.model.lower())
    bs = max(1, int(args.batch_size))

    embs = []
    for i in tqdm(range(0, cap, bs), desc="Encoding passages"):
        batch = metas[i:i+bs]
        texts = []
        for m in batch:
            t = pick_text(m)
            if not t:
                t = ""
            if args.max_chars > 0 and len(t) > args.max_chars:
                t = t[: args.max_chars]
            if use_prefix:
                t = "passage: " + t
            texts.append(t)
        X = model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        X = np.asarray(X, dtype="float32")
        embs.append(X)

    X = np.concatenate(embs, axis=0)
    X = l2_normalize(X).astype("float32", copy=False)

    dim = int(X.shape[1])
    index = faiss.IndexFlatIP(dim) if int(args.use_ip) == 1 else faiss.IndexFlatL2(dim)
    index.add(X)
    faiss.write_index(index, args.out_faiss)

    print("[DONE] wrote:", args.out_faiss, "N=", index.ntotal, "dim=", dim, "use_prefix=", int(use_prefix), "device=", device)


if __name__ == "__main__":
    main()
