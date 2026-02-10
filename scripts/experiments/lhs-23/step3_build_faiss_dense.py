#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, torch, numpy as np
from tqdm import tqdm
import faiss

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docstore", required=True)
    ap.add_argument("--out_faiss", required=True)
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--use_ip", type=int, default=1, help="1: inner product (with normalized vectors)")
    args = ap.parse_args()

    ds = torch.load(args.docstore, map_location="cpu")
    texts = ds["texts"]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device=args.device)

    embs = []
    for i in tqdm(range(0, len(texts), args.batch), desc="embedding"):
        batch = texts[i:i+args.batch]
        # e5류는 query/passage prefix가 도움이 됨. (선택)
        # batch = ["passage: " + t for t in batch]
        e = model.encode(batch, batch_size=len(batch), convert_to_numpy=True, normalize_embeddings=bool(args.use_ip))
        embs.append(e)

    X = np.vstack(embs).astype("float32")
    if args.use_ip and not np.allclose(np.linalg.norm(X, axis=1).mean(), 1.0, atol=1e-2):
        X = l2_normalize(X)

    dim = X.shape[1]
    if args.use_ip:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(X)
    faiss.write_index(index, args.out_faiss)
    print("[DONE] wrote:", args.out_faiss, "N=", index.ntotal, "dim=", dim)

if __name__ == "__main__":
    main()
