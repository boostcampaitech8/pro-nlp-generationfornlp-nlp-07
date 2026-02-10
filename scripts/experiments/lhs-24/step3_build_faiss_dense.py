#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build FAISS index from chunk texts using SentenceTransformer.

Robust fixes:
- docstore items can be str OR dict (e.g., {"text": "...", ...}) -> safely extract text.
- docstore can be dict with key "texts", list of items, or mapping id->item.
- Optional E5-style prefixing:
  * passages encoded as: "passage: {text}"
  * enabled when model name contains "e5" AND --use_e5_prefix=1
- Device fallback: if cuda requested but unavailable -> cpu
- Consistent IP + normalization.
"""

from __future__ import annotations

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import faiss


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def _to_text(x) -> str:
    """
    docstore에서 나오는 item이 str일 수도 있고 dict일 수도 있음.
    embedding에 넣을 텍스트로 안전하게 변환.
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x

    if isinstance(x, dict):
        # 가장 흔한 케이스
        v = x.get("text")
        if isinstance(v, str):
            return v

        # 다른 후보 키들
        for k in ("chunk", "content", "body", "passage", "document", "raw"):
            v = x.get(k)
            if isinstance(v, str):
                return v

        # title + text 같이 있는 케이스면 합치기(선택)
        title = x.get("title")
        if isinstance(title, str) and isinstance(x.get("text"), str):
            return f"{title}\n{x['text']}"

        return str(x)

    # list/tuple 등이면 join 시도
    if isinstance(x, (list, tuple)):
        parts = []
        for it in x:
            t = _to_text(it)
            if t:
                parts.append(t)
        return "\n".join(parts)

    return str(x)


def _extract_texts(ds):
    """
    ds(torch.load 결과)에서 텍스트 리스트를 최대한 안전하게 추출.
    기대 포맷:
      - {"texts": [...]}  (가장 이상적)
      - list[str|dict]
      - dict[id -> str|dict]
      - 기타: ds 자체가 iterable
    """
    if isinstance(ds, dict):
        if "texts" in ds:
            texts = ds["texts"]
            if isinstance(texts, dict):
                # {"texts": {id: item}} 같은 케이스
                texts = list(texts.values())
            elif not isinstance(texts, (list, tuple)):
                # 단일 값이면 리스트로
                texts = [texts]
            return list(texts)

        # id->item 매핑인 케이스로 간주
        return list(ds.values())

    if isinstance(ds, (list, tuple)):
        return list(ds)

    # 기타 iterable
    try:
        return list(ds)
    except Exception:
        return [ds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docstore", required=True, help="torch file (from step2) containing texts")
    ap.add_argument("--out_faiss", required=True, help="output faiss index file path")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--use_ip", type=int, default=1, help="1: inner product (normalized vectors)")
    ap.add_argument("--use_e5_prefix", type=int, default=1, help="1: apply e5 passage prefix when model contains 'e5'")
    args = ap.parse_args()

    # device fallback
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    ds = torch.load(args.docstore, map_location="cpu")
    raw_texts = _extract_texts(ds)

    # convert items -> str
    texts = [_to_text(x) for x in raw_texts]
    # optional: empty 제거(완전 비면 검색/인덱스 품질 떨어짐)
    # 다만 docstore와 인덱스의 row align이 중요하면 제거하지 말아야 함.
    # 여기서는 align 유지 위해 제거하지 않음.

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=device)

    use_ip = bool(args.use_ip)
    use_prefix = bool(args.use_e5_prefix) and ("e5" in args.model.lower())

    embs = []
    for i in tqdm(range(0, len(texts), args.batch), desc="embedding"):
        batch = texts[i:i + args.batch]

        # E5 passage prefix
        if use_prefix:
            batch = ["passage: " + (t or "") for t in batch]
        else:
            batch = [(t or "") for t in batch]

        e = model.encode(
            batch,
            batch_size=min(len(batch), args.batch),
            convert_to_numpy=True,
            normalize_embeddings=use_ip,  # IP면 normalize_embeddings=True 권장
            show_progress_bar=False,
        )
        embs.append(e)

    X = np.vstack(embs).astype("float32", copy=False)

    # Ensure normalization if IP
    if use_ip:
        # sentence-transformers normalize_embeddings가 켜져 있어도 안전 체크
        mean_norm = float(np.linalg.norm(X, axis=1).mean())
        if not np.isfinite(mean_norm) or abs(mean_norm - 1.0) > 1e-2:
            X = l2_normalize(X).astype("float32", copy=False)

    dim = int(X.shape[1])
    index = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
    index.add(X)
    faiss.write_index(index, args.out_faiss)

    print("[DONE] wrote:", args.out_faiss, "N=", index.ntotal, "dim=", dim, "use_prefix=", int(use_prefix), "device=", device)


if __name__ == "__main__":
    main()
