#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="chunks.jsonl.gz")
    ap.add_argument("--out", default="chunk_embeddings.pt")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max_chunks", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    model = SentenceTransformer(args.model, device=device)
    model.eval()

    all_embs = []
    all_metas = []
    buf = []

    with gzip.open(args.chunks, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(tqdm(f, desc="Embedding chunks")):
            j = json.loads(line)

            text = j.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            # e5 passage format
            buf.append("passage: " + text)

            # ★ text 포함 meta (중요)
            all_metas.append({
                "title": j.get("title", ""),
                "page_id": j.get("page_id", None),
                "chunk_id": j.get("chunk_id", None),
                "text": text,
            })

            if len(buf) == args.batch:
                with torch.no_grad():
                    embs = model.encode(
                        buf,
                        batch_size=args.batch,
                        normalize_embeddings=True,
                        convert_to_tensor=True
                    )
                    if args.fp16:
                        embs = embs.half()
                    all_embs.append(embs.cpu())
                buf = []

            if args.max_chunks and (i + 1) >= args.max_chunks:
                break

        # flush
        if buf:
            with torch.no_grad():
                embs = model.encode(
                    buf,
                    batch_size=args.batch,
                    normalize_embeddings=True,
                    convert_to_tensor=True
                )
                if args.fp16:
                    embs = embs.half()
                all_embs.append(embs.cpu())

    all_embs = torch.cat(all_embs, dim=0)

    torch.save(
        {
            "embeddings": all_embs,
            "metas": all_metas,   # ★ text 포함
            "model": args.model,
            "normalized": True,
        },
        args.out
    )

    print(f"[OK] saved embeddings: {all_embs.shape}")
    print(f"[OK] saved metas: {len(all_metas)} (with text)")
    print(f"[OK] output -> {args.out}")
