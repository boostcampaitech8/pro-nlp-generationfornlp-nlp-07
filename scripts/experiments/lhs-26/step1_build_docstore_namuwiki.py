#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build docstore.pt from heegyu/namuwiki-sentences by chunking sentences.

Why chunk?
- The dataset is sentence-level; we group contiguous sentences to restore context.
- Output format matches your HybridWikiRetriever expectations: torch.save(list[dict]) where
  each dict has {"title": ..., "text": ...} and the implicit doc_id is its list index.

Notes / safety:
- Uses datasets streaming by default (memory safe).
- You MUST control the size with --max_docs / --max_sentences to avoid creating multi-GB files.
- Chunks are created per (title, pi) and ordered by si when available.

Example:
  python3 build_docstore_namuwiki_chunks.py \
    --out docstore_namuwiki.pt \
    --chunk_sents 6 --chunk_chars 1200 \
    --max_docs 1500000
"""

import argparse
import math
import os
from collections import defaultdict

import torch

try:
    from datasets import load_dataset
except Exception as e:
    raise SystemExit(
        "Missing dependency: datasets. Install with: pip install datasets\n"
        f"Original error: {e}"
    )


def _norm_ws(s: str) -> str:
    return " ".join((s or "").replace("\u00a0", " ").split()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="heegyu/namuwiki-sentences")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="docstore.pt")

    ap.add_argument("--chunk_sents", type=int, default=6, help="sentences per chunk (soft cap)")
    ap.add_argument("--chunk_chars", type=int, default=1200, help="characters per chunk (hard-ish cap)")

    ap.add_argument("--min_sent_chars", type=int, default=10)
    ap.add_argument("--max_sent_chars", type=int, default=400)

    ap.add_argument("--max_docs", type=int, default=0, help="0 = no limit (DANGEROUS)")
    ap.add_argument("--max_sentences", type=int, default=0, help="0 = no limit (DANGEROUS)")

    ap.add_argument("--streaming", type=int, default=1, help="1=streaming iterator (recommended)")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    chunk_sents = max(1, int(args.chunk_sents))
    chunk_chars = max(200, int(args.chunk_chars))

    # Streaming load
    ds = load_dataset(args.dataset, split=args.split, streaming=bool(args.streaming))

    # We keep a rolling buffer per (title, pi) and flush when large enough.
    # In streaming mode, we can't sort globally; we rely on dataset order.
    # If 'si' exists and the stream is grouped, it works well.
    buffers = {}  # key -> list[str]
    metas = []
    n_sent = 0

    # Determine available keys defensively
    first = None
    for ex in ds.take(1):
        first = ex
        break
    if first is None:
        raise SystemExit("Dataset appears empty or unavailable.")

    keys = set(first.keys())
    # Expected: title, sentence, pi, si (but we keep it robust)
    title_key = "title" if "title" in keys else None
    sent_key = "sentence" if "sentence" in keys else None
    if sent_key is None:
        # try common alternatives
        for k in ("text", "sent", "content"):
            if k in keys:
                sent_key = k
                break
    if sent_key is None:
        raise SystemExit(f"Cannot find sentence/text field in dataset keys: {sorted(keys)}")

    pi_key = "pi" if "pi" in keys else None

    def flush(key):
        nonlocal buffers, metas
        buf = buffers.get(key)
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            title = key[0] if isinstance(key, tuple) and len(key) > 0 else ""
            metas.append({"title": title, "text": text})
        buffers[key] = []

    max_docs = int(args.max_docs)
    max_sentences = int(args.max_sentences)

    # Iterate
    it = ds
    for ex in it:
        s = _norm_ws(ex.get(sent_key, ""))
        if not s:
            continue
        if len(s) < args.min_sent_chars:
            continue
        if len(s) > args.max_sent_chars:
            s = s[: args.max_sent_chars].rstrip()

        title = _norm_ws(ex.get(title_key, "")) if title_key else ""
        pi = ex.get(pi_key, 0) if pi_key else 0
        try:
            pi = int(pi)
        except Exception:
            pi = 0

        key = (title, pi)
        buf = buffers.get(key)
        if buf is None:
            buf = []
            buffers[key] = buf

        buf.append(s)
        n_sent += 1

        # flush conditions
        if len(buf) >= chunk_sents or sum(len(x) for x in buf) >= chunk_chars:
            flush(key)

        if max_sentences > 0 and n_sent >= max_sentences:
            break
        if max_docs > 0 and len(metas) >= max_docs:
            break

        # periodic flush to prevent too many keys in memory (stream disorder)
        if (n_sent % 200000) == 0:
            # flush all to cap memory; slight quality hit if title/pi interleaves
            for k in list(buffers.keys()):
                flush(k)
            buffers = {k: v for k, v in buffers.items() if v}

            print(f"[PROGRESS] sentences={n_sent:,} docs={len(metas):,} active_keys={len(buffers):,}")

    # final flush
    for k in list(buffers.keys()):
        flush(k)

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(metas, out)

    print(f"[DONE] wrote {out}")
    print(f"  docs: {len(metas):,}")
    print(f"  sentences seen: {n_sent:,}")
    print(f"  chunk_sents={chunk_sents} chunk_chars={chunk_chars}")
    if max_docs:
        print(f"  max_docs={max_docs}")
    if max_sentences:
        print(f"  max_sentences={max_sentences}")


if __name__ == "__main__":
    main()
