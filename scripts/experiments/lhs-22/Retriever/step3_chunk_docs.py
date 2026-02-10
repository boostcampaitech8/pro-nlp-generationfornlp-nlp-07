#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json

def chunk_by_words(text, max_words=320, stride=240):
    """
    O(n) guaranteed chunking
    - one split
    - sliding window
    """
    words = text.split()
    n = len(words)
    chunks = []

    i = 0
    while i < n:
        chunk = words[i:i+max_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += stride

    return chunks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_words", type=int, default=320)
    ap.add_argument("--stride", type=int, default=240)
    ap.add_argument("--max_doc_words", type=int, default=6000)  # ★ 결정적
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    n_docs = 0
    n_chunks = 0

    with gzip.open(args.docs, "rt", encoding="utf-8", errors="ignore") as fin, \
         gzip.open(args.out, "wt", encoding="utf-8") as fout:

        for line in fin:
            n_docs += 1
            doc = json.loads(line)

            text = doc.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            # ★ 문서 단어 수 강제 컷 (핵심)
            words = text.split()
            if len(words) > args.max_doc_words:
                words = words[:args.max_doc_words]
                text = " ".join(words)

            chunks = chunk_by_words(
                text,
                max_words=args.max_words,
                stride=args.stride
            )

            for cid, c in enumerate(chunks):
                fout.write(json.dumps({
                    "title": doc.get("title", ""),
                    "page_id": doc.get("page_id", None),
                    "chunk_id": cid,
                    "text": c
                }, ensure_ascii=False) + "\n")
                n_chunks += 1

            if n_docs % args.log_every == 0:
                print(f"[chunking] docs={n_docs}, chunks={n_chunks}")

            if args.limit and n_docs >= args.limit:
                break

    print("\n===== CHUNK SUMMARY =====")
    print(f"docs_in:   {n_docs}")
    print(f"chunks:    {n_chunks}")
    print(f"avg_chunks_per_doc: {n_chunks / max(n_docs,1):.2f}")
    print(f"output:    {args.out}")
    print("=========================\n")
