#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import re
from datasets import load_dataset
from tqdm import tqdm

def normalize(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()

def is_valid(title: str, content: str, min_chars: int) -> bool:
    if not title or not content:
        return False

    content = normalize(content)

    if len(content) < min_chars:
        return False

    # 아주 약한 제거만 (과필터링 방지)
    bad_title_keywords = ("틀:", "파일:", "위키백과:", "분류:")
    if any(k in title for k in bad_title_keywords):
        return False

    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kowiki_filtered.jsonl.gz")
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--streaming", type=int, default=1)
    args = ap.parse_args()

    ds = load_dataset(
        "NLP-07-ODQA/kowiki-cleaned",
        split=args.split,
        streaming=bool(args.streaming),
    )

    processed = 0
    kept = 0

    with gzip.open(args.out, "wt", encoding="utf-8") as fout:
        for ex in tqdm(ds, desc="Filtering docs"):
            processed += 1

            title = str(ex.get("title", "")).strip()
            content = ex.get("content", "")  # ✅ 핵심: content 사용
            if not isinstance(content, str):
                content = str(content)

            if is_valid(title, content, args.min_chars):
                out = {
                    "title": title,
                    "text": normalize(content),          # downstream 호환 위해 text로 저장
                    "page_id": ex.get("page_id", None),
                    "text_length": ex.get("text_length", None),
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept += 1

            if args.limit and processed >= args.limit:
                break

    print("\n===== FILTER SUMMARY =====")
    print(f"processed: {processed}")
    print(f"kept:      {kept} ({kept/processed:.3f})")
    print(f"output:    {args.out}")
    print("==========================\n")
