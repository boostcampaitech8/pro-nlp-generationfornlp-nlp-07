#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, gzip, json, re
from tqdm import tqdm

_WS = re.compile(r"\s+")

def norm(s: str) -> str:
    s = (s or "").strip()
    s = _WS.sub(" ", s)
    return s

def simple_chunk(text: str, chunk_chars: int = 1400, overlap_chars: int = 200):
    """
    토큰 기반이 베스트지만, 구현 단순화를 위해 char 기반 슬라이딩.
    (나중에 tokenizer 기반 토큰 chunk로 바꿔도 됨)
    """
    text = norm(text)
    if not text:
        return []
    out = []
    n = len(text)
    i = 0
    while i < n:
        j = min(n, i + chunk_chars)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap_chars)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_gz", required=True, help="source jsonl.gz (e.g., kowiki_filtered.jsonl.gz)")
    ap.add_argument("--out_gz", required=True, help="chunked jsonl.gz")
    ap.add_argument("--chunk_chars", type=int, default=1400)
    ap.add_argument("--overlap_chars", type=int, default=200)
    args = ap.parse_args()

    with gzip.open(args.in_gz, "rt", encoding="utf-8") as f_in, gzip.open(args.out_gz, "wt", encoding="utf-8") as f_out:
        cid = 0
        for line in tqdm(f_in, desc="chunking"):
            d = json.loads(line)
            title = d.get("title") or d.get("doc_title") or ""
            text = d.get("text") or d.get("content") or ""
            chunks = simple_chunk(text, args.chunk_chars, args.overlap_chars)
            for ci, ch in enumerate(chunks):
                out = {
                    "chunk_id": cid,
                    "doc_id": d.get("id", None),
                    "title": title,
                    "chunk": ch,
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                cid += 1

    print("[DONE] wrote:", args.out_gz)

if __name__ == "__main__":
    main()
