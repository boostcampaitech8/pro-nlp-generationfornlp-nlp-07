#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, gzip, json
from tqdm import tqdm
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_gz", required=True)
    ap.add_argument("--out", required=True, help="docstore.pt")
    args = ap.parse_args()

    texts = []
    metas = []  # title, doc_id, chunk_id 등

    with gzip.open(args.chunk_gz, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="loading chunks"):
            d = json.loads(line)
            texts.append(d["chunk"])
            metas.append({
                "chunk_id": d["chunk_id"],
                "doc_id": d.get("doc_id"),
                "title": d.get("title", ""),
            })

    obj = {"texts": texts, "metas": metas}
    torch.save(obj, args.out)
    print("[DONE] wrote:", args.out, "N=", len(texts))

if __name__ == "__main__":
    main()
