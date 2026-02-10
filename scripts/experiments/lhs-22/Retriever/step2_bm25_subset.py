# step2_bm25_subset.py
import argparse, gzip, json
from tqdm import tqdm
from rank_bm25 import BM25Okapi

def tokenize(s: str):
    return s.split()

def load_docs(path):
    docs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="kowiki_filtered.jsonl.gz")
    ap.add_argument("--queries", default="queries.txt")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--out", default="kowiki_bm25_subset.jsonl.gz")
    args = ap.parse_args()

    docs = load_docs(args.docs)
    corpus = [tokenize(d["text"][:2000]) for d in docs]  # 앞부분만 사용
    bm25 = BM25Okapi(corpus)

    queries = load_queries(args.queries)
    hit_ids = set()

    for q in tqdm(queries, desc="BM25 querying"):
        scores = bm25.get_scores(tokenize(q))
        topk = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.topk]
        hit_ids.update(topk)

    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for i in sorted(hit_ids):
            f.write(json.dumps(docs[i], ensure_ascii=False) + "\n")

    print("\n===== BM25 SUBSET SUMMARY =====")
    print(f"docs_total: {len(docs)}")
    print(f"docs_subset: {len(hit_ids)} ({len(hit_ids)/len(docs):.3f})")
    print(f"output: {args.out}")
    print("================================\n")
