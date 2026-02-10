# step2_bm25_subset.py
import argparse, gzip, json
from tqdm import tqdm
import pandas as pd
from ast import literal_eval
from rank_bm25 import BM25Okapi


def tokenize(s: str):
    return s.split()


def load_docs(path):
    docs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def build_query(row, para_chars=400):
    """
    inference의 build_retrieval_query와 최대한 동일한 형태
    """
    paragraph = str(row.get("paragraph", "")).strip()
    question_plus = row.get("question_plus", "")
    if isinstance(question_plus, float):
        question_plus = ""

    problems = row.get("problems")
    try:
        problems = literal_eval(problems) if isinstance(problems, str) else problems
    except Exception:
        return None

    if isinstance(problems, list):
        p0 = problems[0]
    elif isinstance(problems, dict):
        p0 = problems
    else:
        return None

    question = p0.get("question", "").strip()
    choices = p0.get("choices", [])

    parts = []
    if paragraph:
        parts.append(paragraph[:para_chars])
    if question:
        parts.append(question)
    if question_plus:
        parts.append(str(question_plus)[:200])
    if isinstance(choices, list) and choices:
        parts.append(" / ".join(c.strip() for c in choices))

    return "\n".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="kowiki_filtered.jsonl.gz")
    ap.add_argument("--csv", nargs="+", required=True,
                    help="train.csv / test.csv 등 (복수 가능)")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--out", default="kowiki_bm25_subset.jsonl.gz")
    ap.add_argument("--para_chars", type=int, default=400)
    args = ap.parse_args()

    # 1) load wiki docs
    docs = load_docs(args.docs)
    corpus = [tokenize(d["text"][:2000]) for d in docs]
    bm25 = BM25Okapi(corpus)

    # 2) load queries from csv
    queries = []
    for csv_path in args.csv:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            q = build_query(row, para_chars=args.para_chars)
            if q:
                queries.append(q)

    print(f"[INFO] total queries: {len(queries)}")

    # 3) BM25 querying
    hit_ids = set()
    for q in tqdm(queries, desc="BM25 querying"):
        scores = bm25.get_scores(tokenize(q))
        topk = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:args.topk]
        hit_ids.update(topk)

    # 4) save subset
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for i in sorted(hit_ids):
            f.write(json.dumps(docs[i], ensure_ascii=False) + "\n")

    print("\n===== BM25 SUBSET SUMMARY =====")
    print(f"docs_total: {len(docs)}")
    print(f"docs_subset: {len(hit_ids)} ({len(hit_ids)/len(docs):.3f})")
    print(f"output: {args.out}")
    print("================================\n")



