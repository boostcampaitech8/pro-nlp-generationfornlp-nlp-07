# prepare_queries.py
import pandas as pd
import ast
from tqdm import tqdm

QUESTION_KEYS = ["question", "stem", "query", "question_text"]

def extract_question_from_row(row):
    # 1) direct columns
    for col in QUESTION_KEYS:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col].strip()

    # 2) problems field (string / list / dict)
    if "problems" in row:
        p = row["problems"]

        # string -> parse
        if isinstance(p, str):
            try:
                p = ast.literal_eval(p)
            except Exception:
                return None

        # list -> first problem
        if isinstance(p, list) and len(p) > 0:
            p = p[0]

        # dict -> extract fields
        if isinstance(p, dict):
            q = ""
            for k in QUESTION_KEYS:
                if k in p and isinstance(p[k], str):
                    q = p[k].strip()
                    break

            qp = ""
            for k in ["question_plus", "passage", "context"]:
                if k in p and isinstance(p[k], str):
                    qp = p[k].strip()
                    break

            if q:
                return f"{q} {qp}".strip()

    return None


def load_queries(*paths):
    queries = []
    for path in paths:
        df = pd.read_csv(path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading {path}"):
            q = extract_question_from_row(row)
            if q:
                queries.append(q)
    return list(set(queries))


if __name__ == "__main__":
    queries = load_queries(
        "/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/train/train.csv",
        "/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/train/train_with_cot.csv",
        "/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv",
    )

    with open("queries.txt", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(q + "\n")

    print(f"[OK] saved {len(queries)} unique queries -> queries.txt")
