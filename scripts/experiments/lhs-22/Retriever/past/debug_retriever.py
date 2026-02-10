#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import re
from pathlib import Path


import pandas as pd

# ✅ 네가 만든 Hybrid retriever 파일을 그대로 import
# retriever.py 안의 HybridWikiRetriever 사용
from retriever import HybridWikiRetriever


def safe_parse_problems(x):
    """
    test.csv의 problems는 문자열로 dict/list가 들어있음.
    """
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        try:
            return ast.literal_eval(x)
        except Exception:
            return None
    return None


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def shorten(s: str, n: int) -> str:
    s = normalize_ws(s)
    return s if len(s) <= n else s[:n] + " ..."


def build_query(paragraph, question, question_plus, choices, use_paragraph=True, para_chars=450):
    """
    Retrieval query를 question-only가 아니라:
    - question + choices는 반드시 포함 (선지에 키워드가 많음)
    - paragraph는 너무 길면 retrieval에 노이즈가 되므로 앞부분만 요약해서 포함(옵션)
    """
    parts = []

    # 1) (옵션) 지문 앞부분만 포함: (가)(나) 같은 문항에서 키워드 힌트가 됨
    if use_paragraph and paragraph:
        parts.append("[지문요약] " + shorten(paragraph, para_chars))

    # 2) 질문(+question_plus)
    q = normalize_ws(question)
    if question_plus:
        q = q + " " + normalize_ws(question_plus)
    parts.append("[질문] " + q)

    # 3) 선택지(필수): 사회/국어는 선지에 고유명사/정책/용어가 다 들어있는 경우가 많음
    if choices:
        # 너무 길면 top-k에 불리할 수 있으니 선택지 길이 컷
        c_str = " / ".join([f"{i+1}) {shorten(c, 140)}" for i, c in enumerate(choices)])
        parts.append("[선택지] " + c_str)

    return "\n".join(parts)


def extract_choices(p0: dict):
    """
    problems 구조가 팀/버전에 따라 다를 수 있어서 방어적으로 처리.
    """
    if not isinstance(p0, dict):
        return []
    ch = p0.get("choices", None)
    if isinstance(ch, list) and ch:
        return [str(x) for x in ch]
    # 다른 키 이름 가능성(방어)
    for k in ("choice", "options", "cand", "candidates"):
        if k in p0 and isinstance(p0[k], list):
            return [str(x) for x in p0[k]]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True, help="test.csv 경로")
    ap.add_argument("--n", type=int, default=10, help="확인할 샘플 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="retrieval_debug.jsonl", help="결과 저장(jsonl)")
    ap.add_argument("--use_paragraph", type=int, default=1, help="지문 일부를 query에 포함할지(1/0)")
    ap.add_argument("--para_chars", type=int, default=450)

    # retriever args
    ap.add_argument("--index", default="wiki.faiss")
    ap.add_argument("--docstore", default="docstore.pt")
    ap.add_argument("--tfidf_vec", default="tfidf_vectorizer.joblib")
    ap.add_argument("--tfidf_mat", default="tfidf_matrix.npz")
    ap.add_argument("--dense_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--reranker_model", default="BAAI/bge-reranker-v2-m3")

    ap.add_argument("--k_final", type=int, default=5)
    ap.add_argument("--k_dense", type=int, default=30)
    ap.add_argument("--k_sparse", type=int, default=50)
    ap.add_argument("--k_fuse", type=int, default=80)
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--no_rerank", action="store_true")
    args = ap.parse_args()

    test_path = Path(args.test_csv)
    assert test_path.exists(), f"not found: {test_path}"

    df = pd.read_csv(test_path)

    # problems 파싱 실패가 있을 수 있으니, 샘플링 전에 안전 필터링
    rows = df.to_dict(orient="records")

    # 랜덤 샘플
    import random
    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[:args.n]

    retriever = HybridWikiRetriever(
        faiss_index_path=args.index,
        docstore_path=args.docstore,
        tfidf_vec_path=args.tfidf_vec,
        tfidf_mat_path=args.tfidf_mat,
        dense_model=args.dense_model,
        reranker_model=None if args.no_rerank else args.reranker_model,
    )

    use_para = bool(args.use_paragraph)

    out_f = open(args.out, "w", encoding="utf-8")

    for idx, row in enumerate(rows, 1):
        qid = str(row.get("id", idx))
        paragraph = row.get("paragraph", "")
        problems = safe_parse_problems(row.get("problems"))

        if problems is None:
            print(f"\n[{idx}] id={qid}  (problems parse failed) -> skip")
            continue

        p0 = problems[0] if isinstance(problems, list) else problems
        if not isinstance(p0, dict) or "question" not in p0:
            print(f"\n[{idx}] id={qid}  (invalid problems format) -> skip")
            continue

        question = p0.get("question", "")
        question_plus = row.get("question_plus", None)
        if isinstance(question_plus, float):  # NaN
            question_plus = None
        if question_plus is None:
            question_plus = p0.get("question_plus", None)

        choices = extract_choices(p0)
        k_choices = len(choices)

        query = build_query(
            paragraph=paragraph,
            question=question,
            question_plus=question_plus,
            choices=choices,
            use_paragraph=use_para,
            para_chars=args.para_chars,
        )

        docs = retriever.retrieve(
            query,
            k_final=args.k_final,
            k_dense=args.k_dense,
            k_sparse=args.k_sparse,
            k_fuse=args.k_fuse,
            use_rerank=(not args.no_rerank),
            rrf_k=args.rrf_k,
        )

        print("\n" + "=" * 90)
        print(f"[{idx}] id={qid}  #choices={k_choices}")
        print("QUERY:")
        print(shorten(query, 900))
        print("-" * 90)
        for r, d in enumerate(docs, 1):
            title = d.get("title", "")
            text = d.get("text", "")
            print(f"[DOC {r}] {title}")
            print(" ", shorten(text, 260))
        print("=" * 90)

        out_obj = {
            "id": qid,
            "n_choices": k_choices,
            "query": query,
            "retrieved": [
                {"rank": r, "title": d.get("title", ""), "text_head": shorten(d.get("text", ""), 400)}
                for r, d in enumerate(docs, 1)
            ],
        }
        out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()
    print(f"\n[DONE] wrote -> {args.out}")


if __name__ == "__main__":
    main()
