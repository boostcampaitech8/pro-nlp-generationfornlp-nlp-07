#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build ORPO/DPO preference JSONL from:
- train.csv (gold answer)
- probs_train.csv (p1..p5 from base logits)

Output JSONL with columns: prompt, chosen, rejected
(chosen/rejected are single-token strings: "1".."5")
"""

import json
from ast import literal_eval
import pandas as pd


SYSTEM_PROMPT = "지문을 읽고 질문의 답을 구하세요."

def safe_parse_problems(x):
    if isinstance(x, (list, dict)):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return literal_eval(s)
    except Exception:
        return None

def extract_choices(row, p0):
    ch = p0.get("choices")
    if isinstance(ch, list) and len(ch) > 0:
        return ch
    # fallback
    ch2 = row.get("choices")
    if isinstance(ch2, str):
        try:
            parsed = literal_eval(ch2)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except Exception:
            pass
    if isinstance(ch2, list) and len(ch2) > 0:
        return ch2
    return []

def build_prompt(paragraph: str, question: str, question_plus, choices):
    qps = f"\n<보기>:\n{question_plus}" if (question_plus is not None and str(question_plus).strip() != "") else ""
    ch = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    nums = ", ".join(str(i) for i in range(1, len(choices) + 1))
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"지문:\n{paragraph}\n\n"
        f"질문:\n{question}{qps}\n\n"
        f"선택지:\n{ch}\n\n"
        f"{nums} 중에 하나를 정답으로 고르세요.\n"
        f"[생각]\n"
        f"정답:"
    )

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--probs_csv", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    train_df = pd.read_csv(args.train_csv)
    probs_df = pd.read_csv(args.probs_csv)

    # probs_df는 inference 결과라 id가 문자열일 수 있음 → 문자열로 통일
    probs_df["id"] = probs_df["id"].astype(str)
    train_df["id"] = train_df["id"].astype(str)

    probs_map = probs_df.set_index("id").to_dict(orient="index")

    n_total, n_written = 0, 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for _, row in train_df.iterrows():
            n_total += 1
            qid = str(row.get("id", ""))

            pinfo = probs_map.get(qid)
            if pinfo is None:
                continue

            problems = safe_parse_problems(row.get("problems"))
            if problems is None:
                continue
            p0 = problems[0] if isinstance(problems, list) else problems
            if not isinstance(p0, dict):
                continue

            # gold answer
            gold = p0.get("answer", None)
            if gold is None:
                continue
            gold = int(gold)

            paragraph = row.get("paragraph", "")
            question = p0.get("question", "")
            question_plus = row.get("question_plus", None)
            if question_plus is None:
                question_plus = p0.get("question_plus", None)

            choices = extract_choices(row, p0)
            if not choices:
                continue
            K = len(choices)

            # probabilities p1..pK
            probs = []
            for j in range(1, K + 1):
                pj = pinfo.get(f"p{j}", None)
                probs.append(float(pj) if pj == pj else float("-inf"))  # NaN 처리

            # rejected = argmax_{j != gold} probs[j]
            best_j, best_p = None, float("-inf")
            for j in range(1, K + 1):
                if j == gold:
                    continue
                if probs[j-1] > best_p:
                    best_p = probs[j-1]
                    best_j = j
            if best_j is None:
                continue

            prompt = build_prompt(paragraph, question, question_plus, choices)

            rec = {
                "prompt": prompt,
                "chosen": str(gold),
                "rejected": str(best_j),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[DONE] total={n_total} written={n_written} -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
