#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert detailed JSON predictions to probability CSV (id,p1..p5) compatible with ensemble_soft_voting.py.

JSON format expected (example):
{
  "predictions": [
    {"id": "...", "probabilities": {"1":0.1,"2":0.2,"3":0.3,"4":0.2,"5":0.2}, ...},
    ...
  ]
}
"""

import argparse
import json
import math
import pandas as pd

def to_float(x):
    try:
        if x is None:
            return float("nan")
        v = float(x)
        if math.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Input detailed JSON path")
    ap.add_argument("--out_csv", required=True, help="Output CSV path (id,p1..p5)")
    ap.add_argument("--normalize", action="store_true",
                    help="Row-normalize probabilities to sum to 1 (after NaN->0).")
    ap.add_argument("--fill_missing", default="nan", choices=["nan", "zero"],
                    help="How to fill missing choice probs. Default: nan (ensemble script uses nanmean).")
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    preds = obj.get("predictions", None)
    if not isinstance(preds, list):
        raise ValueError("Invalid JSON: 'predictions' must be a list")

    rows = []
    for item in preds:
        _id = str(item.get("id", ""))
        probs = item.get("probabilities", {}) or {}

        # JSON may store keys as "1".."5" (strings) or 1..5 (ints)
        p = {}
        for k in [1, 2, 3, 4, 5]:
            val = probs.get(str(k), probs.get(k, None))
            p[k] = to_float(val)

        if args.fill_missing == "zero":
            for k in [1,2,3,4,5]:
                if math.isnan(p[k]):
                    p[k] = 0.0

        rows.append({
            "id": _id,
            "p1": p[1],
            "p2": p[2],
            "p3": p[3],
            "p4": p[4],
            "p5": p[5],
        })

    df = pd.DataFrame(rows, columns=["id", "p1", "p2", "p3", "p4", "p5"])

    if args.normalize:
        # Convert NaN -> 0 for normalization
        prob_cols = ["p1","p2","p3","p4","p5"]
        tmp = df[prob_cols].fillna(0.0)
        s = tmp.sum(axis=1)
        s[s == 0.0] = 1.0
        df[prob_cols] = tmp.div(s, axis=0)

    df.to_csv(args.out_csv, index=False)
    print(f"[DONE] wrote: {args.out_csv} (n={len(df)})")

if __name__ == "__main__":
    main()
