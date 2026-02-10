#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soft-voting ensemble for probability-dump inference outputs.

Input: one or more CSVs produced by inference_logit_sc_probs_dump.py
Each CSV must have:
  id, p1..p5  (NaN allowed if <5 choices, but contest is usually 5)

Output:
  - ensemble submission CSV: id, answer
  - optional: dump averaged probs for inspection

Usage:
  python3 ensemble_soft_voting.py --out submission_ens.csv modelA_probs.csv modelB_probs.csv ...

Env vars (optional):
  ENS_TIE_BREAK="base"  # base|min_entropy|max_margin (default base)
  ENS_PROB_OUT="avg_probs.csv"  # if set, writes averaged probs file
"""

import argparse
import os
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output submission csv (id,answer).")
    ap.add_argument("inputs", nargs="+", help="Probability CSVs from inference.")
    args = ap.parse_args()

    tie_break = os.environ.get("ENS_TIE_BREAK", "base").strip().lower()
    prob_out = os.environ.get("ENS_PROB_OUT", "").strip()

    dfs = []
    for p in args.inputs:
        df = pd.read_csv(p)
        need = {"id", "p1", "p2", "p3", "p4", "p5"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"{p}: missing columns: {sorted(missing)}")
        df = df[["id", "p1", "p2", "p3", "p4", "p5"]].copy()
        df.rename(columns={c: f"{c}__{len(dfs)}" for c in ["p1","p2","p3","p4","p5"]}, inplace=True)
        dfs.append(df)

    # merge on id
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="id", how="inner", validate="one_to_one")

    # average probs
    P = []
    for k in range(len(dfs)):
        P.append(merged[[f"p1__{k}", f"p2__{k}", f"p3__{k}", f"p4__{k}", f"p5__{k}"]].to_numpy(dtype=np.float64))
    P = np.stack(P, axis=0)  # [M, N, 5]
    avg = np.nanmean(P, axis=0)  # [N, 5]
    avg = np.nan_to_num(avg, nan=0.0) ####### 오류 수정
    
    # normalize row-wise (in case NaNs or non-summing)
    row_sum = np.sum(avg, axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    avg = avg / row_sum

    pred = np.argmax(avg, axis=1) + 1  # 1..5

    out = pd.DataFrame({"id": merged["id"].astype(str), "answer": pred.astype(str)})

    out.to_csv(args.out, index=False)
    print(f"[DONE] wrote submission: {args.out} (n={len(out)})")

    if prob_out:
        prob_df = pd.DataFrame(avg, columns=["p1","p2","p3","p4","p5"])
        prob_df.insert(0, "id", merged["id"].astype(str))
        prob_df.to_csv(prob_out, index=False)
        print(f"[DONE] wrote avg probs: {prob_out}")

if __name__ == "__main__":
    main()
