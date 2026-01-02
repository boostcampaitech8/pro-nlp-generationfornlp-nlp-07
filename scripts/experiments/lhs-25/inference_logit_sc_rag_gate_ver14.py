#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unsloth  # noqa: F401

import os
import random
from ast import literal_eval
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer
from peft import PeftModel

from rag_gate import decide_use_rag
from rag_utils import filter_docs, run_rag
try:
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except Exception:
    _HAS_UNSLOTH = False
    

from scipy.sparse import load_npz
import joblib

# --------------------
# Utils
# --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_choices(row: dict, p0: dict) -> list[str]:
    ch = p0.get("choices")
    if isinstance(ch, list) and ch:
        return ch
    ch2 = row.get("choices")
    if isinstance(ch2, str):
        try:
            parsed = literal_eval(ch2)
            if isinstance(parsed, list) and parsed:
                return parsed
        except Exception:
            pass
    if isinstance(ch2, list) and ch2:
        return ch2
    return []


def build_user_content(paragraph, question, question_plus, choices):
    qps = f"\n<보기>:\n{question_plus}" if question_plus else ""
    ch = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    nums = ", ".join(str(i) for i in range(1, len(choices) + 1))
    return (
        "지문을 읽고 질문의 답을 구하세요.\n\n"
        f"지문:\n{paragraph}\n\n"
        f"질문:\n{question}{qps}\n\n"
        f"선택지:\n{ch}\n\n"
        f"{nums} 중에 하나를 정답으로 고르세요.\n"
        "정답:"
    )


def build_chatml_prompt(tokenizer, paragraph, question, question_plus, choices):
    messages = [
        {"role": "user", "content": build_user_content(paragraph, question, question_plus, choices)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def label_token_candidates(tokenizer, label: int):
    out = []
    for v in (str(label), " " + str(label), "\n" + str(label)):
        ids = tokenizer.encode(v, add_special_tokens=False)
        if ids:
            out.append(ids[-1])
    return list(dict.fromkeys(out))


@torch.inference_mode()
def logits_label_stats(model, tokenizer, prompt, allow_labels, max_len):
    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(model.device)

    out = model(**inputs)
    scores = out.logits[0, -1]

    label_scores = []
    for l in allow_labels:
        ids = label_token_candidates(tokenizer, l)
        label_scores.append(max(scores[i].item() for i in ids))

    logits = torch.tensor(label_scores, device=scores.device)
    probs = torch.softmax(logits, dim=-1)

    top2 = torch.topk(logits, 2).values
    margin = float(top2[0] - top2[1])
    entropy = float((-probs * torch.log(probs + 1e-12)).sum())
    top1p = float(probs.max())

    pred = str(allow_labels[int(torch.argmax(probs))])
    return pred, margin, entropy, top1p


# --------------------
# Main
# --------------------
def main():
    from retriever_hybrid_faiss_tfidf import HybridWikiRetriever

    retriever = HybridWikiRetriever(
        faiss_path="faiss_index.index",
        docstore_path="docstore.pt",
        tfidf_vec_path="tfidf_vectorizer.joblib",
        tfidf_mat_path="tfidf_matrix.npz",
    )

    base_model = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
    adapter_model = os.environ.get("ADAPTER_MODEL", "").strip()
    test_path = os.environ.get("TEST_PATH", "/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv")
    out_path = os.environ.get("OUT_PATH", "./submission_rag.csv")
    seed = int(os.environ.get("SEED", "42"))
    max_len = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))

    set_seed(seed)

    # Load model
    if _HAS_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_len,
            load_in_4bit=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_model:
        model = PeftModel.from_pretrained(model, adapter_model)

    model.eval()

    df = pd.read_csv(test_path)
    out_rows = []
    cnt = defaultdict(int)

    pbar = tqdm(df.to_dict(orient="records"), total=len(df), dynamic_ncols=True)

    for row in pbar:
        qid = str(row["id"])
        paragraph = row["paragraph"]

        problems = literal_eval(row["problems"])

        if isinstance(problems, list):
            p0 = problems[0]
        elif isinstance(problems, dict):
            p0 = problems
        else:
            # 방어적 fallback
            out_rows.append({"id": qid, "answer": "1"})
            cnt["1"] += 1
            continue
        question = p0["question"]
        question_plus = p0.get("question_plus")
        choices = extract_choices(row, p0)

        allow_labels = list(range(1, len(choices) + 1))
        prompt = build_chatml_prompt(tokenizer, paragraph, question, question_plus, choices)

        pred_base, m_base, e_base, p_base = logits_label_stats(
            model, tokenizer, prompt, allow_labels, max_len
        )
        pred = pred_base

        gate = decide_use_rag(
            model, tokenizer,
            paragraph, question, question_plus, choices, max_len
        )
        did_rag = False
        rag_log = {"id": qid, "gate": gate}

        if gate == "USE":
            docs = retriever.retrieve(
                query=question,
                k_sparse=50,
                k_dense=50,
                k_final=20,   # 후보 넉넉히
                q_emb=None,   # 지금은 dense query 안 쓰는 구조
            )
            filtered = filter_docs(docs, question, topn=5)

            if filtered:
                pred_rag, m_rag, e_rag, p_rag, _ = run_rag(
                    model, tokenizer, prompt, filtered, allow_labels, max_len
                )

                improve = (m_rag > m_base) or (p_rag > p_base) or (e_rag < e_base)
                if improve:
                    pred = pred_rag
                    did_rag = True

                rag_log.update({
                    "used": did_rag,
                    "base": {"pred": pred_base, "margin": m_base, "entropy": e_base, "top1p": p_base},
                    "rag": {"pred": pred_rag, "margin": m_rag, "entropy": e_rag, "top1p": p_rag},
                    "docs": filtered,
                })

        with open("rag_debug.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rag_log, ensure_ascii=False) + "\n")

        out_rows.append({"id": qid, "answer": pred})
        cnt[pred] += 1
        pbar.set_postfix(pred=pred)

    pd.DataFrame(out_rows).to_csv(out_path, index=False)

    print(f"[DONE] {out_path}")
    print("[CNT]", dict(cnt))


if __name__ == "__main__":
    main()
