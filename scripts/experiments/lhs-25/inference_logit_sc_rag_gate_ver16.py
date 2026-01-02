#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Unsloth must be imported BEFORE torch/transformers/peft
import unsloth  # noqa: F401

import os
import math
import random
import json
import re
from ast import literal_eval
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
from peft import PeftModel

from rag_gate import decide_use_rag
from retriever_hybrid_faiss_tfidf import HybridWikiRetriever

try:
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except Exception:
    _HAS_UNSLOTH = False


# --------------------
# Seed
# --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------
# SC: restrict tokens to labels
# --------------------
class RestrictTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allow_token_ids: set[int], mask_value: float = -1e9):
        self.allow_token_ids = allow_token_ids
        self.mask_value = mask_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, self.mask_value)
        idx = torch.tensor(list(self.allow_token_ids), device=scores.device, dtype=torch.long)
        mask[:, idx] = 0.0
        return scores + mask


def _label_token_candidates(tokenizer, label: int) -> list[int]:
    s = str(label)
    cands = []
    for variant in (s, " " + s, "\n" + s):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            cands.append(ids[-1])
    seen = set()
    out = []
    for i in cands:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


@torch.inference_mode()
def logits_label_stats(model, tokenizer, prompt: str, allow_labels: list[int], max_len: int):
    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    ).to(model.device)

    out = model(**inputs)
    scores = out.logits[0, -1]  # [vocab]

    label_to_score = []
    for l in allow_labels:
        cand_ids = _label_token_candidates(tokenizer, l)
        if not cand_ids:
            label_to_score.append(float("-inf"))
            continue
        v = torch.tensor(cand_ids, device=scores.device, dtype=torch.long)
        label_to_score.append(float(torch.max(scores[v]).item()))
    sub = torch.tensor(label_to_score, device=scores.device)  # [K]
    probs = torch.softmax(sub, dim=-1)

    if sub.numel() >= 2:
        top2 = torch.topk(sub, k=2).values
        margin = float((top2[0] - top2[1]).item())
    else:
        margin = float("inf")

    entropy = float((-probs * torch.log(probs + 1e-12)).sum().item())
    top1p = float(probs.max().item())

    pred_idx = int(torch.argmax(sub).item())
    pred_label = str(allow_labels[pred_idx])
    return pred_label, margin, entropy, top1p


@torch.inference_mode()
def self_consistency_logprob_sum(
    model, tokenizer, prompt: str, allow_labels: list[int],
    n_samples: int, temperature: float, max_len: int,
):
    allow_token_ids = set()
    id_to_label = {}
    for l in allow_labels:
        for tid in _label_token_candidates(tokenizer, l):
            allow_token_ids.add(tid)
            id_to_label[tid] = str(l)

    lp = LogitsProcessorList([RestrictTokensLogitsProcessor(allow_token_ids)])

    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    ).to(model.device)

    label_scores = defaultdict(float)
    any_valid = False

    for _ in range(int(n_samples)):
        gen = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=float(temperature),
            top_p=1.0,
            logits_processor=lp,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        scores = gen.scores[0][0]
        new_token_id = int(gen.sequences[0, inputs["input_ids"].shape[1]].item())
        label = id_to_label.get(new_token_id)
        if label is None:
            continue
        any_valid = True
        lpv = float(torch.log_softmax(scores, dim=-1)[new_token_id].item())
        label_scores[label] += lpv

    if not any_valid:
        return str(allow_labels[0]), 0.0

    items = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = items[0]
    second_score = items[1][1] if len(items) > 1 else -1e9
    gap = float(best_score - second_score)
    return best_label, gap


# --------------------
# Prompt builders
# --------------------
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


def build_user_content(paragraph: str, question: str, question_plus: str | None, choices: list[str]) -> str:
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


def build_chatml_prompt(tokenizer, paragraph: str, question: str, question_plus: str | None, choices: list[str]) -> str:
    messages = [{"role": "user", "content": build_user_content(paragraph, question, question_plus, choices)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# --------------------
# Doc cleaning + sentence selection
# --------------------
_url_re = re.compile(r"https?://\S+|www\.\S+")
_ref_re = re.compile(r"\[[0-9]+\]")
_ws_re = re.compile(r"[ \t]+")

def _clean_text(t: str) -> str:
    t = t or ""
    t = t.replace("\u00a0", " ")
    t = _url_re.sub(" ", t)
    t = _ref_re.sub(" ", t)
    t = re.sub(r"==+[^=]+==+", " ", t)
    t = re.sub(r"\{\{.*?\}\}", " ", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", " ", t, flags=re.DOTALL)
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("|", " ")
    t = _ws_re.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _sent_split(t: str) -> list[str]:
    parts = []
    for blk in re.split(r"\n+", t):
        blk = blk.strip()
        if not blk:
            continue

        ss = re.split(r"([\.?!]|다\.|요\.|임\.)\s+", blk)
        buf = ""
        for i, tok in enumerate(ss):
            if i % 2 == 0:
                buf = tok
            else:
                sent = (buf + tok).strip()
                buf = ""
                if 15 <= len(sent) <= 260:
                    parts.append(sent)
        if buf:
            sent = buf.strip()
            if 15 <= len(sent) <= 260:
                parts.append(sent)
    return parts

_tok_re = re.compile(r"[가-힣A-Za-z0-9]+")

def _tokset(s: str) -> set[str]:
    toks = _tok_re.findall(s or "")
    return set(t.lower() for t in toks if len(t) >= 2)


def build_rag_context(
    docs: list[dict],
    query_text: str,
    *,
    top_docs: int = 3,
    max_sentences: int = 10,
    per_doc_sent_cap: int = 4,
    min_doc_score: float = 1.5,
    max_chars: int = 1600,
) -> tuple[str, list[dict]]:
    """
    Top-doc then sentence selection.
    Returns:
      ctx_str, picked_debug
    """
    qset = _tokset(query_text)
    doc_candidates = []
    if not qset:
        return "", []

    for d in docs or []:
        title = (d.get("title") or "").strip()
        raw = d.get("text") or ""
        cleaned = _clean_text(raw)
        if not cleaned:
            continue
        sents = _sent_split(cleaned)
        if not sents:
            continue

        scored = []
        for s in sents:
            sset = _tokset(s)
            if not sset:
                continue
            ov = len(qset & sset)
            if ov <= 0:
                continue
            score = ov / (1.0 + 0.01 * len(s))
            scored.append((score, s))

        if not scored:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored[:per_doc_sent_cap]
        doc_score = float(sum(sc for sc, _ in top_scored))
        if doc_score < float(min_doc_score):
            continue
        did = int(d.get("id", -1))
        doc_candidates.append((doc_score, did, title, top_scored))

    if not doc_candidates:
        return "", []

    doc_candidates.sort(key=lambda x: x[0], reverse=True)
    doc_candidates = doc_candidates[:max(1, int(top_docs))]

    sent_pool = []
    for doc_score, did, title, top_scored in doc_candidates:
        for score, s in top_scored:
            sent_pool.append((score, did, title, s))

    if not sent_pool:
        return "", []

    sent_pool.sort(key=lambda x: x[0], reverse=True)
    picked_sents = sent_pool[:max_sentences]

    by_doc = {}
    for score, did, title, s in picked_sents:
        by_doc.setdefault((did, title), []).append((score, s))

    lines = []
    picked_debug = []
    total_chars = 0
    for (did, title), items in sorted(by_doc.items(), key=lambda x: (-max(sc for sc,_ in x[1]), x[0][0])):
        items.sort(key=lambda x: x[0], reverse=True)
        ss = [s for _, s in items]
        block = f"[{title or '문서'} #{did}]\n" + "\n".join(f"- {s}" for s in ss)
        if total_chars + len(block) > max_chars:
            continue
        lines.append(block)
        total_chars += len(block) + 2
        picked_debug.append({"id": did, "title": title, "score_max": float(items[0][0]), "sentences": ss})

    ctx = "\n\n".join(lines).strip()
    return ctx, picked_debug


def make_query_text(paragraph: str, question: str, question_plus: str | None, choices: list[str]) -> str:
    # general query used for base RAG (kept)
    qp = question_plus or ""
    return f"{question}\n{qp}\n{paragraph[:400]}"


def make_choice_query(question: str, question_plus: str | None, choice: str) -> str:
    # Choice-conditioned query (MCQ-specific)
    qp = question_plus or ""
    c = (choice or "").strip()
    # short instruction-like anchor helps sparse retrieval
    return f"{question}\n{qp}\n선택지: {c}\n핵심 근거"


def safe_prepare_out_path(out_path: str) -> str:
    out_path = os.path.expandvars(out_path)
    out_path = os.path.expanduser(out_path)
    parent = os.path.dirname(out_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return out_path


def main():
    # ===== env =====
    base_model = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
    adapter_model = os.environ.get("ADAPTER_MODEL", "").strip()
    test_path = os.environ.get("TEST_PATH", "data/test.csv")
    out_path = safe_prepare_out_path(os.environ.get("OUT_PATH", "submission_sc_rag_choice.csv"))
    seed = int(os.environ.get("SEED", "42"))
    max_len = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))

    # ----- SC thresholds -----
    MARGIN_THRESHOLD = float(os.environ.get("MARGIN_THRESHOLD", "0.75"))
    ENTROPY_THRESHOLD = float(os.environ.get("ENTROPY_THRESHOLD", "1.40"))
    TOP1P_THRESHOLD = float(os.environ.get("TOP1P_THRESHOLD", "0.40"))

    SC_TRIGGER_K = int(os.environ.get("SC_TRIGGER_K", "3"))
    CONF_MARGIN = float(os.environ.get("CONF_MARGIN", "2.0"))
    CONF_TOP1P = float(os.environ.get("CONF_TOP1P", "0.80"))
    CONF_ENTROPY = float(os.environ.get("CONF_ENTROPY", "0.60"))
    HARD_MARGIN = float(os.environ.get("HARD_MARGIN", "0.40"))

    MAX_SC_RATE = float(os.environ.get("MAX_SC_RATE", "0.10"))
    SC_SAMPLES_BASE = int(os.environ.get("SC_SAMPLES_BASE", "9"))
    SC_TEMPERATURE_BASE = float(os.environ.get("SC_TEMPERATURE_BASE", "0.65"))
    SC_SAMPLES_HARD = int(os.environ.get("SC_SAMPLES_HARD", "13"))
    SC_TEMPERATURE_HARD = float(os.environ.get("SC_TEMPERATURE_HARD", "0.80"))

    SC_ACCEPT_MARGIN = float(os.environ.get("SC_ACCEPT_MARGIN", "1.0"))
    SC_ACCEPT_MARGIN_STRICT = float(os.environ.get("SC_ACCEPT_MARGIN_STRICT", "1.8"))

    # ----- Retrieval budget -----
    RAG_MAX_DOCS = int(os.environ.get("RAG_MAX_DOCS", "30"))  # per choice
    RAG_TOP_DOCS = int(os.environ.get("RAG_TOP_DOCS", "3"))
    RAG_PER_DOC_SENT = int(os.environ.get("RAG_PER_DOC_SENT", "4"))
    RAG_SENTENCES = int(os.environ.get("RAG_SENTENCES", "10"))
    RAG_CTX_CHARS = int(os.environ.get("RAG_CTX_CHARS", "1600"))
    RAG_MIN_DOC_SCORE = float(os.environ.get("RAG_MIN_DOC_SCORE", "1.5"))

    # Choice-conditioned controls
    CHOICE_RAG_TOPK = int(os.environ.get("CHOICE_RAG_TOPK", "2"))  # number of best choice-contexts to consider (speed)
    CHOICE_RAG_MODE = os.environ.get("CHOICE_RAG_MODE", "best_by_margin").strip().lower()
    # accept deltas (make flips safer)
    RAG_ACCEPT_D_MARGIN = float(os.environ.get("RAG_ACCEPT_D_MARGIN", "0.10"))
    RAG_ACCEPT_D_TOP1P = float(os.environ.get("RAG_ACCEPT_D_TOP1P", "0.02"))
    RAG_ACCEPT_D_ENT = float(os.environ.get("RAG_ACCEPT_D_ENT", "0.05"))

    # debug
    FIRST_SAMPLE_DEBUG = int(os.environ.get("FIRST_SAMPLE_DEBUG", "1"))
    SC_DEBUG = int(os.environ.get("SC_DEBUG", "0"))
    SC_DEBUG_MAX = int(os.environ.get("SC_DEBUG_MAX", "200"))
    RAG_DEBUG = int(os.environ.get("RAG_DEBUG", "0"))
    RAG_DEBUG_MAX = int(os.environ.get("RAG_DEBUG_MAX", "200"))
    RAG_DEBUG_JSONL = safe_prepare_out_path(os.environ.get("RAG_DEBUG_JSONL", "rag_debug_choice.jsonl"))

    set_seed(seed)

    # retriever (path-based init)
    retriever = HybridWikiRetriever(
        faiss_path="faiss_index.index",
        docstore_path="docstore.pt",
        tfidf_vec_path="tfidf_vectorizer.joblib",
        tfidf_mat_path="tfidf_matrix.npz",
    )

    # model
    if _HAS_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_len,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_model:
        model = PeftModel.from_pretrained(model, adapter_model)

    model.eval()

    df = pd.read_csv(test_path)
    total = len(df)
    sc_budget = int(math.floor(total * MAX_SC_RATE))
    used_sc = 0
    accept_sc = 0

    out_rows = []
    cnt = defaultdict(int)

    # ensure jsonl clean per run
    try:
        if os.path.exists(RAG_DEBUG_JSONL):
            os.remove(RAG_DEBUG_JSONL)
    except Exception:
        pass

    pbar = tqdm(df.to_dict(orient="records"), total=total, desc="Infer (logit+SC+ChoiceRAG)", dynamic_ncols=True)

    for i, row in enumerate(pbar, start=1):
        qid = str(row.get("id", i))
        paragraph = row.get("paragraph", "")

        problems = row.get("problems")
        if isinstance(problems, str):
            try:
                problems = literal_eval(problems)
            except Exception:
                problems = None

        if isinstance(problems, list) and problems:
            p0 = problems[0]
        elif isinstance(problems, dict):
            p0 = problems
        else:
            out_rows.append({"id": qid, "answer": "1"})
            cnt["1"] += 1
            continue

        question = p0.get("question", "")
        question_plus = p0.get("question_plus")
        qp2 = row.get("question_plus")
        if isinstance(qp2, float) and np.isnan(qp2):
            qp2 = None
        if question_plus is None:
            question_plus = qp2

        choices = extract_choices(row, p0)
        if not choices:
            out_rows.append({"id": qid, "answer": "1"})
            cnt["1"] += 1
            continue

        allow_labels = list(range(1, len(choices) + 1))
        prompt = build_chatml_prompt(tokenizer, paragraph, question, question_plus, choices)

        # ---- Base logits ----
        pred_base, margin, ent, top1p = logits_label_stats(model, tokenizer, prompt, allow_labels, max_len)
        pred = pred_base

        # ---- SC gate ----
        t_margin = (margin < MARGIN_THRESHOLD)
        t_ent = (ent > ENTROPY_THRESHOLD)
        t_top1p = (top1p < TOP1P_THRESHOLD)
        triggers = int(t_margin) + int(t_ent) + int(t_top1p)

        base_confident = (margin >= CONF_MARGIN) and (top1p >= CONF_TOP1P) and (ent <= CONF_ENTROPY)
        need_sc = (not base_confident) and ((triggers >= SC_TRIGGER_K) or (margin < HARD_MARGIN))

        did_sc = False
        accepted_sc = False
        sc_label = None
        sc_gap = None
        accept_thr = None

        if need_sc and used_sc < sc_budget:
            is_hard = (margin < HARD_MARGIN)
            n_samples = SC_SAMPLES_HARD if is_hard else SC_SAMPLES_BASE
            temp = SC_TEMPERATURE_HARD if is_hard else SC_TEMPERATURE_BASE

            sc_label, sc_gap = self_consistency_logprob_sum(
                model, tokenizer, prompt, allow_labels, n_samples, temp, max_len
            )
            used_sc += 1
            did_sc = True

            accept_thr = SC_ACCEPT_MARGIN if is_hard else SC_ACCEPT_MARGIN_STRICT
            accepted_sc = (sc_gap >= accept_thr)
            if accepted_sc:
                pred = sc_label
                accept_sc += 1

            if SC_DEBUG and used_sc <= SC_DEBUG_MAX:
                print(
                    f"[SC] i={i}/{total} id={qid} pred_base={pred_base} m={margin:.4f} H={ent:.4f} p1={top1p:.4f} "
                    f"flags=(m:{int(t_margin)},H:{int(t_ent)},p1:{int(t_top1p)}) trig={triggers}/3 conf={int(base_confident)} "
                    f"sc=({sc_label},{sc_gap:.4f}) thr={accept_thr:.4f} accept={int(accepted_sc)} pred_now={pred}"
                )

        # ---- RAG gate ----
        gate = decide_use_rag(model, tokenizer, paragraph, question, question_plus, choices, max_len)
        gate_norm = str(gate).strip().upper()
        use_rag = gate_norm in {"USE", "USE_RAG", "YES", "Y", "TRUE", "1"}

        # if SC already accepted, default to skipping RAG to avoid harmful overwrite
        if accepted_sc:
            use_rag = False

        did_rag = False
        rag_ran = False
        rag_details = {"gate": gate, "use_rag": bool(use_rag)}

        best_choice_pack = None  # (score_key, label, stats, ctx, picked, query)
        choice_packs = []

        if use_rag:
            rag_ran = True
            # Choice-conditioned retrieval + sentence selection
            # For each choice, retrieve docs with choice-specific query, build ctx, score with logits.
            for idx, choice in enumerate(choices, start=1):
                try:
                    q_choice = make_choice_query(question, question_plus, choice)
                    docs = retriever.retrieve(
                        query=q_choice,
                        k_sparse=60,
                        k_dense=60,
                        k_final=max(5, int(RAG_MAX_DOCS)),
                        q_emb=None,
                    )
                    ctx, picked = build_rag_context(
                        docs,
                        q_choice,  # query_text aligned to retrieval query
                        top_docs=RAG_TOP_DOCS,
                        max_sentences=RAG_SENTENCES,
                        per_doc_sent_cap=RAG_PER_DOC_SENT,
                        min_doc_score=RAG_MIN_DOC_SCORE,
                        max_chars=RAG_CTX_CHARS,
                    )
                    if not ctx:
                        continue

                    rag_prompt = prompt.replace("정답:", f"\n\n[참고 문서]\n{ctx}\n\n정답:")
                    pred_i, m_i, e_i, p_i = logits_label_stats(model, tokenizer, rag_prompt, allow_labels, max_len)

                    pack = {
                        "choice_idx": idx,
                        "choice_text": choice,
                        "query": q_choice,
                        "ctx_chars": len(ctx),
                        "picked": picked,
                        "pred": pred_i,
                        "margin": m_i,
                        "entropy": e_i,
                        "top1p": p_i,
                    }
                    choice_packs.append(pack)
                except Exception:
                    # never crash inference because one choice retrieval failed
                    continue

            # pick best packs for decision (speed/robustness)
            # score_key depends on mode
            def _pack_key(p):
                if CHOICE_RAG_MODE == "best_by_top1p":
                    return (p["top1p"], p["margin"], -p["entropy"])
                # default: best_by_margin
                return (p["margin"], p["top1p"], -p["entropy"])

            choice_packs.sort(key=_pack_key, reverse=True)
            choice_packs = choice_packs[:max(1, int(CHOICE_RAG_TOPK))]

            # choose final candidate among top packs by the same key
            if choice_packs:
                best = choice_packs[0]
                pred_rag = best["pred"]
                m_rag = float(best["margin"])
                e_rag = float(best["entropy"])
                p_rag = float(best["top1p"])

                # delta-based accept to prevent noisy flips
                d_m = m_rag - float(margin)
                d_p = p_rag - float(top1p)
                d_e = float(ent) - e_rag  # decrease is good

                improve = (d_m >= RAG_ACCEPT_D_MARGIN) or (d_p >= RAG_ACCEPT_D_TOP1P) or (d_e >= RAG_ACCEPT_D_ENT)

                if improve:
                    pred = pred_rag
                    did_rag = True

                rag_details.update({
                    "used": bool(did_rag),
                    "base": {"pred": pred_base, "margin": float(margin), "entropy": float(ent), "top1p": float(top1p)},
                    "rag_best": {"pred": pred_rag, "margin": m_rag, "entropy": e_rag, "top1p": p_rag},
                    "delta": {"d_margin": float(d_m), "d_top1p": float(d_p), "d_ent_dec": float(d_e)},
                    "choice_best": {"choice_idx": int(best["choice_idx"]), "choice_text": best["choice_text"][:200]},
                    "choice_candidates": [
                        {"choice_idx": int(p["choice_idx"]), "pred": p["pred"], "margin": float(p["margin"]), "top1p": float(p["top1p"]), "entropy": float(p["entropy"]), "ctx_chars": int(p["ctx_chars"])}
                        for p in choice_packs
                    ],
                    "picked": best.get("picked", []),
                    "ctx_chars": int(best.get("ctx_chars", 0)),
                })
            else:
                rag_details.update({"used": False, "choice_candidates": [], "picked": [], "ctx_chars": 0})

        # ---- debug jsonl ----
        if rag_ran or RAG_DEBUG:
            rec = {
                "id": qid,
                "pred_final": pred,
                "pred_base": pred_base,
                "did_sc": bool(did_sc),
                "accepted_sc": bool(accepted_sc),
                "sc_label": sc_label,
                "sc_gap": sc_gap,
                "sc_thr": accept_thr,
                **rag_details,
            }
            try:
                with open(RAG_DEBUG_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass

            if RAG_DEBUG and i <= RAG_DEBUG_MAX:
                print(f"[RAG] i={i}/{total} id={qid} gate={gate_norm} use={int(use_rag)} used={int(did_rag)} pred={pred}")

        if i == 1 and FIRST_SAMPLE_DEBUG:
            print("\n[DEBUG] First sample prompt:\n")
            print(prompt)
            print("\n[DEBUG] Base:", f"pred={pred_base} m={margin:.4f} H={ent:.4f} p1={top1p:.4f}")
            print("[DEBUG] SC:", f"need={int(need_sc)} did={int(did_sc)} acc={int(accepted_sc)} sc=({sc_label},{sc_gap}) thr={accept_thr}")
            print("[DEBUG] Gate:", f"{gate_norm} -> use_rag={int(use_rag)}")
            print("[DEBUG] Final pred:", pred, "\n")

        out_rows.append({"id": qid, "answer": str(pred)})
        cnt[str(pred)] += 1

        pbar.set_postfix({
            "pred": str(pred),
            "sc": f"{used_sc}/{sc_budget}",
            "cnt1": cnt.get("1", 0),
            "cnt2": cnt.get("2", 0),
            "cnt3": cnt.get("3", 0),
            "cnt4": cnt.get("4", 0),
            "cnt5": cnt.get("5", 0),
        })

    pd.DataFrame(out_rows).to_csv(out_path, index=False)

    print(f"\n[DONE] wrote: {out_path}")
    print(f"[SC] used={used_sc}/{sc_budget} accept={accept_sc} accept_rate={(accept_sc/used_sc if used_sc else 0):.3f}")
    print("[CNT]", ", ".join([f"{k}:{v}" for k, v in sorted(cnt.items())]))


if __name__ == "__main__":
    main()
