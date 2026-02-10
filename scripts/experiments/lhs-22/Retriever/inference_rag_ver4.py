#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference: logit-based classification + optional SC (1-token sampling) WITH probability dump.

This is based on your aligned inference (ChatML via apply_chat_template) and keeps:
- ChatML prompt via tokenizer.apply_chat_template(..., add_generation_prompt=True)
- truncation_side="left" to preserve tail ("선택지", "정답:")
- logit-based label decision and SC budget logic

NEW:
- Saves per-label probabilities from the *base logits* at the answer position.
  Output CSV columns:
    id, answer, p1, p2, p3, p4, p5, pred_base, margin, entropy, top1p, did_sc, sc_label, sc_gap, sc_accepted

Env vars (same as before):
  BASE_MODEL, ADAPTER_MODEL
  TEST_PATH, OUT_PATH
  SEED, MAX_SEQ_LENGTH
  Gate thresholds: MARGIN_THRESHOLD, ENTROPY_THRESHOLD, TOP1P_THRESHOLD
  v3 guard: SC_TRIGGER_K, CONF_MARGIN, CONF_TOP1P, CONF_ENTROPY, HARD_MARGIN
  SC params: SC_SAMPLES_BASE, SC_TEMPERATURE_BASE, SC_SAMPLES_HARD, SC_TEMPERATURE_HARD
  SC accept: SC_ACCEPT_MARGIN, SC_ACCEPT_MARGIN_STRICT
  Budget: MAX_SC_RATE
  Debug: FIRST_SAMPLE_DEBUG, SC_DEBUG, SC_DEBUG_MAX
"""

# ---- Unsloth must be imported before transformers/torch model code ----
try:
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except Exception as e:
    FastLanguageModel = None
    print(f"[WARN] Unsloth import failed: {e}")
    HAS_UNSLOTH = False
# ----------------------------------------------------------------------

import re
import os
import time
import math
import random
from ast import literal_eval
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
from peft import PeftModel

# ===== Sentence-extractive context builder (lightweight, no extra models) =====
_SENT_SPLIT_RE = re.compile(r"(?:\.|\?|\!|다\.|요\.|니다\.|죠\.)\s+")

def split_sentences_safe(text: str):
    text = (text or "").strip()
    if not text:
        return []
    # Replace newlines with spaces
    text = re.sub(r"\s+", " ", text)
    # Split on common sentence enders (Korean + punctuation) without variable-width lookbehind
    parts = _SENT_SPLIT_RE.split(text)
    # Fallback if regex didn't split well
    if len(parts) <= 1:
        parts = re.split(r"[\n\r]+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 5]

_TOK_RE = re.compile(r"[가-힣A-Za-z0-9]+")

def _tokset(s: str):
    return set(_TOK_RE.findall((s or "").lower()))

def score_sentence_overlap(q_tokens: set, sent: str):
    # simple lexical overlap score (fast). You can replace with TF-IDF later.
    s_tokens = _tokset(sent)
    if not s_tokens:
        return 0.0
    inter = q_tokens.intersection(s_tokens)
    # weight longer matches slightly
    return float(len(inter)) + 0.15 * float(sum(len(t) for t in inter))

def build_extractive_ctx(query: str, docs, max_chars: int = 1400, per_doc_sents: int = 2):
    '''
    docs: list of dicts with keys: title, text (or chunk/text field)
    returns a compact evidence block with top-scoring sentences.
    '''
    q_tokens = _tokset(query)
    candidates = []
    for di, d in enumerate(docs):
        title = (d.get("title") or d.get("doc_title") or d.get("source") or f"doc{di}")
        body = d.get("text") or d.get("chunk") or d.get("content") or ""
        sents = split_sentences_safe(body)
        if not sents:
            continue
        # score each sentence; keep top per doc
        scored = []
        for si, s in enumerate(sents):
            sc = score_sentence_overlap(q_tokens, s)
            if sc > 0:
                scored.append((sc, si, s))
        scored.sort(reverse=True, key=lambda x: x[0])
        for sc, si, s in scored[:per_doc_sents]:
            candidates.append((sc, title, s))
    # global top
    candidates.sort(reverse=True, key=lambda x: x[0])
    lines = []
    used = 0
    for sc, title, s in candidates:
        line = f"- [{title}] {s}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return "\n".join(lines)
# ===========================================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RestrictTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allow_token_ids: set[int], mask_value: float = -1e9):
        self.allow_token_ids = allow_token_ids
        self.mask_value = mask_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, self.mask_value)
        idx = torch.tensor(list(self.allow_token_ids), device=scores.device, dtype=torch.long)
        mask[:, idx] = 0.0
        return scores + mask


def extract_choices(row: dict, p0: dict) -> list[str]:
    ch = p0.get("choices")
    if isinstance(ch, list) and len(ch) > 0:
        return ch
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
def logits_label_stats_and_probs(model, tokenizer, prompt: str, allow_labels: list[int], max_len: int):
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

    # score per label: max over candidate token ids
    label_scores = []
    label_cand_ids = []
    for l in allow_labels:
        cand_ids = _label_token_candidates(tokenizer, l)
        label_cand_ids.append(cand_ids)
        if not cand_ids:
            label_scores.append(float("-inf"))
            continue
        v = torch.tensor(cand_ids, device=scores.device, dtype=torch.long)
        label_scores.append(float(torch.max(scores[v]).item()))

    sub = torch.tensor(label_scores, device=scores.device)  # [K]
    probs = torch.softmax(sub, dim=-1)  # probability over labels (approx; max-cand trick)

    if sub.numel() >= 2:
        top2 = torch.topk(sub, k=2).values
        margin = float((top2[0] - top2[1]).item())
    else:
        margin = float("inf")

    entropy = float((-probs * torch.log(probs + 1e-12)).sum().item())
    top1p = float(probs.max().item())

    pred_idx = int(torch.argmax(sub).item())
    pred_label = str(allow_labels[pred_idx])

    # return python list for csv
    prob_list = [float(p.item()) for p in probs]
    return pred_label, margin, entropy, top1p, prob_list


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

    for _ in range(n_samples):
        gen = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=temperature,
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


def _safe_parse_problems(x):
    """
    problems column can be:
    - python literal string of list/dict
    - json-ish string
    - list/dict already
    """
    if isinstance(x, (list, dict)):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    # common: "[{...}]" python literal
    try:
        return literal_eval(s)
    except Exception:
        return None




def build_retrieval_query(paragraph: str, question: str, question_plus: str | None, choices: list[str],
                         use_paragraph: int = 1, para_chars: int = 450) -> str:
    qp = (question_plus or "").strip()
    ch = " / ".join([c.strip() for c in choices[:5]])
    parts = []
    if use_paragraph and paragraph:
        parts.append(paragraph.strip()[:para_chars])
    parts.append(question.strip())
    if qp:
        parts.append(qp[:200])
    if ch:
        parts.append(ch[:200])
    return "\n".join([p for p in parts if p])


def inject_ctx_before_answer(prompt: str, ctx: str) -> str:
    if not ctx:
        return prompt
    key = "정답:"
    j = prompt.rfind(key)
    block = "\n\n[참고 문서]\n" + ctx.strip() + "\n\n"
    if j == -1:
        return prompt + block
    return prompt[:j] + block + prompt[j:]


def main():
    base_model = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
    adapter_model = os.environ.get("ADAPTER_MODEL", "").strip()
    test_path = os.environ.get("TEST_PATH", "data/test.csv")
    out_path = os.environ.get("OUT_PATH", "submission_probs.csv")
    seed = int(os.environ.get("SEED", "42"))
    max_len = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))

    # gating thresholds
    MARGIN_THRESHOLD = float(os.environ.get("MARGIN_THRESHOLD", "0.75"))
    ENTROPY_THRESHOLD = float(os.environ.get("ENTROPY_THRESHOLD", "1.40"))
    TOP1P_THRESHOLD = float(os.environ.get("TOP1P_THRESHOLD", "0.40"))

    # v3 guard
    SC_TRIGGER_K = int(os.environ.get("SC_TRIGGER_K", "3"))
    CONF_MARGIN = float(os.environ.get("CONF_MARGIN", "2.0"))
    CONF_TOP1P = float(os.environ.get("CONF_TOP1P", "0.80"))
    CONF_ENTROPY = float(os.environ.get("CONF_ENTROPY", "0.60"))
    HARD_MARGIN = float(os.environ.get("HARD_MARGIN", "0.40"))

    # SC params
    SC_SAMPLES_BASE = int(os.environ.get("SC_SAMPLES_BASE", "9"))
    SC_TEMPERATURE_BASE = float(os.environ.get("SC_TEMPERATURE_BASE", "0.65"))
    SC_SAMPLES_HARD = int(os.environ.get("SC_SAMPLES_HARD", "13"))
    SC_TEMPERATURE_HARD = float(os.environ.get("SC_TEMPERATURE_HARD", "0.80"))

    # SC accept
    SC_ACCEPT_MARGIN = float(os.environ.get("SC_ACCEPT_MARGIN", "1.0"))
    SC_ACCEPT_MARGIN_STRICT = float(os.environ.get("SC_ACCEPT_MARGIN_STRICT", "1.8"))

    # budget
    MAX_SC_RATE = float(os.environ.get("MAX_SC_RATE", "0.10"))

    # ===== RAG gating =====
    RAG_MAX_RATE = float(os.environ.get("RAG_MAX_RATE", "0.25"))
    RAG_CONF_THR = float(os.environ.get("RAG_CONF_THR", "0.80"))
    RAG_ENTROPY_NORM_THR = float(os.environ.get("RAG_ENTROPY_NORM_THR", "0.65"))
    RAG_MARGIN_THR = float(os.environ.get("RAG_MARGIN_THR", "0.70"))
    RAG_TOP1P_THR = float(os.environ.get("RAG_TOP1P_THR", "0.45"))
    RAG_AVOID_HARD = int(os.environ.get("RAG_AVOID_HARD", "1"))

    # ===== RAG retrieval =====
    RAG_RUN = int(os.environ.get("RAG_RUN", "0"))
    RAG_INDEX = os.environ.get("RAG_INDEX", "wiki.faiss")
    RAG_DOCSTORE = os.environ.get("RAG_DOCSTORE", "docstore.pt")
    RAG_TFIDF_VEC = os.environ.get("RAG_TFIDF_VEC", "tfidf_vectorizer.joblib")
    RAG_TFIDF_MAT = os.environ.get("RAG_TFIDF_MAT", "tfidf_matrix.npz")
    RAG_DENSE_MODEL = os.environ.get("RAG_DENSE_MODEL", "BAAI/bge-m3")
    RAG_RERANKER_MODEL = os.environ.get("RAG_RERANKER_MODEL", "").strip()
    RAG_DENSE_DEVICE = os.environ.get("RAG_DENSE_DEVICE", "cpu")
    RAG_RERANK_DEVICE = os.environ.get("RAG_RERANK_DEVICE", "cpu")

    RAG_USE_RERANK = int(os.environ.get("RAG_USE_RERANK", "0"))
    RAG_USE_PARAGRAPH = int(os.environ.get("RAG_USE_PARAGRAPH", "1"))
    RAG_PARA_CHARS = int(os.environ.get("RAG_PARA_CHARS", "450"))
    RAG_CTX_MAX_CHARS = int(os.environ.get("RAG_CTX_MAX_CHARS", "2400"))
    RAG_DOC_MAX_CHARS = int(os.environ.get("RAG_DOC_MAX_CHARS", "550"))

    RAG_K_FINAL = int(os.environ.get("RAG_K_FINAL", "5"))
    RAG_K_DENSE = int(os.environ.get("RAG_K_DENSE", "30"))
    RAG_K_SPARSE = int(os.environ.get("RAG_K_SPARSE", "50"))
    RAG_K_FUSE = int(os.environ.get("RAG_K_FUSE", "80"))

    # debug
    FIRST_SAMPLE_DEBUG = int(os.environ.get("FIRST_SAMPLE_DEBUG", "1"))
    SC_DEBUG = int(os.environ.get("SC_DEBUG", "0"))
    SC_DEBUG_MAX = int(os.environ.get("SC_DEBUG_MAX", "200"))
    RAG_DEBUG = int(os.environ.get("RAG_DEBUG", "0"))
    RAG_DEBUG_MAX = int(os.environ.get("RAG_DEBUG_MAX", "10"))

    set_seed(seed)

    # load model/tokenizer
    if HAS_UNSLOTH:
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

    # Ensure Unsloth inference kernels (if available)
    if HAS_UNSLOTH:
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            pass

    model.eval()
    # ---- Unsloth/runtime diagnostics ----
    try:
        dev = next(model.parameters()).device
    except Exception:
        dev = 'unknown'
    print(f"[INFO] HAS_UNSLOTH={HAS_UNSLOTH}")
    print(f"[INFO] base_model={base_model} adapter={adapter_model or 'NONE'}")
    print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={dev}")
    if HAS_UNSLOTH and FastLanguageModel is not None:
        try:
            FastLanguageModel.for_inference(model)
            print("[INFO] FastLanguageModel.for_inference(model) applied")
        except Exception as e:
            print(f"[WARN] FastLanguageModel.for_inference failed: {e}")
    # -------------------------------------

    df = pd.read_csv(test_path)
    total = len(df)

    sc_budget = int(math.floor(total * MAX_SC_RATE))
    rag_budget = int(math.floor(total * RAG_MAX_RATE))

    used_sc = 0
    accept_sc = 0

    # RAG counters (separate)
    rag_flagged = 0
    rag_attempts = 0
    rag_used = 0
    rag_failures = 0
    rag_disabled = 0

    cnt = defaultdict(int)
    out_rows = []

    # lazy retriever
    retriever = None

    def _get_retriever():
        nonlocal retriever, rag_disabled  # main() 로컬을 참조해야 함
        if rag_disabled:
            return None
        if retriever is not None:
            return retriever
        try:
            # BM25 버전 리트리버로 교체
            from retriever_cpu_bm25 import HybridWikiRetriever

            retriever = HybridWikiRetriever(
                faiss_path=RAG_INDEX,
                docstore_path=RAG_DOCSTORE,

                # dense
                embed_model_name=RAG_DENSE_MODEL,
                device=RAG_DENSE_DEVICE,

                # ✅ BM25 artifacts (ENV 추가 없이, 고정 파일명으로 읽도록)
                bm25_vocab_path=os.environ.get("RAG_BM25_VOCAB", "bm25_vocab.joblib"),
                bm25_tf_path=os.environ.get("RAG_BM25_TF", "bm25_tf_matrix.npz"),
                bm25_meta_path=os.environ.get("RAG_BM25_META", "bm25_meta.npz"),

                # reranker (옵션)
                reranker_name=(RAG_RERANKER_MODEL if RAG_RERANKER_MODEL else "BAAI/bge-reranker-v2-m3"),
            )
            return retriever
        except Exception as e:
            rag_disabled = 1
            print(f"[WARN] Retriever init failed -> disabling RAG. err={e}")
            return None


    pbar = tqdm(
        df.to_dict(orient="records"),
        total=total,
        desc="Batch Inference (logit+SC+probs+RAG+Unsloth)",
        dynamic_ncols=True,
    )

    for i, row in enumerate(pbar, start=1):
        qid = str(row.get("id", i))
        paragraph = row.get("paragraph", "")

        problems = _safe_parse_problems(row.get("problems"))
        if problems is None:
            out_rows.append({"id": qid, "answer": "1"})
            cnt["1"] += 1
            continue
        p0 = problems[0] if isinstance(problems, list) else problems
        if not isinstance(p0, dict) or "question" not in p0:
            out_rows.append({"id": qid, "answer": "1"})
            cnt["1"] += 1
            continue

        question = p0["question"]
        question_plus = row.get("question_plus")
        if isinstance(question_plus, float) and np.isnan(question_plus):
            question_plus = None
        if question_plus is None:
            question_plus = p0.get("question_plus")

        choices = extract_choices(row, p0)
        if not choices:
            out_rows.append({"id": qid, "answer": "1"})
            cnt["1"] += 1
            continue

        allow_labels = list(range(1, len(choices) + 1))
        prompt0 = build_chatml_prompt(tokenizer, paragraph, question, question_plus, choices)

        t0_fw = time.time()
        pred_base, margin0, ent0, top1p0, prob_list0 = logits_label_stats_and_probs(
            model, tokenizer, prompt0, allow_labels, max_len
        )

        pred = pred_base
        margin, ent, top1p, prob_list = margin0, ent0, top1p0, prob_list0
        prompt = prompt0

        # ----- confidence features -----
        K = len(allow_labels)
        ent_norm = float(ent / math.log(K)) if K > 1 else 0.0
        conf_score = float(margin + top1p - ent_norm)

        # gating
        t_margin = (margin < MARGIN_THRESHOLD)
        t_ent = (ent > ENTROPY_THRESHOLD)
        t_top1p = (top1p < TOP1P_THRESHOLD)
        triggers = int(t_margin) + int(t_ent) + int(t_top1p)

        base_confident = (margin >= CONF_MARGIN) and (top1p >= CONF_TOP1P) and (ent <= CONF_ENTROPY)

        # ===== RAG gate =====
        rag_budget_left = (rag_attempts < rag_budget)
        is_hard_case = (margin < HARD_MARGIN)
        rag_avoid = (RAG_AVOID_HARD == 1 and is_hard_case)
        rag_uncertain = (
            (ent_norm > RAG_ENTROPY_NORM_THR)
            or (margin < RAG_MARGIN_THR)
            or (top1p < RAG_TOP1P_THR)
        )
        need_rag = (
            (not base_confident)
            and (conf_score < RAG_CONF_THR)
            and rag_uncertain
            and (not rag_avoid)
            and rag_budget_left
        )
        if need_rag:
            rag_flagged += 1

        did_rag = 0
        if need_rag and RAG_RUN == 1 and (not rag_disabled):
            rag_attempts += 1
            try:
                r = _get_retriever()
                if r is None:
                    raise RuntimeError("retriever unavailable")

                rq = build_retrieval_query(
                    paragraph, question, question_plus, choices,
                    use_paragraph=RAG_USE_PARAGRAPH, para_chars=RAG_PARA_CHARS
                )
                docs = r.retrieve(
                    query=rq,  # rq로 수정
                    k_final=RAG_K_FINAL,
                    k_dense=RAG_K_DENSE,
                    k_sparse=RAG_K_SPARSE,
                    k_fuse=RAG_K_FUSE,
                    use_rerank=bool(RAG_USE_RERANK),
                )

                ctx_parts = []
                total_chars = 0
                for d in docs:
                    title = str(d.get("title", ""))
                    txt = str(d.get("text", d.get("content", "")))
                    if not txt:
                        continue
                    snippet = txt.strip().replace("\n", " ")
                    snippet = snippet[:RAG_DOC_MAX_CHARS]
                    piece = (f"- {title}\n  {snippet}" if title else f"- {snippet}")
                    if total_chars + len(piece) > RAG_CTX_MAX_CHARS:
                        break
                    ctx_parts.append(piece)
                    total_chars += len(piece) + 1

                # # compact extractive ctx (문장 단위)
                # ctx = build_extractive_ctx(
                #     query=rq,
                #     docs=docs,
                #     max_chars=RAG_CTX_MAX_CHARS,
                #     per_doc_sents=2,
                # )
                ctx = "\n".join(ctx_parts).strip()
                if ctx:
                    prompt = inject_ctx_before_answer(prompt0, ctx)
                    pred, margin, ent, top1p, prob_list = logits_label_stats_and_probs(
                        model, tokenizer, prompt, allow_labels, max_len
                    )
                    did_rag = 1
                    rag_used += 1

                    if RAG_DEBUG and rag_used <= RAG_DEBUG_MAX:
                        titles = [str(d.get("title", "")) for d in docs[:RAG_K_FINAL]]
                        print(f"[RAG] i={i}/{total} id={qid} used=1 titles={titles}")
                else:
                    # no ctx -> treat as failure for monitoring
                    rag_failures += 1
            except Exception as e:
                rag_failures += 1
                if RAG_DEBUG and rag_failures <= 20:
                    import traceback
                    print(f"[WARN] RAG failed id={qid} type={type(e).__name__} repr={repr(e)}")
                    print(traceback.format_exc())

        # ===== decide SC on the FINAL prompt =====
        need_sc = (not base_confident) and ((triggers >= SC_TRIGGER_K) or (margin < HARD_MARGIN))

        did_sc = False
        accepted = False
        sc_label = ""
        sc_gap = 0.0
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
            accepted = (sc_gap >= accept_thr)
            if accepted:
                pred = sc_label
                accept_sc += 1

        if i == 1 and FIRST_SAMPLE_DEBUG:
            print("\n[DEBUG] First sample prompt:\n")
            print(prompt)
            print("\n[DEBUG] Base:", f"pred={pred_base} m={margin0:.4f} H={ent0:.4f} p1={top1p0:.4f}")
            print("[DEBUG] Gate:", f"need_rag={int(need_rag)} did_rag={did_rag} need_sc={int(need_sc)}")
            if did_sc:
                print("[DEBUG] SC:", f"sc=({sc_label},{sc_gap:.4f}) thr={accept_thr:.4f} accept={int(accepted)}")
            print("[DEBUG] Final pred:", pred, "\n")

        row_out = {
            "id": qid,
            "answer": str(pred),
            "pred_base": str(pred_base),
            "margin": float(margin),
            "entropy": float(ent),
            "entropy_norm": float(ent_norm),
            "conf_score": float(conf_score),
            "need_rag": int(need_rag),
            "did_rag": int(did_rag),
            "top1p": float(top1p),
            "did_sc": int(did_sc),
            "sc_label": str(sc_label) if did_sc else "",
            "sc_gap": float(sc_gap) if did_sc else 0.0,
            "sc_accepted": int(accepted),
        }
        for j, p in enumerate(prob_list, start=1):
            row_out[f"p{j}"] = float(p)
        for j in range(len(prob_list) + 1, 6):
            row_out[f"p{j}"] = np.nan

        out_rows.append(row_out)
        cnt[str(pred)] += 1

        pbar.set_postfix({
            "pred": str(pred),
            "rag": f"{rag_used}/{rag_budget} (flag {rag_flagged}, att {rag_attempts})",
            "sc": f"{used_sc}/{sc_budget}",
            "cnt1": cnt.get("1", 0),
            "cnt2": cnt.get("2", 0),
            "cnt3": cnt.get("3", 0),
            "cnt4": cnt.get("4", 0),
            "cnt5": cnt.get("5", 0),
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False)

    # RAG gate summary (same style as before)
    try:
        df_sum = out_df.copy()
        n = len(df_sum)
        if n > 0:
            import pandas as _pd
            mask_rag = (df_sum['need_rag'] == 1) if 'need_rag' in df_sum else _pd.Series(False, index=df_sum.index)
            mask_sc  = (df_sum['did_sc'] == 1) if 'did_sc' in df_sum else _pd.Series(False, index=df_sum.index)
            need_rag_rate = float(mask_rag.mean())
            did_sc_rate = float(mask_sc.mean())
            overlap_rate = float((mask_rag & mask_sc).mean())
            pcols = [c for c in df_sum.columns if c.startswith('p')]
            if pcols and 'need_rag' in df_sum:
                K_series = df_sum[pcols].notna().sum(axis=1)
                byK = df_sum.assign(K=K_series).groupby('K')['need_rag'].mean().to_dict()
            else:
                byK = {}
            rag_df = df_sum.loc[mask_rag]
            def _stat(s):
                return {'mean': float(s.mean()), 'p25': float(s.quantile(0.25)), 'p50': float(s.quantile(0.50)), 'p75': float(s.quantile(0.75))} if len(s)>0 else {}
            ent_stat = _stat(rag_df['entropy_norm']) if 'entropy_norm' in rag_df else {}
            mar_stat = _stat(rag_df['margin']) if 'margin' in rag_df else {}
            top1_stat = _stat(rag_df['top1p']) if 'top1p' in rag_df else {}
            print("\n===== END SUMMARY (RAG Gate Checks) =====")
            print(f"Samples: {n}")
            print(f"need_rag rate: {need_rag_rate:.3f} (target ~0.10~0.25)")
            print(f"did_sc rate:   {did_sc_rate:.3f} (budgeted)")
            print(f"RAG ∧ SC overlap: {overlap_rate:.3f} (target <0.30)")
            if byK:
                print('need_rag by #choices K:', ', '.join([f"K={k}:{v:.3f}" for k,v in sorted(byK.items())]))
            print('RAG subset stats:')
            print('  entropy_norm:', ent_stat)
            print('  margin:      ', mar_stat)
            print('  top1p:       ', top1_stat)
            print('=======================================\n')
    except Exception as e:
        print('[WARN] RAG summary skipped:', e)

    print(f"\n[DONE] wrote: {out_path}")
    print(f"[SC] used={used_sc}/{sc_budget} accept={accept_sc} accept_rate={(accept_sc/used_sc if used_sc else 0):.3f}")
    print(f"[RAG] used={rag_used}/{rag_budget} rate={(rag_used/total if total else 0):.3f}")
    print(f"[RAG] flagged={rag_flagged}/{rag_budget} attempts={rag_attempts} failures={rag_failures} disabled={rag_disabled}")
    print("[CNT]", ", ".join([f"{k}:{v}" for k, v in sorted(cnt.items())]))


if __name__ == "__main__":
    main()