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
import inspect
import re
import os
import time
import math
import random
from ast import literal_eval
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
from peft import PeftModel

# ===== Sentence-extractive context builder (lightweight, no extra models) =====
_SENT_SPLIT_RE = re.compile(r"(?:\.|\?|\!|다\.|요\.|니다\.|죠\.)\s+")

import re

# --- Discourse / reading-comprehension type filter ---
_DISCOURSE_PAT = re.compile(
    r"(서술|전개|구성|문단|요지|주제|제목|요약|핵심|근거|"
    r"표현\s*방식|논리\s*전개|비교|대조|나열|원인|결과|"
    r"문장\s*배열|빈칸|이어질|다음\s*내용|"
    r"태도|관점|의도|필자|글쓴이|서술자|심정|"
    r"맞는\s*것|옳은\s*것|틀린\s*것|적절(한|하지)\s*것)"
)

def rag_allowed_by_type(question, question_plus, choices):
    s = (question or "") + "\n" + (question_plus or "")
    s += "\n" + "\n".join(choices[:5])
    return not bool(_DISCOURSE_PAT.search(s))


_TOK_RE = re.compile(r"[가-힣A-Za-z0-9]+")

def _tokset(s: str):
    return set(_TOK_RE.findall((s or "").lower()))

def retrieval_quality_ok(query: str, docs, min_overlap: float = 0.06):
    q = _tokset(query)
    if not q:
        return False
    hits, total = 0, 0
    for d in docs[:5]:
        t = _tokset((d.get("text") or "")[:800])
        if not t:
            continue
        ov = len(q & t) / max(1, len(q))
        hits += (ov >= min_overlap)
        total += 1
    return (total >= 2) and (hits >= 2)


def accept_rag_update(
    base_pred, base_margin, base_ent, base_top1p,
    rag_pred,  rag_margin,  rag_ent,  rag_top1p,
    min_margin_gain=0.12, min_ent_drop=0.06, min_top1p_gain=0.05,
):
    margin_gain = rag_margin - base_margin
    ent_drop = base_ent - rag_ent
    top1p_gain = rag_top1p - base_top1p

    if rag_pred != base_pred:
        return (margin_gain >= min_margin_gain) and (ent_drop >= min_ent_drop)
    else:
        return (
            margin_gain >= min_margin_gain
            or ent_drop >= min_ent_drop
            or top1p_gain >= min_top1p_gain
        )



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

def build_choice_aware_ctx(
    question: str,
    question_plus: str | None,
    choices: list[str],
    docs,
    max_chars: int = 1800,
    per_choice_sents: int = 3,
    debug: bool = False,
    debug_topn: int = 2,
):
    candidates = []
    for di, d in enumerate(docs):
        title = (d.get("title") or d.get("doc_title") or d.get("source") or f"doc{di}")
        body = d.get("text") or d.get("chunk") or d.get("content") or ""
        for s in split_sentences_safe(body):
            candidates.append((title, s))

    if not candidates:
        return ("", {}) if debug else ""

    q_base = question.strip()
    if question_plus:
        q_base += "\n" + question_plus.strip()

    ctx_lines = []
    used = 0
    debug_map = {}

    for ci, ch in enumerate(choices, start=1):
        query_i = (q_base + "\n" + ch.strip()).strip()
        q_tokens = _tokset(query_i)

        scored = []
        for title, s in candidates:
            sc = score_sentence_overlap(q_tokens, s)
            if sc > 0:
                scored.append((sc, title, s))
        scored.sort(reverse=True, key=lambda x: x[0])

        if debug:
            debug_map[ci] = scored[:debug_topn]

        header = f"(선택지 {ci} 관련 근거)"
        if used + len(header) + 1 > max_chars:
            break
        ctx_lines.append(header)
        used += len(header) + 1

        take = 0
        for sc, title, s in scored:
            line = f"- [{title}] {s}"
            if used + len(line) + 1 > max_chars:
                break
            ctx_lines.append(line)
            used += len(line) + 1
            take += 1
            if take >= per_choice_sents:
                break

        if used + 1 <= max_chars:
            ctx_lines.append("")
            used += 1

    ctx = "\n".join(ctx_lines).strip()
    return (ctx, debug_map) if debug else ctx


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
    RAG_DEBUG_CHOICE = int(os.environ.get("RAG_DEBUG_CHOICE", "0"))
    RAG_DEBUG_CHOICE_MAX = int(os.environ.get("RAG_DEBUG_CHOICE_MAX", "3"))  # 몇 개 샘플만
    RAG_DEBUG_CHOICE_TOPN = int(os.environ.get("RAG_DEBUG_CHOICE_TOPN", "2"))  # 선택지당 몇 문장

    
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
    dense_encoder = None
    
    def _get_dense_encoder():
        nonlocal dense_encoder
        if dense_encoder is not None:
            return dense_encoder
        from sentence_transformers import SentenceTransformer
        dense_encoder = SentenceTransformer(RAG_DENSE_MODEL, device=RAG_DENSE_DEVICE)
        return dense_encoder
    
    def _get_retriever():
        nonlocal retriever, rag_disabled
        if rag_disabled:
            if RAG_DEBUG:
                print("[RAG-INIT] already disabled")
            return None
        if retriever is not None:
            return retriever

        try:
            from retriever_hybrid_faiss_tfidf import HybridWikiRetriever

            # 경로 로그
            if RAG_DEBUG:
                import os
                print("[RAG-INIT] paths")
                print("  RAG_INDEX   :", RAG_INDEX, "exists=", os.path.exists(RAG_INDEX))
                print("  RAG_DOCSTORE:", RAG_DOCSTORE, "exists=", os.path.exists(RAG_DOCSTORE))
                print("  RAG_TFIDF_VEC:", RAG_TFIDF_VEC, "exists=", os.path.exists(RAG_TFIDF_VEC))
                print("  RAG_TFIDF_MAT:", RAG_TFIDF_MAT, "exists=", os.path.exists(RAG_TFIDF_MAT))

            retriever = HybridWikiRetriever(
                faiss_path=RAG_INDEX,
                docstore_path=RAG_DOCSTORE,
                tfidf_vec_path=RAG_TFIDF_VEC,
                tfidf_mat_path=RAG_TFIDF_MAT,
            )

            if RAG_DEBUG:
                print("[RAG-INIT] success")
            return retriever

        except Exception as e:
            rag_disabled = 1
            import traceback
            print(f"[RAG-INIT-FAIL] type={type(e).__name__} repr={repr(e)}")
            print(traceback.format_exc())
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
        type_ok = rag_allowed_by_type(question, question_plus, choices)

        need_rag = (
            type_ok
            and (not base_confident)
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
                enc = _get_dense_encoder()
                q_emb = enc.encode([rq], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
                
                sig = inspect.signature(r.retrieve)
                supported = set(sig.parameters.keys())

                kwargs = {
                    "query": rq,
                    "q_emb": q_emb,
                    "k_final": RAG_K_FINAL,
                    "k_dense": RAG_K_DENSE,
                    "k_sparse": RAG_K_SPARSE,
                    "k_fuse": RAG_K_FUSE,
                    "use_rerank": bool(RAG_USE_RERANK),
                }
                
                # filter only supported args
                call_kwargs = {k: v for k, v in kwargs.items() if k in supported}

                # If retriever doesn't accept q_emb, drop it (sparse-only / internal embed)
                if ("q_emb" not in supported) and ("q_emb" in call_kwargs):
                    call_kwargs.pop("q_emb", None)

                docs = r.retrieve(**call_kwargs)
                
                
                # --- Debug: retrieval summary ---
                if RAG_DEBUG and rag_attempts <= RAG_DEBUG_MAX:
                    titles = [str(d.get("title", "")) for d in docs[:RAG_K_FINAL]]
                    print(f"[RAG-TRY] id={qid} type_ok={int(type_ok)} titles={titles}")

                # --- retrieval quality gate ---
                if not retrieval_quality_ok(rq, docs):
                    rag_failures += 1
                    if RAG_DEBUG and rag_attempts <= RAG_DEBUG_MAX:
                        print(f"[RAG-SKIP] id={qid} reason=low_retrieval_quality")
                else:
                    ctx = build_choice_aware_ctx(
                        question=question,
                        question_plus=question_plus,
                        choices=choices,
                        docs=docs,
                        max_chars=RAG_CTX_MAX_CHARS,
                        per_choice_sents=3,
                    )

                    if not ctx:
                        rag_failures += 1
                        if RAG_DEBUG and rag_attempts <= RAG_DEBUG_MAX:
                            print(f"[RAG-SKIP] id={qid} reason=empty_ctx")
                    else:
                        prompt_rag = inject_ctx_before_answer(prompt0, ctx)

                        rag_pred, rag_margin, rag_ent, rag_top1p, rag_probs = \
                            logits_label_stats_and_probs(
                                model, tokenizer, prompt_rag, allow_labels, max_len
                            )

                        accepted = accept_rag_update(
                            pred_base, margin0, ent0, top1p0,
                            rag_pred, rag_margin, rag_ent, rag_top1p
                        )

                        # --- Debug: accept / reject ---
                        if RAG_DEBUG and rag_attempts <= RAG_DEBUG_MAX:
                            print(
                                f"[RAG-EVAL] id={qid} "
                                f"base=({pred_base}, m={margin0:.2f}, H={ent0:.2f}, p={top1p0:.2f}) "
                                f"rag=({rag_pred}, m={rag_margin:.2f}, H={rag_ent:.2f}, p={rag_top1p:.2f}) "
                                f"accept={int(accepted)}"
                            )

                        if accepted:
                            pred, margin, ent, top1p, prob_list = (
                                rag_pred, rag_margin, rag_ent, rag_top1p, rag_probs
                            )
                            did_rag = 1
                            rag_used += 1
                            if RAG_DEBUG and rag_used <= RAG_DEBUG_MAX:
                                print(f"[RAG-USE] id={qid} final_pred={pred}")
                        else:
                            if RAG_DEBUG and rag_attempts <= RAG_DEBUG_MAX:
                                print(f"[RAG-REJECT] id={qid} keep_base={pred_base}")

            except Exception as e:
                rag_failures += 1
                if RAG_DEBUG and rag_attempts <= RAG_DEBUG_MAX:
                    import traceback
                    print(f"[RAG-ERROR] id={qid} err={repr(e)}")
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