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



# Optional: sentence-transformers for dense query embedding (FAISS part).
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (x / n).astype(np.float32, copy=False)


def _is_mostly_hangul(s: str, min_ratio: float = 0.55) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    hangul = sum(1 for ch in s if "가" <= ch <= "힣")
    letters = sum(1 for ch in s if ch.isalpha() or ("가" <= ch <= "힣"))
    if letters <= 0:
        return False
    return (hangul / max(1, letters)) >= float(min_ratio)
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
    # unique preserve order
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

    probs_list = probs.detach().float().cpu().tolist() 
    return pred_label, margin, entropy, top1p, probs_list



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
# Doc cleaning + sentence selection (noise reduction)
# --------------------
_url_re = re.compile(r"https?://\S+|www\.\S+")
_ref_re = re.compile(r"\[[0-9]+\]")
_ws_re = re.compile(r"[ \t]+")

def _clean_text(t: str) -> str:
    t = t or ""
    t = t.replace("\u00a0", " ")
    t = _url_re.sub(" ", t)
    t = _ref_re.sub(" ", t)
    # remove common wiki section markers
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

        # 문장 종결 패턴을 "보존"하면서 split
        ss = re.split(r"([\.?!]|다\.|요\.|임\.)\s+", blk)

        # ss = [문장, 구분자, 문장, 구분자, ...]
        buf = ""
        for i, tok in enumerate(ss):
            if i % 2 == 0:
                buf = tok
            else:
                sent = (buf + tok).strip()
                buf = ""
                if 15 <= len(sent) <= 260:
                    parts.append(sent)

        # 끝에 구분자 없는 잔여 문장
        if buf:
            sent = buf.strip()
            if 15 <= len(sent) <= 260:
                parts.append(sent)

    return parts

_tok_re = re.compile(r"[가-힣A-Za-z0-9]+")


# 시험/지문 보일러플레이트 + 기능어 중심(보수적으로; 너무 넓히면 recall 저하)
_STOPWORDS = {
    "다음", "다음은", "다음의", "다음과", "다음중", "중", "중에", "하나", "고르", "고르세요", "정답",
    "옳은", "옳지", "틀린", "아닌", "맞는", "것", "내용", "설명", "보기", "밑줄", "문장", "글",
    "지문", "질문", "선택지", "해당", "위", "아래", "이", "그", "저", "수", "등",
    "하고", "하며", "하여", "에서", "으로", "에게", "에서의", "및", "또는",
}

def _tokenize(s: str) -> list[str]:
    toks = [t.lower() for t in _tok_re.findall(s or "") if len(t) >= 2]
    out = []
    for t in toks:
        if t in _STOPWORDS:
            continue
        out.append(t)
    return out

def _tokset(s: str) -> set[str]:
    return set(_tokenize(s))

def build_rag_context(
    docs: list[dict],
    query_text: str,
    *,
    top_docs: int = 3,             # 추가: 상위 문서 수
    max_sentences: int = 10,
    per_doc_sent_cap: int = 3,
    min_doc_score: float = 1.5,    # 추가: 문서 점수 하한(노이즈 컷)
    max_chars: int = 1800,
) -> tuple[str, list[dict]]:

    """
    Returns:
      ctx_str: selected sentences with titles
      picked: debug info list with {id,title,chosen_sentences,score}
    """
    
    qset = _tokset(query_text)
    doc_candidates = []  # (doc_score, did, title, top_scored_sentences)
    if not qset:
        return "", []

    for d in docs:
        title = (d.get("title") or "")
        if any(k in title for k in BAN_TITLE_KEYWORDS):
            continue

        # hard filter: 영문/잡문서 드리프트 방지
        if not _is_mostly_hangul(
            title,
            min_ratio=float(os.environ.get("RAG_MIN_HANGUL_RATIO", "0.55")),
        ):
            continue

        raw = d.get("text") or ""
        cleaned = _clean_text(raw)
        sents = _sent_split(cleaned)

        scored = []
        for s in sents:
            sset = _tokset(s)
            if not sset:
                continue

            inter = (qset & sset)
            if not inter:
                continue

            # 가중 overlap: 긴/정보성 토큰 우대
            ov_w = 0.0
            for tok in inter:
                ov_w += 1.0 + 0.15 * min(12, len(tok))

            score = ov_w / (1.0 + 0.01 * len(s))
            scored.append((float(score), s))

        # 문서 내에서 유효 문장 없음
        if not scored:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored[:per_doc_sent_cap]

        # 문서 점수: 상위 문장 점수 합
        doc_score = float(sum(sc for sc, _ in top_scored))
        if doc_score < min_doc_score:
            continue

        did = int(d.get("id", -1))
        doc_candidates.append((doc_score, did, title, top_scored))
    # ✅ Top N 문서만 선택
    doc_candidates.sort(key=lambda x: x[0], reverse=True)
    doc_candidates = doc_candidates[:max(1, int(top_docs))]

    if not doc_candidates:
        return "", []

    sent_pool = []  # (score, did, title, sent)
    for doc_score, did, title, top_scored in doc_candidates:
        for score, s in top_scored:
            sent_pool.append((score, did, title, s))

    sent_pool.sort(key=lambda x: x[0], reverse=True)
    picked_sents = sent_pool[:max_sentences]

    # Build ctx
    lines = []
    picked_debug = []
    # group by doc id for cleaner ctx
    by_doc = {}
    for score, did, title, s in picked_sents:
        by_doc.setdefault((did, title), []).append((score, s))

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
    # include choices to reduce ambiguity + improve overlap
    ch = " ".join(choices[:5])
    qp = question_plus or ""
    return f"{question}\n{qp}\n{paragraph[:800]}\n{ch}"


# --------------------
# Retrieval query building (domain-boost + multi-query RRF)
# --------------------
_DOMAIN_BOOST = {
    "law": ["민법", "형법", "법정대리인", "의사표시", "취소", "불법행위", "책임", "계약"],
    "politics": ["헌법", "국회", "대통령", "행정부", "사법부", "헌법재판소", "권력", "민주주의"],
    "economy": ["수요", "공급", "가격", "시장", "인플레이션", "실업", "재정", "통화"],
    "science": ["원리", "에너지", "힘", "압력", "확산", "농도", "반응", "세포", "유전", "지구"],
    "language": ["문법", "어휘", "문장성분", "피동", "사동", "어법", "고쳐쓰기", "추론"],
    "literature": ["화자", "서술", "상징", "운율", "갈등", "구성", "시점", "고전소설"],
}

def _detect_domain(text: str) -> str:
    """보수적 규칙: 강한 신호가 2개 이상일 때만 도메인 판정."""
    t = (text or "")
    hits = {
        "law": sum(k in t for k in ["법정대리인", "의사표시", "불법행위", "취소", "손해배상", "미성년자", "계약"]),
        "politics": sum(k in t for k in ["헌법", "국회", "대통령", "헌법재판소", "선거", "국정", "권력"]),
        "economy": sum(k in t for k in ["수요", "공급", "시장", "가격", "인플레이션", "실업", "환율"]),
        "science": sum(k in t for k in ["모세관", "표면장력", "마찰", "전류", "압력", "에너지", "세포", "유전", "광합성"]),
        "language": sum(k in t for k in ["고쳐", "문법", "어휘", "문장", "피동", "사동", "추론", "접속"]),
        "literature": sum(k in t for k in ["화자", "서술", "운율", "상징", "갈등", "시점", "고전", "인물"]),
    }
    dom, sc = max(hits.items(), key=lambda x: x[1])
    return dom if sc >= 2 else "general"

def _extract_keywords(text: str, k: int = 12) -> list[str]:
    toks = _tokenize(text)
    # 길이 긴 토큰 우선 + 중복 제거
    toks = sorted(set(toks), key=lambda x: (-len(x), x))
    return toks[:max(0, int(k))]

def make_retrieval_queries(paragraph: str, question: str, question_plus: str | None, choices: list[str]) -> list[str]:
    qp = question_plus or ""
    base = f"{question} {qp}".strip()

    # 키워드: 질문/보기 + 선택지(짧게)
    kw_src = f"{question} {qp} " + " ".join(choices[:5])
    kws = _extract_keywords(kw_src, k=int(os.environ.get("RAG_KW_TOPK", "12")))
    kw_q = " ".join(kws)

    # 도메인 부스트(확실할 때만)
    dom = _detect_domain(f"{question}\n{qp}\n{paragraph[:600]}\n" + " ".join(choices[:5]))
    boosts = " ".join(_DOMAIN_BOOST.get(dom, [])[: int(os.environ.get("RAG_DOMAIN_BOOST_K", "6"))])

    qs = []
    if base:
        qs.append(base)
    if kw_q:
        qs.append(f"{kw_q} {boosts}".strip())
    else:
        if boosts:
            qs.append(boosts)

    # 너무 긴 쿼리는 sparse를 오염시켜서 잘라줌
    out = []
    for q in qs:
        q = " ".join(q.split())
        if len(q) > 300:
            q = q[:300]
        if q and q not in out:
            out.append(q)
    return out[: int(os.environ.get("RAG_MULTI_Q", "2"))]

def _rrf_merge(rank_lists: list[list[int]], k_final: int, k0: int = 60) -> list[int]:
    score = {}
    k0 = max(1, int(k0))
    for lst in rank_lists:
        for r, did in enumerate(lst, start=1):
            did = int(did)
            score[did] = score.get(did, 0.0) + 1.0 / (k0 + r)
    items = sorted(score.items(), key=lambda x: (-x[1], x[0]))
    return [did for did, _ in items[: int(k_final)]]

def multi_retrieve_rrf(
    retriever: HybridWikiRetriever,
    embedder,
    queries: list[str],
    *,
    k_sparse: int,
    k_dense: int,
    k_final: int,
    rrf_k0: int = 60,
    use_e5_prefix: bool = True,
) -> list[dict]:
    """여러 쿼리로 retrieve → doc_id 랭킹들을 RRF로 합치고, 최종 doc dict로 반환."""
    rank_lists = []
    per_query_docs = []
    for q in queries:
        q_emb = None
        if embedder is not None:
            q2 = q
            if use_e5_prefix and ("e5" in str(getattr(embedder, "model_name_or_path", "")).lower()):
                q2 = "query: " + q2
            try:
                X = embedder.encode([q2], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
                q_emb = _l2_normalize(X)[0]
            except Exception:
                q_emb = None
        docs = retriever.retrieve(query=q, k_sparse=k_sparse, k_dense=k_dense, k_final=max(5, int(k_final)), q_emb=q_emb)
        per_query_docs.append(docs)
        rank_lists.append([int(d["id"]) for d in docs])
    fused = _rrf_merge(rank_lists, k_final=int(k_final), k0=int(rrf_k0))

    # pick first occurrence metadata
    by_id = {}
    for docs in per_query_docs:
        for d in docs:
            did = int(d.get("id", -1))
            if did not in by_id:
                by_id[did] = d
    return [by_id[did] for did in fused if did in by_id]

BAN_TITLE_KEYWORDS = [
    "대학수학능력시험",
    "수능",
    "모의고사",
    "평가원",
    "정답",
    "해설",
    "기출",
]

def main():
    # ===== env =====
    base_model = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
    adapter_model = os.environ.get("ADAPTER_MODEL", "").strip()
    test_path = os.environ.get("TEST_PATH", "data/test.csv")
    out_path = os.environ.get("OUT_PATH", "submission_sc_rag.csv")
    seed = int(os.environ.get("SEED", "42"))
    max_len = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))

    # ----- SC thresholds (keep identical to your aligned SC script) -----
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

    # ----- RAG ctx controls -----
    RAG_MAX_DOCS = int(os.environ.get("RAG_MAX_DOCS", "20"))
    RAG_SENTENCES = int(os.environ.get("RAG_SENTENCES", "10"))
    RAG_PER_DOC_SENT = int(os.environ.get("RAG_PER_DOC_SENT", "3"))
    RAG_CTX_CHARS = int(os.environ.get("RAG_CTX_CHARS", "1800"))

    # debug
    FIRST_SAMPLE_DEBUG = int(os.environ.get("FIRST_SAMPLE_DEBUG", "1"))
    SC_DEBUG = int(os.environ.get("SC_DEBUG", "0"))
    SC_DEBUG_MAX = int(os.environ.get("SC_DEBUG_MAX", "200"))
    RAG_DEBUG = int(os.environ.get("RAG_DEBUG", "0"))
    RAG_DEBUG_MAX = int(os.environ.get("RAG_DEBUG_MAX", "200"))
    RAG_DEBUG_JSONL = os.environ.get("RAG_DEBUG_JSONL", "rag_debug.jsonl")
    out_probs_path = os.environ.get("OUT_PROBS_PATH", "").strip()
    dump_probs = bool(out_probs_path)
    prob_rows = []
    set_seed(seed)

    # retriever (path-based init)
    retriever = HybridWikiRetriever(
        faiss_path="faiss_index.index",
        docstore_path="docstore.pt",
        tfidf_vec_path="tfidf_vectorizer.joblib",
        tfidf_mat_path="tfidf_matrix.npz",
    )
    # Query embedder for dense retrieval (must match FAISS build model, e.g., intfloat/multilingual-e5-base)
    embedder = None
    if _HAS_ST and int(os.environ.get("RAG_QUERY_EMBED", "1")) == 1:
        emb_model = os.environ.get("RAG_EMBED_MODEL", "intfloat/multilingual-e5-base")
        emb_device = os.environ.get("RAG_EMBED_DEVICE", "cpu")
        try:
            embedder = SentenceTransformer(emb_model, device=emb_device)
            # keep name for prefix detection
            embedder.model_name_or_path = emb_model  # type: ignore
        except Exception:
            embedder = None

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

    pbar = tqdm(df.to_dict(orient="records"), total=total, desc="Infer (logit+SC+RAG)", dynamic_ncols=True)

    for i, row in enumerate(pbar, start=1):
        qid = str(row["id"])
        paragraph = row["paragraph"]

        problems = row.get("problems")
        if isinstance(problems, str):
            problems = literal_eval(problems)

        if isinstance(problems, list):
            p0 = problems[0]
        elif isinstance(problems, dict):
            p0 = problems
        else:
            out_rows.append({"id": qid, "answer": "1"})
            cnt["1"] += 1
            continue

        question = p0.get("question", "")
        question_plus = p0.get("question_plus")
        # some datasets have question_plus top-level
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
        pred_base, margin, ent, top1p, probs_base = logits_label_stats(
            model, tokenizer, prompt, allow_labels, max_len
        )
        pred = pred_base

        # ---- SC gate (same as aligned script) ----
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

        # ---- RAG gate (LLM) ----
        gate = decide_use_rag(model, tokenizer, paragraph, question, question_plus, choices, max_len)
        gate_norm = str(gate).strip().upper()
        use_rag = gate_norm in {"USE", "USE_RAG", "YES", "Y", "TRUE", "1"}

        did_rag = False
        rag_used = False
        rag_details = {"gate": gate, "use_rag": bool(use_rag)}

        if use_rag:
            # build query text and multi-queries (domain-boost)
            qtext = make_query_text(paragraph, question, question_plus, choices)
            queries = make_retrieval_queries(paragraph, question, question_plus, choices)

            # retrieve candidate docs (hybrid: TF-IDF + FAISS dense if available) with multi-query RRF
            docs = multi_retrieve_rrf(
                retriever,
                embedder=embedder,
                queries=queries,
                k_sparse=int(os.environ.get("RAG_K_SPARSE", "80")),
                k_dense=int(os.environ.get("RAG_K_DENSE", "80")),
                k_final=max(5, int(RAG_MAX_DOCS)),
                rrf_k0=int(os.environ.get("RAG_RRF_K0", "60")),
                use_e5_prefix=True,
            )

            # build condensed context from sentences
            ctx, picked = build_rag_context(
                docs,
                qtext,
                max_sentences=RAG_SENTENCES,
                per_doc_sent_cap=RAG_PER_DOC_SENT,
                max_chars=RAG_CTX_CHARS,
            )

            if ctx:
                # Build RAG prompt by inserting ctx before "정답:"
                rag_prompt = prompt.replace("정답:", f"\n\n[참고 문서]\n{ctx}\n\n정답:")

                pred_rag, m_rag, e_rag, p_rag, probs_rag = logits_label_stats(
                    model, tokenizer, rag_prompt, allow_labels, max_len
                )


            # accept only if improvement is meaningful (prevents noisy flips)
            d_top1p = float(p_rag - top1p)
            d_margin = float(m_rag - margin)
            d_ent = float(ent - e_rag)  # decrease is good

            thr_top1p = float(os.environ.get("RAG_ACCEPT_DELTA_TOP1P", "0.04"))
            thr_margin = float(os.environ.get("RAG_ACCEPT_DELTA_MARGIN", "0.25"))
            thr_ent = float(os.environ.get("RAG_ACCEPT_DELTA_ENTROPY", "0.10"))

            improve = (d_top1p >= thr_top1p) or (d_margin >= thr_margin) or (d_ent >= thr_ent)
            if improve:
                pred = pred_rag
                did_rag = True
                rag_used = True
                rag_details.update({
                    "used": bool(did_rag),
                    "base": {"pred": pred_base, "margin": margin, "entropy": ent, "top1p": top1p},
                    "rag": {"pred": pred_rag, "margin": m_rag, "entropy": e_rag, "top1p": p_rag},
                    "picked": picked,               # sentence-level, already denoised
                    "ctx_chars": len(ctx),
                })
            else:
                rag_details.update({"used": False, "picked": [], "ctx_chars": 0})

        # ---- debug jsonl ----
        if rag_used or RAG_DEBUG:
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
            with open(RAG_DEBUG_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

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
        if dump_probs:
            # 기본은 base 확률
            final_probs = probs_base

            # RAG가 실제로 사용되었으면 RAG 프롬프트 기준 확률 사용
            if did_rag:
                try:
                    final_probs = probs_rag
                except Exception:
                    final_probs = probs_base

            # p1~p5로 패딩
            ps = [0.0] * 5
            for j, pj in enumerate(final_probs[:5]):
                ps[j] = float(pj)

            prob_rows.append({
                "id": qid,
                "p1": ps[0],
                "p2": ps[1],
                "p3": ps[2],
                "p4": ps[3],
                "p5": ps[4],
            })

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
    if dump_probs:
        os.makedirs(os.path.dirname(out_probs_path) or ".", exist_ok=True)
        pd.DataFrame(prob_rows).to_csv(out_probs_path, index=False)
        print(f"[DONE] wrote probs: {out_probs_path}")

    print(f"\n[DONE] wrote: {out_path}")
    print(f"[SC] used={used_sc}/{sc_budget} accept={accept_sc} accept_rate={(accept_sc/used_sc if used_sc else 0):.3f}")
    print("[CNT]", ", ".join([f"{k}:{v}" for k, v in sorted(cnt.items())]))


if __name__ == "__main__":
    main()
