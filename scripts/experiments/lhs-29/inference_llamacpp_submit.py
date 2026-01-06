#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, json, time, datetime
from typing import Any, Dict, Optional, List
from collections import Counter

import pandas as pd
from tqdm import tqdm

from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class TeeWriter:
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]
    def write(self, s: str) -> None:
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
    def flush(self) -> None:
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass
    def isatty(self) -> bool:
        for st in self.streams:
            fn = getattr(st, "isatty", None)
            if callable(fn) and fn():
                return True
        return False


def build_prompt(row: Dict[str, Any]) -> str:
    paragraph = str(row.get("paragraph", "") or "").strip()
    problems  = row.get("problems", "")
    q         = str(row.get("question", "") or "").strip()
    choices   = row.get("choices", None)

    parts = ["지문을 읽고 질문의 답을 구하세요.\n\n"]

    if paragraph:
        parts.append("지문:\n")
        parts.append(paragraph)
        parts.append("\n\n")

    if q and choices:
        parts.append("질문:\n")
        parts.append(q)
        parts.append("\n\n선택지:\n")
        if isinstance(choices, (list, tuple)):
            for i, c in enumerate(choices, 1):
                parts.append(f"{i}. {str(c).strip()}\n")
        else:
            parts.append(str(choices).strip() + "\n")
        parts.append("\n")
    else:
        pt = problems.strip() if isinstance(problems, str) else str(problems).strip()
        if pt:
            parts.append(pt)
            parts.append("\n\n")

    parts.append("1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답: ")
    return "".join(parts)


def extract_answer_1to5(text: str) -> Optional[str]:
    if not text:
        return None
    # "정답:" 이후를 우선
    idx = text.rfind("정답:")
    tail = text[idx + len("정답:"):] if idx != -1 else text
    m = re.search(r"([1-5])", tail)
    if m:
        return m.group(1)
    # fallback: 끝부분에서 찾기
    m = re.search(r"([1-5])", text[-200:])
    return m.group(1) if m else None


def http_post_json(url: str, payload: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)

import math

def _softmax_logps(logps: Dict[str, float]) -> Dict[str, float]:
    # logps: {"1": lp1, ...}
    if not logps:
        return {}
    m = max(logps.values())
    exps = {k: math.exp(v - m) for k, v in logps.items()}
    z = sum(exps.values())
    if z <= 0:
        return {}
    return {k: exps[k] / z for k in exps}

def extract_p1to5_from_logprobs(raw: Dict[str, Any], fallback_ans: str) -> Dict[str, float]:
    """
    raw: llama-server /v1/completions response json
    return: {"p1":..., "p2":..., "p3":..., "p4":..., "p5":...} (sum ~ 1)
    """
    # 기본 one-hot fallback
    onehot = {f"p{i}": (1.0 if str(i) == fallback_ans else 0.0) for i in range(1, 6)}

    try:
        choices = raw.get("choices") or []
        if not choices:
            return onehot
        lp = choices[0].get("logprobs")
        if not lp:
            return onehot

        # OpenAI-style: lp["top_logprobs"] is a list of dicts (per token position)
        top_list = lp.get("top_logprobs")
        if not top_list or not isinstance(top_list, list) or not top_list[0]:
            return onehot

        top0 = top_list[0]  # dict: token -> logprob
        # digit tokens만 수집
        digit_logps = {}
        for d in ["1", "2", "3", "4", "5"]:
            if d in top0 and isinstance(top0[d], (int, float)):
                digit_logps[d] = float(top0[d])

        if not digit_logps:
            return onehot

        probs = _softmax_logps(digit_logps)
        out = {f"p{i}": float(probs.get(str(i), 0.0)) for i in range(1, 6)}

        # 누락된 digit이 있으면 합이 1보다 작을 수 있음 -> 재정규화
        s = sum(out.values())
        if s > 0:
            out = {k: v / s for k, v in out.items()}
        else:
            out = onehot
        return out

    except Exception:
        return onehot



def completion_call(base_url: str, prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
    """
    return: (text, raw_json, used_endpoint)
    """

    # 1) OpenAI-compatible
    MCQ_1TO5_GRAMMAR = r'root ::= ("1" | "2" | "3" | "4" | "5")'
    url1 = base_url.rstrip("/") + "/v1/completions"
    payload1 = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "grammar": MCQ_1TO5_GRAMMAR,
        "stop": ["\n"],
        "logprobs": 10,   # <= 추가 (top_logprobs를 받아 p1..p5 구성 시도)
    }
    try:
        j = http_post_json(url1, payload1, timeout=600)
        choices = j.get("choices") or []
        if choices and isinstance(choices, list):
            text = str(choices[0].get("text", "") or "")
            return text, j, "/v1/completions"
    except (HTTPError, URLError, json.JSONDecodeError):
        pass

    # 2) legacy
    url2 = base_url.rstrip("/") + "/completion"
    payload2 = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "grammar": MCQ_1TO5_GRAMMAR,
        "stop": ["\n"],
        "logprobs": 10,   # <= 추가
    }


    try:
        j = http_post_json(url2, payload2, timeout=600)
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Both endpoints failed: /v1/completions and /completion; last_err={repr(e)}")


    for key in ("content", "completion", "text"):
        if key in j and isinstance(j[key], str):
            return j[key], j, "/completion"

    choices = j.get("choices") or []
    if choices and isinstance(choices, list):
        text = str(choices[0].get("text", "") or "")
        return text, j, "/completion"

    raise RuntimeError(f"Unexpected server response keys: {list(j.keys())}")


def wait_server(base_url: str, tries: int = 120, sleep_s: float = 1.0) -> None:
    # 서버 health endpoint는 버전별로 다를 수 있어, /v1/models -> /health -> / 를 순서대로 찍어봄
    candidates = [
        base_url.rstrip("/") + "/v1/models",
        base_url.rstrip("/") + "/health",
        base_url.rstrip("/") + "/",
    ]
    for _ in range(tries):
        for u in candidates:
            try:
                req = Request(u, headers={"Accept": "application/json"})
                with urlopen(req, timeout=5) as resp:
                    _ = resp.read(200)
                return
            except Exception:
                pass
        time.sleep(sleep_s)
    raise RuntimeError(f"Server not responding: {base_url}")


def main():
    BASE_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080").strip()
    TEST_PATH = os.environ.get("TEST_PATH", "/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv")
    OUT_PATH  = os.environ.get("OUT_PATH",  "/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/submission_llamacpp.csv")
    LOG_PATH  = os.environ.get("LOG_PATH",  "./logs/llamacpp_http_infer.log")

    THREADS = int(os.environ.get("THREADS", "8"))  # 서버 쪽 옵션이므로 여기선 참고용
    CTX = int(os.environ.get("CTX", "4096"))
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8"))
    TEMP = float(os.environ.get("TEMP", "0.0"))
    TOP_P = float(os.environ.get("TOP_P", "1.0"))
    TOP_K = int(os.environ.get("TOP_K", "1"))

    DUMP_PROBS = os.environ.get("DUMP_PROBS", "0").strip() == "1"
    PROBS_OUT_PATH = os.environ.get(
        "PROBS_OUT_PATH",
        os.path.splitext(OUT_PATH)[0] + "_probs.csv"
    )
    probs_rows: List[Dict[str, Any]] = []
    
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    log_f = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    tee = TeeWriter(sys.stderr, log_f)

    
    
    def log(msg: str) -> None:
        tee.write(msg + "\n")
        tee.flush()

    log_f.write("\n================ RUN START ================\n")
    log_f.write(f"time: {now_iso()}\n")
    log_f.write(f"server: {BASE_URL}\n")
    log_f.write(f"test  : {TEST_PATH}\n")
    log_f.write(f"out   : {OUT_PATH}\n")
    log_f.write(f"ctx={CTX} max_tokens={MAX_TOKENS} temp={TEMP} top_p={TOP_P} top_k={TOP_K}\n")
    log_f.write("===========================================\n\n")

    log("[INFO] waiting for llama-server...")
    wait_server(BASE_URL)
    log("[INFO] server is up")

    df = pd.read_csv(TEST_PATH)
    rows = df.to_dict(orient="records")

    cnt = Counter()
    out_rows: List[Dict[str, Any]] = []

    pbar = tqdm(rows, total=len(rows), desc="llama-server HTTP inference",
                dynamic_ncols=True, file=tee, mininterval=0.5, ascii=True, leave=True)

    for i, row in enumerate(pbar, 1):
        qid = str(row.get("id", f"row-{i-1}"))
        prompt = build_prompt(row)

        log_f.write(f"\n----- SAMPLE {i}/{len(rows)} id={qid} -----\n")
        log_f.write(f"[PROMPT_CHARS] {len(prompt)}\n")

        last_err = None
        text = ""
        raw = None
        used_ep = None

        for attempt in range(3):
            try:
                text, raw, used_ep = completion_call(BASE_URL, prompt, MAX_TOKENS, TEMP, TOP_P, TOP_K)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(1.0 + attempt)

        if last_err is not None:
            log_f.write(f"[ERROR] completion failed: {repr(last_err)}\n")
            ans = "1"
            out_rows.append({"id": qid, "answer": ans})
            cnt["other"] += 1
            continue

        # ====== 디버그 로그(핵심) ======
        log_f.write(f"[ENDPOINT] {used_ep}\n")
        log_f.write(f"[RAW_JSON_KEYS] {list(raw.keys()) if isinstance(raw, dict) else type(raw)}\n")
        log_f.write("--- RAW_JSON BEGIN ---\n")
        log_f.write(json.dumps(raw, ensure_ascii=False) + "\n")
        log_f.write("--- RAW_JSON END ---\n")

        # text 원문 확인
        log_f.write("--- RAW_TEXT BEGIN ---\n")
        log_f.write(repr(text) + "\n")  # 빈 문자열/공백/개행까지 보이게
        log_f.write("--- RAW_TEXT END ---\n")

        # 파싱 근거를 남기기 위해 '정답:' 이후 tail을 따로 출력
        probe = "정답:" + (text or "")
        idx = probe.rfind("정답:")
        tail = probe[idx + len("정답:"):] if idx != -1 else probe
        log_f.write(f"[PARSE_TAIL_REPR] {repr(tail[:80])}\n")  # 앞 80자만

        ans = (text or "").strip()
        if ans not in {"1","2","3","4","5"}:
            log_f.write("[PARSE_FAIL] unexpected -> fallback 1\n")
            ans = "1"
        # =============================

        log_f.write(f"parsed_answer: {ans}\n")
        if DUMP_PROBS and isinstance(raw, dict):
            p = extract_p1to5_from_logprobs(raw, ans)
            probs_rows.append({"id": qid, **p})

        if ans in {"1","2","3","4","5"}:
            cnt[ans] += 1
        else:
            cnt["other"] += 1
            ans = "1"

        out_rows.append({"id": qid, "answer": ans})
        pbar.set_postfix(ans=ans, c1=cnt["1"], c2=cnt["2"], c3=cnt["3"], c4=cnt["4"], c5=cnt["5"])

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    pd.DataFrame(out_rows).to_csv(OUT_PATH, index=False)
    if DUMP_PROBS:
        os.makedirs(os.path.dirname(PROBS_OUT_PATH) or ".", exist_ok=True)
        pd.DataFrame(probs_rows).to_csv(PROBS_OUT_PATH, index=False)
        print(f"[DONE] wrote probs: {PROBS_OUT_PATH}")

    log_f.write("\n================ RUN END ==================\n")
    log_f.write(f"time: {now_iso()}\n")
    log_f.write(f"total: {len(rows)}\n")
    log_f.write(f"final_count: 1={cnt['1']} 2={cnt['2']} 3={cnt['3']} 4={cnt['4']} 5={cnt['5']} other={cnt['other']}\n")
    log_f.write("===========================================\n")
    log_f.close()

    print(f"\n[DONE] wrote: {OUT_PATH}")
    print(f"[DONE] log  : {os.path.abspath(LOG_PATH)}")


if __name__ == "__main__":
    main()
