# mcq_infer_only_llamacpp_json.py
from __future__ import annotations

import ast
import gc
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp  # LangChain wrapper [web:25]
from tqdm import tqdm


# ----------------------------
# 1) Prompt (JSON only)
# ----------------------------
PROMPT_TEMPLATE = """당신은 대학 수학 능력 평가를 치르고 있는 최상위권 수험생입니다. 지문을 근거로 객관식 정답을 고르세요.

[내부 풀이 규칙(출력 금지)]
- 지문 우선주의: 사전 지식보다 지문 근거를 최우선으로 사용하세요.
- 질문의 부정형 표현(않는/없는/옳지 않은/근거 없는 등)을 먼저 확인하세요.
- 각 선택지를 지문의 근거와 매핑하고, 오답 소거로 정답을 결정하세요.
- <보기>/도표/조건이 있으면 지문 원리를 적용하세요.
- 위 과정에서의 생각/근거/해설은 절대 출력하지 말고 내부적으로만 수행하세요.

[출력 규칙(최우선)]
- 최종 출력은 반드시 JSON 오브젝트 하나만 출력하세요. 다른 텍스트를 절대 출력하지 마세요.
- 마크다운/코드블록(```)/설명/해설/근거/추가 문장/공백 설명을 모두 금지합니다.
- 형식은 정확히 다음과 같아야 합니다: {{"정답":"번호"}}
- "번호"는 1~{num_choices} 중 하나의 정수만 문자열로 넣으세요. (예: {{"정답":"2"}})
- 첫 글자는 반드시 {{ 이고, 마지막 글자는 반드시 }} 여야 합니다.
{qp_text}
지문:
{paragraph}

질문:
{question}

선택지:
{choices_text}

정답(JSON만 출력):"""



def build_prompt(paragraph: str, question: str, choices: List[str], question_plus: Optional[str]) -> str:
    choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    qp_text = f"<보충 제시문>\n{question_plus.strip()}\n" if question_plus else ""

    return PROMPT_TEMPLATE.format(
        qp_text=qp_text,
        paragraph=paragraph,
        question=question,
        choices_text=choices_text,
        num_choices=len(choices),
    )


# ----------------------------
# 2) JSON parsing utilities
# ----------------------------
def extract_first_json_object(text: str) -> Optional[dict]:
    """
    모델이 규칙을 어기고 설명/코드블록 등을 섞는 경우를 대비해,
    텍스트에서 첫 번째 JSON 오브젝트 후보를 찾아 json.loads 시도.
    """
    if not text:
        return None

    # 1) 가장 단순: 전체가 JSON일 때
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass

    # 2) 텍스트 중간의 {...} 추출 (탐욕 최소)
    #    주의: 중괄호 중첩이 복잡하면 완벽하진 않지만, 여기서는 {"정답":"n"}만 기대. [web:30]
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not m:
        return None

    cand = m.group(0).strip()
    try:
        return json.loads(cand)
    except Exception:
        return None


def parse_answer_number(text: str) -> Optional[str]:
    obj = extract_first_json_object(text)
    if not isinstance(obj, dict):
        return None
    ans = obj.get("정답", None)
    if ans is None:
        return None
    # "3", 3 모두 허용 -> 문자열로 정규화
    ans_str = str(ans).strip()
    # 숫자만 허용
    if not re.fullmatch(r"\d+", ans_str):
        return None
    return ans_str


def is_valid_choice(ans_str: str, num_choices: int) -> bool:
    try:
        n = int(ans_str)
    except Exception:
        return False
    return 1 <= n <= num_choices


# ----------------------------
# 3) One problem solve with retry
# ----------------------------
def solve_one_with_retry(
    llm: LlamaCpp,
    paragraph: str,
    question: str,
    choices: List[str],
    question_plus: Optional[str],
    max_retries: int = 5,
    verbose: bool = False,
) -> Tuple[str, str, dict]:
    prompt = build_prompt(paragraph, question, choices, question_plus)
    num_choices = len(choices)

    if verbose:
        print("=== Prompt ===")
        print(prompt)
        print("==============")

    last_raw = ""
    for attempt in range(max_retries + 1):
        raw = llm.invoke(prompt).strip()
        last_raw = raw

        if verbose:
            print("=== Raw ===")
            print(raw)
            print("==============")

        ans = parse_answer_number(raw)
        if ans is not None and is_valid_choice(ans, num_choices):
            meta = {
                "ok": True,
                "fallback": False,
                "attempts": attempt + 1,
                "retries": attempt,
                "attempt_used": attempt + 1,
            }
            return ans, raw, meta

        prompt = (
            prompt
            + "\n\n[오류] 출력이 JSON 형식이 아니거나 정답 번호가 범위를 벗어났습니다."
              f"\n반드시 {{\"정답\":\"1~{num_choices} 중 하나\"}} JSON 오브젝트 하나만 다시 출력하세요."
              "\n다른 텍스트 금지."
        )

    meta = {
        "ok": False,
        "fallback": True,
        "attempts": max_retries + 1,
        "retries": max_retries,
        "attempt_used": None,
    }
    return "1", last_raw, meta


# ----------------------------
# 4) Main
# ----------------------------
if __name__ == "__main__":
    # ---- model config ----
    repo_id = "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF"
    gguf_filename = "Qwen3-30B-A3B-Thinking-2507-UD-Q5_K_XL.gguf"
    exp_name = "qwen3-30b-a3b-tk2507-q5kxl_llamacpp-v1"

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=gguf_filename,
        cache_dir="/data/ephemeral/home/.cache/huggingface/hub",
        local_files_only=False,
    )

    llm = LlamaCpp(
        model_path=local_path,
        n_gpu_layers=40,
        n_ctx=8192,
        n_batch=512,
        temperature=0.0,  # JSON 안정성에 보통 유리
        verbose=False,
    )

    # ---- data load ----
    csv_path = "../../../../data/test/test.csv"
    df = pd.read_csv(csv_path, converters={"problems": ast.literal_eval})
    print(f"Loaded {len(df)} problems from: {csv_path}")

    # ---- inference ----
    submission_results = []
    detail_results = []

    start_total = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        problem_id = row["id"]
        paragraph = row["paragraph"]
        problems = row["problems"]
        question = problems.get("question", "")
        choices = problems.get("choices", [])
        question_plus = problems.get("question_plus", None)

        try:
            ans, raw, meta = solve_one_with_retry(
                llm=llm,
                paragraph=paragraph,
                question=question,
                choices=choices,
                question_plus=question_plus,
                max_retries=5,
                verbose=(idx == 0),
            )

            submission_results.append({"id": problem_id, "answer": ans})
            detail_results.append({
                "id": problem_id,
                "prediction": ans,
                "num_choices": len(choices),
                "raw": raw,          # 디테일 json에는 그대로 저장
                "meta": meta,        # 여기에 retry/성공여부 저장
            })

            # ---- logging (raw 제외) ----
            status = "OK" if meta["ok"] else "FALLBACK"
            print(
                f"\n[success] id={problem_id} pred={ans} status={status} retries={meta['retries']}",
                flush=True,
            )

            if idx % 10 == 0:
                gc.collect()

        except Exception as e:
            tqdm.write(f"[error] id={problem_id} error={repr(e)}")
            submission_results.append({"id": problem_id, "answer": "0"})
            detail_results.append({"id": problem_id, "error": str(e)})

    total_time = time.time() - start_total
    print(f"Done. total={total_time/3600:.2f}h avg={total_time/len(df):.2f}s/problem")

    # ---- save ----
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    sub_path = output_dir / f"{exp_name}.csv"
    pd.DataFrame(submission_results).to_csv(sub_path, index=False)
    print(f"Saved submission: {sub_path}")

    detail_path = output_dir / f"{exp_name}_detailed.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": exp_name,
                "total_samples": len(df),
                "predictions": detail_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved details: {detail_path}")
