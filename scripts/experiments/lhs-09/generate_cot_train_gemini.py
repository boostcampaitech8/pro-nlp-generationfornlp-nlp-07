import os
import re
import time
import argparse
import gc
from ast import literal_eval

import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types

# -----------------------------
# 1. Prompt 설정 (수능 전문가 페르소나 강화)
# -----------------------------
SYSTEM_PROMPT = """당신은 대한민국 수능형 객관식 문제(국어/사회/역사/과학 등)의 풀이를 돕는 전문가 교사입니다.
학생이 지문을 통해 논리적으로 사고하여 정답을 도출할 수 있도록 근거 중심의 풀이(CoT)를 생성하세요.

[출력 규칙]
- 반드시 한국어로 작성할 것.
- 핵심 요약: 지문의 전체적인 주제나 상황을 1문장으로 정리.
- 근거: 지문에서 정답과 직접적으로 연결되는 핵심 문구 1~3개 추출.
- 풀이: 단계별로 논리를 전개하여 정답이 도출되는 과정을 설명.
- 오답 포인트: 학생들이 정답으로 오해하기 쉬운 선택지 1~2개가 왜 오답인지 짧게 설명.

[주의 사항]
- "정답은 X번입니다"와 같이 번호를 직접적으로 언급하지 마세요.
- 선택지 번호를 직접 쓰기보다 선택지의 내용을 바탕으로 설명하세요.
"""

def safe_parse_problems(x):
    if isinstance(x, dict):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return {}
    try:
        # csv에 저장된 문자열 형태의 dict를 파이썬 객체로 변환
        return literal_eval(x)
    except Exception:
        return {}

def build_prompt(paragraph, question, question_plus, choices):
    qplus = ""
    if question_plus is not None and not (isinstance(question_plus, float) and pd.isna(question_plus)):
        qplus = str(question_plus).strip()
    
    extra_box = f"\n[추가 보기/조건]\n{qplus}\n" if qplus else ""
    choices_str = "\n".join([f"{i+1}) {c}" for i, c in enumerate(choices)])

    user_prompt = f"""
[지문]
{paragraph}

[문제]
{question}
{extra_box}
[선택지]
{choices_str}

위 문제에 대해 '핵심 요약, 근거, 풀이, 오답 포인트' 순서로 CoT를 작성하세요.
""".strip()
    return user_prompt

def preview_print(idx, _id, text, chars=450):
    # 줄바꿈을 공백으로 치환하여 한 줄로 간단히 보기 위함
    one = re.sub(r"\s+", " ", (text or "")).strip()
    print("\n" + "="*30 + " PREVIEW " + "="*30)
    print(f"Index: {idx} | ID: {_id}")
    print(f"Content: {one[:chars]}...")
    print("="*69 + "\n", flush=True)

def gemini_generate_with_retry(client, model_name, prompt, max_attempts=3, sleep_sec=1.5):
    last_err = ""
    temps = [0.2, 0.4, 0.6]
    
    for attempt in range(max_attempts):
        try:
            temp = temps[min(attempt, len(temps)-1)]
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=temp,
                    # [수정 완료] 최신 SDK 규격에 맞는 카테고리 명칭 사용
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH", 
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT", 
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT", 
                            threshold="BLOCK_NONE"
                        ),
                    ]
                )
            )
            
            if response.text:
                txt = response.text.strip()
                if len(txt) > 50:
                    return True, txt, ""
                else:
                    last_err = f"응답이 너무 짧음 (len={len(txt)})"
            else:
                # 차단 사유 확인 (Safety Filter 등)
                finish_reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
                last_err = f"빈 응답 (사유: {finish_reason})"
                
        except Exception as e:
            last_err = str(e)
            # 모델명을 찾을 수 없는 경우에 대한 디버깅 로그 추가
            if "404" in last_err or "not found" in last_err.lower():
                last_err = f"모델명 오류: '{model_name}' 모델을 찾을 수 없습니다. (gemini-1.5-flash 등을 시도하세요)"
        
        time.sleep(sleep_sec)
    
    return False, "", last_err
def main():
    parser = argparse.ArgumentParser(description="Gemini API를 활용한 수능 데이터 CoT 증강")
    parser.add_argument("--input_csv", type=str, default="data/train.csv")
    parser.add_argument("--output_csv", type=str, default="data/train_with_cot.csv")
    parser.add_argument("--api_key", type=str, required=True, help="Google API Key")
    parser.add_argument("--model_name", type=str, default="gemini-1.5-flash")
    parser.add_argument("--sleep_sec", type=float, default=1.2)
    parser.add_argument("--save_every", type=int, default=10) # 10개마다 파일 저장
    parser.add_argument("--preview_every", type=int, default=1) # 매 샘플 프리뷰
    parser.add_argument("--preview_chars", type=int, default=450)    # 추가됨
    parser.add_argument("--max_attempts", type=int, default=3)      # 추가됨
    args = parser.parse_args()

    # 신규 SDK 클라이언트 초기화
    client = genai.Client(api_key=args.api_key)

    if not os.path.exists(args.input_csv):
        print(f"Error: {args.input_csv} 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(args.input_csv)

    # 출력 컬럼 초기화 (이어하기 지원)
    for col in ["cot_text", "cot_status", "cot_error"]:
        if col not in df.columns:
            df[col] = ""

    ok_cnt, fail_cnt = 0, 0
    pbar = tqdm(range(len(df)), desc=f"Generating CoT ({args.model_name})")

    for i in pbar:
        row = df.iloc[i]
        _id = row.get("id", i)

        # 이미 데이터가 있는 경우 건너뛰기 (Resume 기능)
        if str(df.at[i, "cot_text"]).strip():
            ok_cnt += 1
            continue

        paragraph = str(row.get("paragraph", "") or "")
        probs = safe_parse_problems(row.get("problems", ""))
        question_plus = row.get("question_plus", "")
        question = probs.get("question", "")
        choices = probs.get("choices", [])

        if not paragraph or not question or len(choices) < 2:
            df.at[i, "cot_status"] = "bad_row"
            fail_cnt += 1
            continue

        prompt = build_prompt(paragraph, question, question_plus, choices)
        
        ok, txt, err = gemini_generate_with_retry(
            client, args.model_name, prompt,
            max_attempts=3, sleep_sec=args.sleep_sec
        )

        if ok:
            df.at[i, "cot_text"] = txt
            df.at[i, "cot_status"] = "ok"
            ok_cnt += 1
        else:
            df.at[i, "cot_status"] = "fail"
            df.at[i, "cot_error"] = err[:500]
            fail_cnt += 1

        # 실시간 프리뷰 출력
        if args.preview_every > 0 and (ok_cnt + fail_cnt) % args.preview_every == 0:
            preview_print(i, _id, txt if ok else f"[FAIL] {err}")

        # 중간 저장
        if (ok_cnt + fail_cnt) % args.save_every == 0:
            df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

        pbar.set_postfix({"OK": ok_cnt, "Fail": fail_cnt})
        time.sleep(args.sleep_sec)

    # 최종 저장
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 작업 완료: {args.output_csv}에 저장됨 (성공: {ok_cnt}, 실패: {fail_cnt})")

if __name__ == "__main__":
    main()