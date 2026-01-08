import os
import ast
import re
import pandas as pd
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm

# ================= [최종 안전 설정] =================
# 파일 경로 (사용자 환경에 맞게 수정)
RAW_PATH = "scripts/experiments/psj_11/submission/submission_cot_v1_raw.csv"
SUB_PATH = "scripts/experiments/psj_11/submission/submission_cot_v1.csv"
TEST_CSV = "data/test/test.csv"
CKPT_DIR = "scripts/experiments/psj_11/qwen32_cot_finetuned"

# [안전 제일 설정]
MAX_NEW_TOKENS = 800     # 800토큰이면 충분합니다.
MAX_SEQ_LEN = 4096
BATCH_SIZE = 1           # 🚨 [핵심] 4 -> 1 (속도와 안정성 모두 잡는 선택)
REPETITION_PENALTY = 1.2 # 무한 반복 방지
SAVE_EVERY_BATCHES = 10  # 10문제 풀 때마다 저장
# ==================================================

def parse_answer_from_cot(text: str):
    """
    CoT 텍스트에서 정답 번호를 추출합니다.
    """
    if not isinstance(text, str): return None
    
    # 1순위: "정답 : X" 형식 (공백 유연하게)
    match = re.search(r"정답\s*[:：]\s*([1-5①②③④⑤])", text)
    if match:
        val = match.group(1)
        mapper = {"①":1, "②":2, "③":3, "④":4, "⑤":5}
        return mapper.get(val, int(val))
    
    # 2순위: 텍스트 맨 끝부분에서 "~번" 찾기 (마지막 150자 이내)
    match = re.search(r"([1-5①②③④⑤])\s*(번|입니다|이다)", text[-150:])
    if match:
        val = match.group(1)
        mapper = {"①":1, "②":2, "③":3, "④":4, "⑤":5}
        return mapper.get(val, int(val))
        
    return None

def is_failed_parsing(row):
    """
    재추론이 필요한 행인지 판별합니다.
    """
    text = str(row.get('reasoning', row.get('generated_text', '')))
    
    # "정답 :" 패턴이 없으면 생성하다 끊긴 것으로 간주
    if not re.search(r"정답\s*[:：]", text):
        return True
    return False

def build_prompt(passage, question, choices):
    PROMPT_TEMPLATE = """당신은 수능 및 공무원 시험 한국사/국어/사회 과목의 1타 강사입니다.
주어진 지문과 문제를 읽고, 정답을 도출하는 과정을 [해설]에 작성하세요.

[주의사항]
1. 각 선택지의 근거를 한 문장으로 핵심만 요약하세요.
2. 마지막에 반드시 '정답 : X' 형식으로 결론을 내리세요.

[지문]
{passage}

[문제]
{question}

[선택지]
{choices}

[해설]"""
    return PROMPT_TEMPLATE.format(passage=passage, question=question, choices=choices)

def load_test_data(target_ids):
    df_raw = pd.read_csv(TEST_CSV)
    
    rows = []
    for _, row in df_raw.iterrows():
        if row["id"] not in target_ids: continue 
        
        try:
            if isinstance(row["problems"], str): 
                qa_dict = ast.literal_eval(row["problems"])
            else: 
                qa_dict = row["problems"]
                
            question = qa_dict.get("question", "")
            choices = qa_dict.get("choices", [])
        except:
            question, choices = "", []

        choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        
        rows.append({
            "id": row["id"],
            "passage": row["paragraph"] if pd.notna(row["paragraph"]) else "",
            "question": str(question),
            "choices": choices_str,
        })
    return pd.DataFrame(rows)

def main():
    print("🔍 재채점 대상 식별 중...")
    if not os.path.exists(RAW_PATH):
        print(f"❌ 파일을 찾을 수 없습니다: {RAW_PATH}")
        return

    try:
        df_results = pd.read_csv(RAW_PATH, encoding='utf-8')
    except:
        df_results = pd.read_csv(RAW_PATH, encoding='cp949')
    
    # 1. 재시도 대상 추출
    retry_mask = df_results.apply(is_failed_parsing, axis=1)
    retry_ids = set(df_results[retry_mask]['id'].tolist())
    
    print(f"📊 전체 {len(df_results)}개 중 {len(retry_ids)}개가 재시도 대상입니다.")
    
    if len(retry_ids) == 0:
        print("✅ 모든 데이터가 정상입니다.")
        return

    # 2. 데이터 로드
    retry_df = load_test_data(retry_ids)
    
    # 3. 모델 로드
    print(f"🚀 모델 로드 중 (Batch Size: {BATCH_SIZE})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = CKPT_DIR,
        max_seq_length = MAX_SEQ_LEN,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    
    # 4. 추론 시작
    print("🔥 재추론 시작 (안전 모드)...")
    
    update_dict_sub = {}
    update_dict_reasoning = {}
    
    all_prompts = []
    all_ids = []
    
    for _, row in retry_df.iterrows():
        all_prompts.append(build_prompt(row['passage'], row['question'], row['choices']))
        all_ids.append(row['id'])
        
    # 배치 단위로 추론 진행
    for i in tqdm(range(0, len(all_ids), BATCH_SIZE)):
        batch_prompts = all_prompts[i : i + BATCH_SIZE]
        batch_ids = all_ids[i : i + BATCH_SIZE]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                use_cache=True, 
                repetition_penalty=REPETITION_PENALTY,
                do_sample=False
            )
            
        generated_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        for idx, text in zip(batch_ids, generated_texts):
            pred = parse_answer_from_cot(text)
            
            # 실패 시 로그 찍고 1번으로 임시 저장
            if pred is None:
                # print(f"⚠️ {idx}: 파싱 실패 (길이: {len(text)})")
                pred = 1 
            
            update_dict_sub[idx] = pred
            update_dict_reasoning[idx] = text

        # ==========================================================
        # [중간 저장]
        # ==========================================================
        current_batch_count = (i // BATCH_SIZE) + 1
        if current_batch_count % SAVE_EVERY_BATCHES == 0:
            
            temp_raw = df_results.copy()
            temp_sub = pd.read_csv(SUB_PATH)
            
            for uid, reason in update_dict_reasoning.items():
                mask_raw = temp_raw['id'] == uid
                if mask_raw.any():
                    temp_raw.loc[mask_raw, 'reasoning'] = reason
                    temp_raw.loc[mask_raw, 'answer'] = update_dict_sub[uid]
                
                mask_sub = temp_sub['id'] == uid
                if mask_sub.any():
                    temp_sub.loc[mask_sub, 'answer'] = update_dict_sub[uid]
            
            temp_raw.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")
            temp_sub.to_csv(SUB_PATH, index=False)
            tqdm.write(f"💾 [자동 저장] {len(update_dict_reasoning)}건 완료")
        # ==========================================================

    # 5. 최종 저장
    print("\n🏁 최종 결과 저장 중...")
    
    for idx, new_reasoning in update_dict_reasoning.items():
        mask = df_results['id'] == idx
        df_results.loc[mask, 'reasoning'] = new_reasoning
        df_results.loc[mask, 'answer'] = update_dict_sub[idx]
        
    df_results.to_csv(RAW_PATH, index=False, encoding="utf-8-sig")
    
    df_sub = pd.read_csv(SUB_PATH)
    for idx, new_ans in update_dict_sub.items():
        mask = df_sub['id'] == idx
        df_sub.loc[mask, 'answer'] = new_ans
        
    df_sub.to_csv(SUB_PATH, index=False)
    
    print(f"🎉 고생하셨습니다! 총 {len(retry_ids)}건 수정 완료!")

if __name__ == "__main__":
    main()