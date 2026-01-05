import os
import ast
import time
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

GEMINI_API_KEY = 
INPUT_CSV = "data/train/train_augmented_concat.csv"
OUTPUT_CSV = "data/train/train_cot_test_all.csv"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-3-flash-preview')

def generate_reasoning(passage, question, choices_str, answer):
    """
    Gemini에게 Few-Shot 프롬프트(한국사+사회+세계사)로 해설 생성 요청
    """
    prompt = f"""
    당신은 수능 및 공무원 시험 1타 강사입니다. 
    아래 [예시 1], [예시 2], [예시 3]의 **논리적인 흐름, 말투, 구조**를 완벽하게 모방하여, 새로운 문제에 대한 명쾌한 해설을 작성해 주세요.

    === [예시 1: 한국사 (사건 순서 나열형)] ===
    [지문]
    (가) 김시민이 진주성에서 일본군을 저지하였다.
    (나) 조선수군이 명량해전에서 크게 승리하였다.
    (다) 이순신이 옥포해전에서 승리하였다.
    (라) 조명연합 수군이 노량해전에서 승리하였다.
    (마) 조명연합군이 평양성을 탈환하였다.
    [문제]
    임진왜란의 주요 사건을 시기 순으로 바르게 나열한 것은?
    [선택지]
    1. (가)→(다)→(마)→(라)→(나)
    2. (가)→(마)→(다)→(나)→(라)
    3. (다)→(가)→(나)→(마)→(라)
    4. (다)→(가)→(마)→(나)→(라)
    [정답 번호]
    4
    [해설]
    이 문제는 임진왜란 주요 전투들의 발생 시기를 정확히 알고 있는지 묻는 문제입니다.
    각 사건의 연도를 정리하면 다음과 같습니다.
    (다) 옥포해전: 1592년 5월, 이순신 장군의 첫 승리입니다.
    (가) 진주 대첩: 1592년 10월, 김시민 장군이 진주성에서 승리하였습니다.
    (마) 평양성 탈환: 1593년 1월, 조명 연합군이 평양성을 되찾았습니다.
    (나) 명량해전: 1597년 9월, 정유재란 시기 이순신 장군의 승리입니다.
    (라) 노량해전: 1598년 11월, 이순신 장군이 전사한 임진왜란 최후의 전투입니다.
    따라서 시간 순서는 (다)→(가)→(마)→(나)→(라)가 됩니다.
    그러므로 정답은 4번입니다.

    === [예시 2: 사회/비문학 (정보 추출형)] ===
    [지문]
    더불어민주당 국회 소병훈 의원(경기 광주시갑·국토교통위원회)이 제공한 경찰청 데이트폭력 현황자료에 따르면, 2016년부터 올해 상반기까지 총 43,046명이 데이트폭력으로 검거된 것으로 나타났다. 이는 연간 9,566명, 하루 평균 26명이 검거됐음을 의미한다. (중략) 한편 2019년 대비 2020년 상반기 데이트폭력 검거인원은 9,858명에서 4,273명으로 13.3% 감소한 것으로 나타났다.
    [문제]
    2016년부터 올해 상반기까지 데이트폭력으로 검거된 총 인원은 얼마인가?
    [선택지]
    1. 43,046명
    2. 9,566명
    3. 26명
    4. 10,798명
    5. 4,273명
    [정답 번호]
    1
    [해설]
    이 문제는 제시된 자료에서 구체적인 수치 정보를 정확하게 찾아내는 문제입니다.
    지문의 첫 번째 문장을 보면 "2016년부터 올해 상반기까지 총 43,046명이 데이트폭력으로 검거된 것으로 나타났다"라고 명시되어 있습니다.
    다른 선택지들을 살펴보면, 2번(9,566명)은 연간 평균 인원이고, 3번(26명)은 하루 평균 인원입니다. 5번(4,273명)은 2020년 상반기 검거 인원입니다.
    따라서 문제에서 묻는 총 인원은 1번입니다.

    === [예시 3: 세계사 (자료 해석형)] ===
    [지문]
    오, 수치스럽도다, 불쌍한 겨울의 왕이여! 그대는 도대체 무슨 짓을 벌인 것인가? 카이저의 왕좌를 찬탈하는 것은 무척이나 나쁜 일이 아니던가? 이제 그대는 라인강과 프라하 모두로부터 멀어져야 할지니, 무엇보다도 수치와 경멸에 의해 밤낮으로 괴로움에 떨 것이다. (중략) 그러니 프리츠여, 일어서서 그대의 왕 페르디난트에게 가라, 그대의 왕에게 부디 그 죄를 사하게 해달라 은혜롭게 간청하라. ”불쌍한 겨울의 왕,” 17세기의 노래
    [문제]
    다음 중 화자가 "라인강과 프라하"를 언급하는 이유를 가장 잘 설명하는 것은 무엇인가?
    [선택지]
    1. 겨울왕의 계획된 순례 여행의 목적지이다.
    2. 겨울왕이 전투에서 패배한 장소다.
    3. 신성 로마 제국의 요새였기 때문에 화자는 겨울 왕에게 멀리 떨어지라고 경고하는 것이다.
    4. 신성 로마 제국 국경 내의 중요한 군사 기지이다.
    [정답 번호]
    2
    [해설]
    이 문제는 17세기 30년 전쟁 당시 '겨울왕'이라 불린 프리드리히 5세에 대한 풍자시를 해석하는 문제입니다.
    지문에서 '겨울의 왕'은 신성 로마 제국 황제(카이저)에 대항해 보헤미아의 왕이 되었으나 곧 패배한 인물입니다.
    여기서 '라인강'은 그의 본거지인 팔츠 지방을, '프라하'는 그가 잠시 차지했던 보헤미아의 수도를 의미합니다. 화자는 그가 패배하여 이 두 곳에서 모두 쫓겨나게 된 상황을 "멀어져야 할지니"라고 조롱하고 있습니다.
    따라서 '라인강과 프라하'는 그가 지배력을 상실하고 패배하여 잃어버린 장소들을 의미하므로, 가장 적절한 설명은 2번입니다.
    =========================================

    이제 위 예시들처럼 논리적이고 친절하게, 아래 문제에 대한 해설을 작성하세요.

    [지문]
    {passage}

    [문제]
    {question}

    [선택지]
    {choices_str}

    [정답 번호]
    {answer}

    [해설]
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return None

def find_json_column(row):
    """어느 컬럼에 QA 데이터가 있는지 자동으로 찾습니다."""
    for col in row.index:
        val = str(row[col])
        if val.strip().startswith("{") and "question" in val:
            return val
    return None

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"파일 없음: {INPUT_CSV}")
        return

    print(f"[로드 중] {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # 해설 컬럼 없으면 생성
    if "reasoning" not in df.columns:
        df["reasoning"] = ""

    print(f"총 {len(df)}개 데이터. 작업을 시작합니다...")
    
    success_count = 0
    save_interval = 10
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 이미 해설 있으면 스킵
        if pd.notna(row["reasoning"]) and str(row["reasoning"]).strip() != "":
            continue

        try:
            # 1. QA 데이터 찾기 (자동 감지)
            qa_str = find_json_column(row)
            
            if not qa_str: continue

            # 2. 파싱 시도
            qa_dict = ast.literal_eval(qa_str)
            
            question = qa_dict.get("question", "")
            choices = qa_dict.get("choices", [])
            answer = qa_dict.get("answer", None)
            
            # 지문 찾기 (passage, paragraph 등등)
            passage = str(row.get("passage", row.get("paragraph", "")))

            if not question or answer is None:
                continue

            choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])

            # 3. Gemini 호출 (Few-Shot 프롬프트 사용)
            reasoning = generate_reasoning(passage, question, choices_str, answer)
            
            if reasoning:
                df.at[idx, "reasoning"] = reasoning
                success_count += 1
            
            time.sleep(0.5) # 속도 조절

        except Exception as e:
            # 에러 나면 무시하고 진행
            continue

        # 중간 저장
        if (idx + 1) % save_interval == 0:
            df.to_csv(OUTPUT_CSV, index=False)

    # 최종 저장
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[완료] 총 {success_count}개의 고품질 해설을 생성했습니다! 저장 경로: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
