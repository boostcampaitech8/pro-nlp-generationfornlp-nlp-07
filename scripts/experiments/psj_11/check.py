import pandas as pd
import os

file_path = 'scripts/experiments/psj_11/submission/submission_cot_v1_raw.csv'

if not os.path.exists(file_path):
    print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
else:
    try:
        # 데이터 로드
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp949')

        # 분석 대상 컬럼 설정
        target_col = 'reasoning' if 'reasoning' in df.columns else 'generated_text'
        
        # 전처리
        df['answer'] = df['answer'].astype(str).str.strip()
        df[target_col] = df[target_col].astype(str)

        # -------------------------------------------------------
        # [수정 핵심] 정규표현식 사용 (Regex)
        # r"정답\s*[:]" 의미: '정답' 뒤에 공백(\s)이 0개 이상(*) 있고 콜론[:]이 오는 패턴
        # -------------------------------------------------------
        cond_not_1 = df['answer'] != '1'
        
        # "정답:" 패턴이 아예 없는 경우만 필터링
        cond_cut_off = ~df[target_col].str.contains(r"정답\s*[:]", na=False, regex=True)

        problem_cases = df[cond_not_1 & cond_cut_off]

        print("-" * 50)
        print(f"📂 분석 파일: {file_path}")
        print(f"🔍 분석 타겟: {target_col}")
        print(f"📊 전체 데이터: {len(df)}개")
        # 이제 481건보다 훨씬 줄어들 것입니다.
        print(f"🚨 수정된 조건으로 발견된 건수: {len(problem_cases)}건") 
        print("-" * 50)

        if len(problem_cases) > 0:
            print("\n[진짜로 잘린 케이스 예시]")
            for idx, row in problem_cases.head(3).iterrows():
                print(f"\nRow Index: {idx}")
                print(f"실제 정답: {row['answer']}")
                print(f"생성 텍스트(끝부분): ...{row[target_col][-80:].replace(chr(10), ' ')}")
        else:
            print("\n✅ 모든 데이터에 '정답:' 포맷이 존재합니다!")

    except Exception as e:
        print(f"❌ 오류: {e}")