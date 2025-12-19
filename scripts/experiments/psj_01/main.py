# scripts/experiments/psj_01/main.py
import pandas as pd
import os
import sys

# 같은 폴더에 있는 파일을 불러오기 위해 현재 폴더 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from agent_orchestrator import SuneungAgent

def main():
    # 1. 설정 (API 키 입력)
    API_KEY = "AIzaSyA7r4K6oTiq-LKdkVQU358R3TCjeyYzJDY" 
    
    # 2. 데이터 로드 (절대 경로 고정)
    data_path = "/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/train/train.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return

    # 에이전트 초기화
    agent = SuneungAgent(API_KEY)
    df = pd.read_csv(data_path)
    
    results = []
    print("🚀 Gemini 3.0-Flash 분석 시작")
    print("-" * 50)
    
    for idx, row in df.iterrows():
        cat, rag, info = agent.solve(row)
        results.append({
            "id": row['id'], 
            "category": cat, 
            "use_rag": rag, 
            "reason_or_query": info
        })

    # 결과 저장 (현재 폴더에 저장됨)
    res_df = pd.DataFrame(results)
    res_df.to_csv("classification_results_psj.csv", index=False, encoding='utf-8-sig')
    print("-" * 50)
    print(f"✅ 완료! 'scripts/experiments/psj_01/classification_results_psj.csv' 확인")

if __name__ == "__main__":
    main()