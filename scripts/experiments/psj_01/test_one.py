import sys
import os

# psj_01 폴더 경로 추가 (모듈 임포트용)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from agent_orchestrator import SuneungAgent

def test_single_sample():
    # 1. 환경 설정
    API_KEY = "" # API 키를 입력하세요
    agent = SuneungAgent(API_KEY)

    # 2. 테스트 데이터 구성
    sample_data = {
        "id": "generation-for-nlp-717",
        "paragraph": "인기 TV 쇼의 한 재무 설계사가 더 많은 미국인들이 은퇴를 위해 저축할 것을 설득합니다.",
        "problems": "{'question': '대출 가능한 자금의 수요와 공급에 대한 결과는 어떻게 됩니까?', 'choices': ['공급 곡선이 평형 이자율을 증가시키면서 위로 이동한다.', '수요 곡선이 평형 이자율을 증가시키면서 위로 이동한다.', '공급 곡선이 평형 이자율을 감소시키면서 아래로 이동한다.', '수요 곡선이 평형 이자율을 감소시키면서 아래로 이동한다.'], 'answer': 3}"
    }

    print(f"\n🔍 [테스트 시작] ID: {sample_data['id']}")
    print(f"지문: {sample_data['paragraph']}")
    
    # 3. 에이전트 실행
    # solve 함수는 row 형태를 받으므로 dict를 그대로 전달
    cat, rag, info = agent.solve(sample_data)

    print("\n" + "="*50)
    print(f"📌 최종 분류 결과: {cat}")
    print(f"📌 RAG 필요 여부: {'필요(RAG)' if rag else '불필요(INTERNAL)'}")
    print(f"📌 상세 근거/쿼리: {info}")
    print("="*50)

if __name__ == "__main__":
    test_single_sample()
