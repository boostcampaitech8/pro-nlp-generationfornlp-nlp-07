# psj_01/agent_orchestrator.py
import google.generativeai as genai
import ast
from prompts import ROUTER_PROMPT, JUDGE_PROMPT

class SuneungAgent:
    def __init__(self, api_key):
        # API 키 공백 제거 및 설정
        genai.configure(api_key=api_key.strip())
        # 최신 고성능 모델 사용
        self.model = genai.GenerativeModel('models/gemini-3-flash-preview') 
        self.categories = ["한국사", "역사(유럽)", "역사(미국)", "역사(세계)", "경제(거시)", "경제(미시)", "지리", "정치/법령", "사회/교육", "인문", "미분류"]

    def route_and_judge(self, paragraph, question):
        try:
            # 1. 카테고리 분류
            router_input = ROUTER_PROMPT.format(paragraph=paragraph[:1200], question=question)
            res = self.model.generate_content(router_input).text.strip()
            
            category = "미분류"
            for cat in self.categories:
                if cat in res:
                    category = cat
                    break
            
            # 2. RAG 필요성 및 근거/쿼리 판단
            judge_input = JUDGE_PROMPT.format(paragraph=paragraph[:1000], question=question)
            judge_res_raw = self.model.generate_content(judge_input).text.strip()
            
            use_rag = False
            info = ""
            
            # 판단 결과 파싱 로직
            if "RAG" in judge_res_raw.upper():
                use_rag = True
            
            if "이유/쿼리:" in judge_res_raw:
                info = judge_res_raw.split("이유/쿼리:")[1].strip()
            else:
                info = judge_res_raw # 형식이 깨질 경우 대비
                
            return category, use_rag, info

        except Exception as e:
            print(f"❌ API 에러 발생: {e}")
            return "미분류", False, "시스템 에러로 인한 기본값"

    def solve(self, row):
        try:
            p = row['paragraph']
            probs = row['problems']
            # 데이터가 문자열일 경우 딕셔너리로 변환
            if isinstance(probs, str):
                probs = ast.literal_eval(probs)
            q = probs['question']
            
            cat, rag, info = self.route_and_judge(p, q)
            
            # 진행 상황 터미널 출력
            status = "🔍 [RAG]" if rag else "📝 [INT]"
            print(f"ID: {row['id']} | {cat:10} | {status} {info[:50]}...")
            
            return cat, rag, info
        except Exception as e:
            print(f"❌ 데이터 파싱 에러 (ID: {row.get('id')}): {e}")
            return "미분류", False, "데이터 파싱 에러"