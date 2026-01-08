"""지문 분석 컴포넌트"""
from typing import List, Dict, Any, Optional
from .llm_wrapper import MultipleChoiceLLM


class ParagraphAnalyzer:
    """지문만으로 답을 구할 수 있는지 분석"""
    
    def __init__(self, llm: MultipleChoiceLLM):
        self.llm = llm
    
    def analyze(
        self,
        paragraph: str,
        question: str,
        choices: List[str],
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """지문 분석"""
        prompt = f"""다음 지문과 질문을 읽고, 지문만으로 답을 구할 수 있는지 판단하세요.

지문:
{paragraph}

질문:
{question}

선택지:
{chr(10).join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])}

지문만으로 답을 구할 수 있으면 "예"를, 배경지식이 필요하면 "아니오"를 답하세요.
답변 형식: "예" 또는 "아니오"로만 답하세요."""
        
        response = self.llm.generate(prompt, max_new_tokens=100)
        
        can_answer = "예" in response or "yes" in response.lower() or "가능" in response
        
        # 디버깅: 실제 응답 확인
        print(f"🔍 ParagraphAnalyzer 응답: {response[:200]}")
        print(f"   → can_answer_without_search: {can_answer}")
        
        return {
            'can_answer_without_search': can_answer,
            'reasoning': response
        }

