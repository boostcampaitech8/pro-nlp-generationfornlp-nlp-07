"""통합 추론 컴포넌트"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from .llm_wrapper import MultipleChoiceLLM


class IntegratedReasoner:
    """지문 + 위키 검색 결과 통합 추론"""
    
    def __init__(self, llm: MultipleChoiceLLM):
        self.llm = llm
    
    def reason(
        self,
        paragraph: str,
        question: str,
        choices: List[str],
        search_results: List[Document],
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """통합 추론"""
        # 검색 결과 포맷팅
        rag_context = "\n\n".join([
            f"[{i+1}] {doc.metadata.get('title', 'N/A')}\n{doc.page_content}"
            for i, doc in enumerate(search_results)
        ])
        
        choices_text = "\n".join([
            f"{i+1}. {choice}" for i, choice in enumerate(choices)
        ])
        
        prompt = f"""다음 참고 자료는 문제 해결을 돕기 위한 배경 지식입니다. 이 자료는 위키피디아에서 검색된 내용으로, 문제의 지문과 직접 관련된 역사적 사실, 사회적 개념, 경제 원리, 정치 제도, 지리적 정보, 심리학 이론 등을 보충합니다.

참고 자료를 활용할 때는 다음을 유념하세요:
1. 원래 지문이 최우선 판단 기준입니다. 참고 자료는 지문의 맥락을 이해하거나 불분명한 개념을 보충하는 용도로만 사용하세요.
2. 참고 자료와 지문 내용이 충돌하면 반드시 지문을 따르세요.
3. 해당 문제는 지문 기반 추론이 핵심입니다. 선택지 판단 시 지문의 논리 흐름과 근거를 우선 분석하고, 참고 자료는 보조적으로만 활용하세요.
4. 각 선택지가 지문 및 참고 자료의 근거에 부합하는지 하나씩 검토한 뒤, 가장 타당한 하나만 고르세요.

참고 자료:
{rag_context}

---

지문을 읽고 질문의 답을 구하세요.

지문:
{paragraph}

질문:
{question}

선택지:
{choices_text}

1~{len(choices)} 중에 하나를 정답으로 고르세요.
정답:"""
        
        result = self.llm.predict_choice(prompt, num_choices=len(choices))
        
        return {
            'answer': result['answer'],
            'confidence': result['confidence'],
            'probs': result['probs'],
            'reasoning': '통합 추론 완료'
        }

