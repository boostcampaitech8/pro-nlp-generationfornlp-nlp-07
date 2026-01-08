"""선택지 평가 컴포넌트"""
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from .llm_wrapper import MultipleChoiceLLM


class ChoiceEvaluator:
    """각 선택지별 개별 판단"""
    
    def __init__(self, llm: MultipleChoiceLLM):
        self.llm = llm
    
    def evaluate_choice(
        self,
        paragraph: str,
        question: str,
        choice: str,
        choice_num: int,
        choice_docs: List[Document],
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        선택지 개별 판단
        
        Returns:
            {
                'choice_num': "1",
                'judgment': "correct" | "incorrect" | "ambiguous",
                'confidence': 0.0 ~ 1.0,
                'reasoning': "판단 근거"
            }
        """
        # 검색 결과 포맷팅
        rag_context = "\n\n".join([
            f"[{i+1}] {doc.metadata.get('title', 'N/A')}\n{doc.page_content}"
            for i, doc in enumerate(choice_docs)
        ])
        
        prompt = f"""지문과 질문, 그리고 특정 선택지를 읽고, 이 선택지가 지문과 질문에 부합하는지 판단하세요.

지문:
{paragraph}

질문:
{question}

선택지 {choice_num}: {choice}

검색된 참고 자료:
{rag_context if rag_context else "참고 자료 없음"}

판단 기준:
1. 지문의 내용과 논리적 흐름을 우선 고려하세요
2. 참고 자료는 보조적으로만 활용하세요
3. 지문과 참고 자료가 충돌하면 반드시 지문을 따르세요

이 선택지가 지문과 질문에 부합하는지 판단하세요.
- 판단: "맞음", "틀림", "애매함" 중 하나로 답하세요
- 신뢰도: 0.0 ~ 1.0 사이의 숫자로 답하세요 (맞음이면 높은 값, 틀림이면 낮은 값, 애매함이면 중간 값)

답변 형식:
판단: [맞음/틀림/애매함]
신뢰도: [0.0~1.0]"""
        
        response = self.llm.generate(prompt, max_new_tokens=200)
        
        # 응답 파싱
        judgment = "ambiguous"
        confidence = 0.5
        
        if "맞음" in response or "correct" in response.lower() or "참" in response:
            judgment = "correct"
        elif "틀림" in response or "incorrect" in response.lower() or "거짓" in response:
            judgment = "incorrect"
        else:
            judgment = "ambiguous"
        
        # 신뢰도 추출
        confidence_match = re.search(r'신뢰도[:\s]*([0-9.]+)', response)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                # 0.0~1.0 범위로 정규화
                confidence = max(0.0, min(1.0, confidence))
            except:
                pass
        else:
            # 판단 결과에 따라 기본 신뢰도 설정
            if judgment == "correct":
                confidence = 0.8
            elif judgment == "incorrect":
                confidence = 0.2
            else:
                confidence = 0.5
        
        return {
            'choice_num': str(choice_num),
            'judgment': judgment,
            'confidence': confidence,
            'reasoning': response
        }


class ChoiceValidator:
    """모든 선택지가 충분히 검증되었는지 확인"""
    
    def validate_all_choices(
        self,
        reasoning_result: Dict[str, Any],
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """선택지 검증"""
        # 간단한 검증: 확률이 충분히 높으면 검증 완료
        confidence = reasoning_result.get('confidence', 0.0)
        threshold = 0.6  # 임계값
        
        all_validated = confidence >= threshold
        
        return {
            'all_validated': all_validated,
            'confidence': confidence,
            'threshold': threshold
        }


class FinalReasoner:
    """애매한 선택지들만 최종 판단"""
    
    def __init__(self, llm: MultipleChoiceLLM):
        self.llm = llm
    
    def reason_ambiguous_choices(
        self,
        paragraph: str,
        question: str,
        ambiguous_choices: Dict[str, Dict[str, Any]],
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        애매한 선택지들만 비교하여 최종 판단
        
        ambiguous_choices 구조:
        {
            "1": {
                "choice": "선택지 텍스트",
                "docs": [Document, ...],
                "evaluation": {...},  # 1단계 판단 결과
                ...
            },
            "3": {...},
            ...
        }
        """
        # 애매한 선택지별 정보 포맷팅
        ambiguous_sections = []
        choice_nums = []
        
        for choice_num, choice_info in ambiguous_choices.items():
            choice = choice_info.get('choice', '')
            docs = choice_info.get('docs', [])
            evaluation = choice_info.get('evaluation', {})
            confidence = evaluation.get('confidence', 0.5)
            
            choice_nums.append(choice_num)
            
            # 검색 결과 포맷팅
            rag_context = "\n\n".join([
                f"{i+1}. [{doc.metadata.get('title', 'N/A')}]\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ])
            
            section = f"""[선택지 {choice_num}] {choice}

1단계 판단: 애매함 (신뢰도: {confidence:.2f})

검색된 참고 자료:
{rag_context if rag_context else "참고 자료 없음"}"""
            
            ambiguous_sections.append(section)
        
        ambiguous_text = "\n\n---\n\n".join(ambiguous_sections)
        
        prompt = f"""지문과 질문을 읽고, 애매한 선택지들을 비교하여 가장 부합하는 하나를 선택하세요.

지문:
{paragraph}

질문:
{question}

애매한 선택지들 (비교 판단 필요):

{ambiguous_text}

위 애매한 선택지들을 비교하여 지문과 질문에 가장 부합하는 하나를 선택하세요.
- 지문의 논리적 흐름과 근거를 우선 고려하세요
- 각 선택지의 검색 결과를 보조적으로 활용하세요
- 지문과 참고 자료가 충돌하면 반드시 지문을 따르세요

{', '.join(choice_nums)} 중에 하나를 정답으로 고르세요.
정답:"""
        
        result = self.llm.predict_choice(prompt, num_choices=len(ambiguous_choices))
        
        return {
            'answer': result['answer'],
            'confidence': result['confidence'],
            'probs': result['probs'],
            'reasoning': f'애매한 선택지 {len(ambiguous_choices)}개 중 최종 판단'
        }

