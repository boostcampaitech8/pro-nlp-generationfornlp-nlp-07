"""검색 쿼리 생성 컴포넌트"""
from typing import List, Optional
from .llm_wrapper import MultipleChoiceLLM


class QueryGenerator:
    """검색 쿼리 생성"""
    
    def __init__(self, llm: MultipleChoiceLLM):
        self.llm = llm
    
    def generate_query(
        self,
        paragraph: str,
        question: str,
        choices: List[str],
        previous_queries: Optional[List[str]] = None,
        callbacks: Optional[List] = None
    ) -> str:
        """검색 쿼리 생성"""
        previous_info = ""
        if previous_queries:
            previous_info = f"\n\n이전 검색 쿼리:\n{chr(10).join([f'- {q}' for q in previous_queries])}\n\n위 쿼리로 충분한 정보를 얻지 못했습니다. 다른 관점에서 쿼리를 생성하세요."
        
        prompt = f"""다음 지문과 질문을 읽고, 위키피디아에서 검색할 쿼리를 생성하세요.

지문:
{paragraph}

질문:
{question}

선택지:
{chr(10).join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])}
{previous_info}

위키피디아에서 검색할 키워드나 질문을 한 문장으로 작성하세요.
검색 쿼리만 답하세요 (설명 없이)."""
        
        query = self.llm.generate(prompt, max_new_tokens=50)
        return query.strip()
    
    def generate_query_for_choice(
        self,
        paragraph: str,
        question: str,
        choice: str,
        choice_num: int,
        previous_queries: Optional[List[str]] = None,
        callbacks: Optional[List] = None
    ) -> str:
        """특정 선택지에 대한 검색 쿼리 생성"""
        previous_info = ""
        if previous_queries:
            previous_info = f"\n\n이전 검색 쿼리:\n{chr(10).join([f'- {q}' for q in previous_queries])}\n\n위 쿼리로 충분한 정보를 얻지 못했습니다. 다른 관점에서 쿼리를 생성하세요."
        
        prompt = f"""다음 지문과 질문, 그리고 특정 선택지를 읽고, 이 선택지가 참인지 거짓인지 판단하기 위해 필요한 위키피디아 검색 쿼리를 생성하세요.

지문:
{paragraph}

질문:
{question}

선택지 {choice_num}: {choice}
{previous_info}

위키피디아에서 검색할 키워드나 질문을 한 문장으로 작성하세요.
이 선택지의 핵심 개념이나 주장을 검증하기 위한 검색 쿼리를 생성하세요.
검색 쿼리만 답하세요 (설명 없이)."""
        
        query = self.llm.generate(prompt, max_new_tokens=50)
        return query.strip()

