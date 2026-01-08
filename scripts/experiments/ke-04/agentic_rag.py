"""
Agentic RAG 핵심 로직
- Thinking 모드 완전히 비활성화
- 메모리 최적화 전략 적용
- LangSmith 트래킹

이 파일은 하위 모듈들을 import하여 호환성을 유지합니다.
실제 구현은 다음 파일들에 있습니다:
- utils/: 유틸리티 함수들
- components/: 컴포넌트 클래스들
- agentic_rag_chain.py: AgenticRAGChain 클래스
- build_system.py: build_agentic_qa_system 함수
"""
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 하위 모듈에서 모든 것을 import
from utils import cleanup_memory, get_memory_usage, setup_langsmith
from components import (
    MultipleChoiceLLM, FAISSRetriever, ParagraphAnalyzer, QueryGenerator,
    SearchResultFilter, IntegratedReasoner, ChoiceValidator, ChoiceEvaluator,
    FinalReasoner
)
from agentic_rag_chain import AgenticRAGChain
from build_system import build_agentic_qa_system

# 호환성을 위해 export
__all__ = [
    'cleanup_memory',
    'get_memory_usage',
    'setup_langsmith',
    'MultipleChoiceLLM',
    'FAISSRetriever',
    'ParagraphAnalyzer',
    'QueryGenerator',
    'SearchResultFilter',
    'IntegratedReasoner',
    'ChoiceValidator',
    'ChoiceEvaluator',
    'FinalReasoner',
    'AgenticRAGChain',
    'build_agentic_qa_system'
]
