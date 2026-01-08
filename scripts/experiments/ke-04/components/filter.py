"""검색 결과 필터링 컴포넌트"""
from typing import List, Optional
from langchain_core.documents import Document


class SearchResultFilter:
    """검색 결과 필터링 및 리랭킹"""
    
    def filter_and_rerank(
        self,
        docs: List[Document],
        query: str,
        top_k: int = 5,
        callbacks: Optional[List] = None
    ) -> List[Document]:
        """검색 결과 필터링 및 리랭킹"""
        # 간단한 필터링: 관련성 점수 기반 (거리 기반)
        # 실제로는 Cross-encoder 리랭커를 사용할 수 있지만 메모리 고려하여 생략
        
        # 문서 길이 기반 필터링 (너무 짧거나 긴 문서 제외)
        filtered_docs = []
        for doc in docs:
            content_len = len(doc.page_content)
            if 100 <= content_len <= 5000:  # 적절한 길이
                filtered_docs.append(doc)
        
        # 상위 k개만 반환
        return filtered_docs[:top_k]

