"""FAISS Retriever (메모리 최적화)"""
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict, Field
from sentence_transformers import SentenceTransformer
from utils.memory import cleanup_memory


class FAISSRetriever(BaseRetriever):
    """HuggingFace FAISS 인덱스를 사용하는 Retriever (메모리 최적화)"""
    
    index: Any = Field(exclude=True)
    metadata_list: List[Dict[str, Any]] = Field(exclude=True)
    embedding_model: SentenceTransformer = Field(exclude=True)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, index, metadata_list, embedding_model):
        super().__init__(index=index, metadata_list=metadata_list, embedding_model=embedding_model)
        # Pydantic 필드가 아닌 일반 인스턴스 변수로 정의 (언더스코어 사용 가능)
        self._model_on_gpu = False
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """검색 수행 (동적 GPU 로딩)"""
        # 임베딩 모델을 GPU로 일시 이동
        if not self._model_on_gpu:
            if hasattr(self.embedding_model, 'model'):
                self.embedding_model.model = self.embedding_model.model.to('cuda')
            else:
                # SentenceTransformer의 경우
                for module in self.embedding_model.modules():
                    if hasattr(module, 'to'):
                        module.to('cuda')
            self._model_on_gpu = True
        
        try:
            # 검색 수행
            # E5 instruct 모델의 올바른 prefix 형식 사용
            task_description = "한국 수능 문제의 질문에 답하기 위해 관련된 한국어 위키피디아 문서를 검색하세요"
            query_text = f"Instruct: {task_description}\nQuery: {query}"
            query_embedding = self.embedding_model.encode(
                [query_text],
                normalize_embeddings=True,  # 코사인 유사도 인덱스 사용했으니 필수
                convert_to_numpy=True
            )
            
            # FAISS 검색
            distances, indices = self.index.search(
                query_embedding.astype('float32'), 
                k=10  # 기본값
            )
            
            # 결과 생성
            docs = []
            for idx in indices[0]:
                if idx < len(self.metadata_list):
                    metadata = self.metadata_list[idx]
                    docs.append(Document(
                        page_content=metadata.get('page_content', ''),
                        metadata={
                            'title': metadata.get('title', 'N/A'),
                            'page_id': metadata.get('page_id', ''),
                            'index': int(idx)
                        }
                    ))
            
            # 임베딩 벡터 즉시 삭제
            del query_embedding, distances, indices
            
        finally:
            # 임베딩 모델을 CPU로 다시 이동 (메모리 절약)
            if self._model_on_gpu:
                if hasattr(self.embedding_model, 'model'):
                    self.embedding_model.model = self.embedding_model.model.to('cpu')
                else:
                    for module in self.embedding_model.modules():
                        if hasattr(module, 'to'):
                            module.to('cpu')
                self._model_on_gpu = False
                cleanup_memory()
        
        return docs

