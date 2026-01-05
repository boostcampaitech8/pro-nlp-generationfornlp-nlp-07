"""임베딩 모듈 (GPU 배치 처리 최적화)"""

import torch
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """임베딩 생성 클래스 (GPU 배치 처리 지원)"""
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
        use_fp16: bool = False
    ):
        """
        Embedder 초기화
        
        Args:
            model_name: 임베딩 모델명
            device: 디바이스 ("cuda" 또는 "cpu")
            batch_size: 배치 크기
            max_length: 최대 토큰 길이 (512 토큰 제한)
            use_fp16: FP16 사용 여부
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Device: {self.device}, Batch size: {batch_size}, Max length: {max_length}")
        
        # 모델 로딩
        self.model = SentenceTransformer(model_name, device=self.device)
        
        if self.use_fp16 and self.device == "cuda":
            logger.info("Using FP16 precision")
            self.model = self.model.half()
    
    def encode_texts(
        self,
        texts: List[str],
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        텍스트 리스트를 임베딩으로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            show_progress_bar: 진행 상황 표시 여부
            normalize_embeddings: 임베딩 정규화 여부
        
        Returns:
            임베딩 벡터 배열 (numpy array)
        """
        if not texts:
            logger.warning("Empty texts list provided")
            return np.array([])
        
        try:
            # e5 모델의 instruction 포맷 적용 (passage: 접두사)
            formatted_texts = [f"passage: {text}" for text in texts]
            
            # sentence-transformers의 encode()는 max_length, truncate 파라미터를 지원하지 않음
            # 대신 청킹 단계에서 이미 400자로 제한했으므로 대부분 512 토큰을 넘지 않음
            embeddings = self.model.encode(
                formatted_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                device=self.device,
                normalize_embeddings=normalize_embeddings
            )
            
            # logger.info(f"Generated embeddings for {len(texts)} texts, shape: {embeddings.shape}")
            return embeddings
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"GPU out of memory, reducing batch size from {self.batch_size} to {self.batch_size // 2}")
                self.batch_size = max(1, self.batch_size // 2)
                # 재시도
                return self.encode_texts(texts, show_progress_bar, normalize_embeddings)
            else:
                logger.error(f"Error encoding texts: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error encoding texts: {e}")
            raise
    
    def encode_documents(
        self,
        documents: List[Document],
        prepare_text_func=None,
        show_progress_bar: bool = True
    ) -> np.ndarray:
        """
        Document 리스트를 임베딩으로 변환
        
        Args:
            documents: LangChain Document 리스트
            prepare_text_func: 텍스트 준비 함수 (기본값: page_content 사용)
            show_progress_bar: 진행 상황 표시 여부
        
        Returns:
            임베딩 벡터 배열
        """
        if not documents:
            logger.warning("Empty documents list provided")
            return np.array([])
        
        # 텍스트 추출
        if prepare_text_func:
            texts = [prepare_text_func(doc) for doc in documents]
        else:
            texts = [doc.page_content for doc in documents]
        
        return self.encode_texts(texts, show_progress_bar)
    
    def get_embedding_dim(self) -> int:
        """임베딩 차원 반환"""
        return self.model.get_sentence_embedding_dimension()


def create_embedder(
    model_name: str = "intfloat/multilingual-e5-large-instruct",
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 512,
    use_fp16: bool = False
) -> Embedder:
    """
    Embedder 인스턴스 생성 헬퍼 함수
    
    Args:
        model_name: 임베딩 모델명
        device: 디바이스
        batch_size: 배치 크기
        max_length: 최대 토큰 길이
        use_fp16: FP16 사용 여부
    
    Returns:
        Embedder 인스턴스
    """
    return Embedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        use_fp16=use_fp16
    )


