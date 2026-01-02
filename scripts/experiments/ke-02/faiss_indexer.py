"""FAISS 인덱싱 모듈"""

import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)


class FAISSIndexer:
    """FAISS 인덱스 생성 및 관리 클래스"""
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "L2"  # "L2" 또는 "cosine"
    ):
        """
        FAISSIndexer 초기화
        
        Args:
            embedding_dim: 임베딩 차원
            index_type: 인덱스 타입 ("L2" 또는 "cosine")
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # FAISS 인덱스 생성
        if index_type == "cosine":
            # 코사인 유사도를 위한 내적 인덱스 (정규화된 벡터 사용)
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            # L2 거리 인덱스
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        self.metadata_list: List[Dict[str, Any]] = []
        logger.info(f"Initialized FAISS index: type={index_type}, dim={embedding_dim}")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ):
        """
        임베딩 벡터와 메타데이터를 인덱스에 추가
        
        Args:
            embeddings: 임베딩 벡터 배열 (numpy array)
            metadata: 메타데이터 리스트
        """
        if len(embeddings) != len(metadata):
            raise ValueError(f"Embeddings length ({len(embeddings)}) != metadata length ({len(metadata)})")
        
        # 코사인 유사도인 경우 벡터 정규화
        if self.index_type == "cosine":
            # 이미 정규화되어 있다고 가정 (embedder에서 normalize_embeddings=True)
            pass
        
        # FAISS는 float32를 요구
        embeddings = embeddings.astype('float32')
        
        # 인덱스에 추가
        self.index.add(embeddings)
        
        # 메타데이터 저장
        self.metadata_list.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} embeddings to index. Total: {self.index.ntotal}")
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[Document]
    ):
        """
        Document 리스트를 인덱스에 추가
        
        Args:
            embeddings: 임베딩 벡터 배열
            documents: LangChain Document 리스트
        """
        # Document를 메타데이터 딕셔너리로 변환
        metadata = []
        for doc in documents:
            meta = {
                "title": doc.metadata.get("title", ""),
                "page_id": doc.metadata.get("page_id", ""),
                "page_content": doc.page_content,
            }
            # Header 정보 추가
            for level in ["Header 1", "Header 2", "Header 3"]:
                if level in doc.metadata:
                    meta[level] = doc.metadata[level]
            
            metadata.append(meta)
        
        self.add_embeddings(embeddings, metadata)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        쿼리 임베딩으로 유사한 문서 검색
        
        Args:
            query_embedding: 쿼리 임베딩 벡터 (1차원 또는 2차원)
            k: 반환할 상위 k개 결과
        
        Returns:
            검색 결과 리스트 (각각 distance, metadata 포함)
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # 2차원으로 변환
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # float32로 변환
        query_embedding = query_embedding.astype('float32')
        
        # 코사인 유사도인 경우 정규화
        if self.index_type == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # 검색
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # 결과 구성
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = float(distances[0][i])
            
            if idx < len(self.metadata_list):
                result = {
                    "distance": distance,
                    "metadata": self.metadata_list[idx].copy()
                }
                results.append(result)
        
        return results
    
    def save(
        self,
        index_path: Path,
        metadata_path: Optional[Path] = None
    ):
        """
        인덱스와 메타데이터를 파일로 저장
        
        Args:
            index_path: FAISS 인덱스 저장 경로
            metadata_path: 메타데이터 저장 경로 (None이면 index_path와 같은 디렉토리에 저장)
        """
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # 메타데이터 저장
        if metadata_path is None:
            metadata_path = index_path.parent / f"{index_path.stem}_metadata.json"
        
        metadata_path = Path(metadata_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    @classmethod
    def load(
        cls,
        index_path: Path,
        metadata_path: Optional[Path] = None
    ) -> 'FAISSIndexer':
        """
        저장된 인덱스와 메타데이터 로드
        
        Args:
            index_path: FAISS 인덱스 파일 경로
            metadata_path: 메타데이터 파일 경로 (None이면 자동 탐색)
        
        Returns:
            FAISSIndexer 인스턴스
        """
        index_path = Path(index_path)
        
        # FAISS 인덱스 로드
        index = faiss.read_index(str(index_path))
        
        # 인덱스 타입 확인
        if isinstance(index, faiss.IndexFlatIP):
            index_type = "cosine"
        else:
            index_type = "L2"
        
        # 임베딩 차원 확인
        embedding_dim = index.d
        
        # 인스턴스 생성
        indexer = cls(embedding_dim=embedding_dim, index_type=index_type)
        indexer.index = index
        
        # 메타데이터 로드
        if metadata_path is None:
            metadata_path = index_path.parent / f"{index_path.stem}_metadata.json"
        
        metadata_path = Path(metadata_path)
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                indexer.metadata_list = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
        
        logger.info(f"Loaded FAISS index from {index_path}, total vectors: {index.ntotal}")
        return indexer
    
    def get_stats(self) -> Dict[str, Any]:
        """인덱스 통계 정보 반환"""
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "metadata_count": len(self.metadata_list)
        }


