"""FAISS 임베딩 인덱스 구축 스크립트

Hugging Face에서 청킹된 데이터셋을 로드하여 임베딩을 생성하고 FAISS 인덱스를 구축합니다.
"""

import sys
import os
from pathlib import Path
import numpy as np
from datasets import load_dataset
from langchain_core.documents import Document
import logging
from tqdm import tqdm
import torch

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# 설정 임포트
from src.config.config import (
    FAISS_INDEX_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MAX_TOKENS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    HF_ORG
)

# ke-02 모듈 임포트 (디렉토리 이름에 하이픈이 있어서 직접 경로 추가)
ke02_path = project_root / "scripts" / "experiments" / "ke-02"
sys.path.insert(0, str(ke02_path))

from markdown_chunker import prepare_embedding_text
from embedder import create_embedder
from faiss_indexer import FAISSIndexer
from hf_uploader import upload_embeddings_to_hf, upload_faiss_index_to_hf, download_embeddings_from_hf

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU memory cleared")


def check_gpu_memory():
    """GPU 메모리 상태 확인"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - reserved
        
        logger.info(f"GPU Memory Status:")
        logger.info(f"  Allocated: {allocated:.2f} GB")
        logger.info(f"  Reserved: {reserved:.2f} GB")
        logger.info(f"  Total: {total:.2f} GB")
        logger.info(f"  Free: {free:.2f} GB")
        
        return free
    return None


def load_chunked_dataset_from_hf(dataset_name: str = None):
    """Hugging Face에서 청킹된 데이터셋을 로드하고 Document 객체로 변환"""
    if dataset_name is None:
        dataset_name = f"{HF_ORG}/kowiki-cleaned-chunked-markdown-400"
    
    logger.info(f"Loading chunked dataset from Hugging Face: {dataset_name}")
    
    try:
        # 데이터셋 로드
        chunked_dataset = load_dataset(dataset_name, split="train")
        logger.info(f"✅ Dataset loaded successfully!")
        logger.info(f"Total chunks: {len(chunked_dataset)}")
        logger.info(f"Dataset features: {chunked_dataset.features}")
        
        # Dataset을 Document 객체 리스트로 변환
        logger.info("Converting dataset to Document objects...")
        all_chunks = []
        
        for item in tqdm(chunked_dataset, desc="Converting to Documents", unit="chunk"):
            # 메타데이터 구성
            metadata = {
                "title": item.get("title", ""),
                "page_id": item.get("page_id", ""),
            }
            
            # 헤더 정보 추가 (비어있지 않은 경우만)
            if item.get("header_1"):
                metadata["Header 1"] = item["header_1"]
            if item.get("header_2"):
                metadata["Header 2"] = item["header_2"]
            if item.get("header_3"):
                metadata["Header 3"] = item["header_3"]
            
            # Document 객체 생성
            doc = Document(
                page_content=item.get("page_content", ""),
                metadata=metadata
            )
            all_chunks.append(doc)
        
        logger.info(f"✅ Conversion completed!")
        logger.info(f"Total Document objects: {len(all_chunks)}")
        logger.info(f"Sample chunk metadata: {all_chunks[0].metadata if all_chunks else 'N/A'}")
        
        return all_chunks
        
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        logger.error("Please check the dataset name and try again.")
        raise


def save_embeddings(embeddings, save_path: Path):
    """임베딩을 디스크에 저장"""
    logger.info(f"Saving embeddings to {save_path}...")
    np.save(save_path, embeddings)
    logger.info(f"✅ Embeddings saved successfully! Size: {save_path.stat().st_size / 1e9:.2f} GB")


def load_embeddings(load_path: Path):
    """디스크에서 임베딩 로드"""
    logger.info(f"Loading embeddings from {load_path}...")
    embeddings = np.load(load_path)
    logger.info(f"✅ Embeddings loaded successfully! Shape: {embeddings.shape}")
    return embeddings


def generate_embeddings(all_chunks, embedder, save_path: Path = None, load_if_exists: bool = True):
    """임베딩 생성 (Python 스크립트에서는 embedder 내부 프로그래스바 사용)"""
    # 저장 경로가 있고 파일이 존재하면 로드
    if save_path and load_if_exists and save_path.exists():
        logger.info(f"Found existing embeddings at {save_path}, loading...")
        return load_embeddings(save_path)
    
    logger.info("Preparing embedding texts...")
    
    # 임베딩용 텍스트 준비
    embedding_texts = [prepare_embedding_text(chunk) for chunk in tqdm(all_chunks, desc="Preparing texts", unit="chunk")]
    
    logger.info(f"Total texts to embed: {len(embedding_texts)}")
    logger.info(f"Sample text length: {len(embedding_texts[0]) if embedding_texts else 0} characters")
    
    # 임베딩 생성 (내부 프로그래스바 사용)
    logger.info("Generating embeddings...")
    logger.info(f"Using embedder batch_size: {embedder.batch_size}")
    
    # 전체 리스트를 한 번에 전달 (내부에서 배치 처리)
    embeddings = embedder.encode_texts(
        texts=embedding_texts,
        show_progress_bar=True  # Python 스크립트에서는 내부 프로그래스바 사용
    )
    
    logger.info(f"✅ Embeddings generated successfully!")
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")
    
    # 저장 경로가 있으면 저장
    if save_path:
        save_embeddings(embeddings, save_path)
    
    return embeddings


def build_faiss_index_from_hf_streaming(
    all_chunks,
    embeddings_repo_id: str,
    embeddings_path: Path,
    index_dir: Path = None,
    batch_size: int = 100000,
    upload_to_hf: bool = False,
    hf_repo_id: str = None
):
    """Hugging Face에서 임베딩을 다운로드하고 메모리 매핑으로 배치 단위 로드하여 FAISS 인덱스 구축"""
    if index_dir is None:
        index_dir = FAISS_INDEX_DIR
    
    # Hugging Face에서 embeddings.npy 다운로드 (파일이 없을 때만)
    if not embeddings_path.exists():
        logger.info("Downloading embeddings from Hugging Face...")
        logger.info(f"Repository: {embeddings_repo_id}")
        download_embeddings_from_hf(embeddings_path, embeddings_repo_id)
    else:
        logger.info(f"Embeddings file already exists: {embeddings_path}")
        logger.info("Skipping download, using existing file.")
    
    logger.info("Loading embeddings using memory mapping (no full memory load)...")
    
    # 메모리 매핑으로 임베딩 로드 (전체를 메모리에 로드하지 않음)
    embeddings_mmap = np.load(embeddings_path, mmap_mode='r')
    embedding_dim = embeddings_mmap.shape[1]
    total_embeddings = embeddings_mmap.shape[0]
    
    logger.info(f"Embeddings shape: {embeddings_mmap.shape}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info("Using memory mapping - embeddings are not fully loaded into RAM")
    
    # FAISS 인덱스 생성
    logger.info("Creating FAISS index...")
    indexer = FAISSIndexer(
        embedding_dim=embedding_dim,
        index_type="cosine"
    )
    
    # 배치 단위로 임베딩 읽기 및 인덱스에 추가
    logger.info(f"Adding embeddings to index in batches of {batch_size}...")
    num_batches = (total_embeddings + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, total_embeddings, batch_size), desc="Adding to FAISS index", total=num_batches, unit="batch"):
        end_idx = min(i + batch_size, total_embeddings)
        
        # 메모리 매핑에서 배치만 읽기 (전체를 메모리에 로드하지 않음)
        batch_embeddings = np.array(embeddings_mmap[i:end_idx], dtype=np.float32)
        batch_chunks = all_chunks[i:end_idx]
        
        indexer.add_documents(batch_embeddings, batch_chunks)
        
        # 메모리 정리 (주기적으로)
        if (i // batch_size) % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 메모리 매핑 닫기
    del embeddings_mmap
    
    # 통계 정보
    stats = indexer.get_stats()
    logger.info("Index statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # embeddings.npy 삭제 (디스크 공간 확보 - FAISS 인덱스 저장 전에)
    logger.info(f"Deleting embeddings file to free disk space before saving index: {embeddings_path}")
    if embeddings_path.exists():
        embeddings_path.unlink()
        logger.info(f"✅ Embeddings file deleted! (Freed ~8.85 GB)")
    else:
        logger.warning(f"Embeddings file not found: {embeddings_path}")
    
    # 인덱스 저장
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss_index.bin"
    metadata_path = index_dir / "metadata.json"
    
    logger.info(f"Saving index to: {index_path}")
    indexer.save(index_path, metadata_path)
    
    logger.info(f"✅ Index saved successfully!")
    logger.info(f"Index file: {index_path}")
    logger.info(f"Metadata file: {metadata_path}")
    
    # Hugging Face에 업로드
    if upload_to_hf and hf_repo_id:
        upload_faiss_index_to_hf(index_path, metadata_path, hf_repo_id)
    
    return indexer


def build_faiss_index(
    embeddings, 
    all_chunks, 
    index_dir: Path = None, 
    batch_size: int = 100000,
    upload_to_hf: bool = False,
    hf_repo_id: str = None,
    embeddings_path: Path = None
):
    """FAISS 인덱스 구축 및 저장 (배치 단위로 처리하여 메모리 절약)"""
    if index_dir is None:
        index_dir = FAISS_INDEX_DIR
    
    logger.info("Creating FAISS index...")
    indexer = FAISSIndexer(
        embedding_dim=embeddings.shape[1],
        index_type="cosine"  # 코사인 유사도 사용
    )
    
    # 문서와 임베딩을 배치 단위로 추가 (메모리 절약)
    logger.info(f"Adding embeddings to index in batches of {batch_size}...")
    total = len(embeddings)
    num_batches = (total + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, total, batch_size), desc="Adding to FAISS index", total=num_batches, unit="batch"):
        end_idx = min(i + batch_size, total)
        batch_embeddings = embeddings[i:end_idx]
        batch_chunks = all_chunks[i:end_idx]
        
        indexer.add_documents(batch_embeddings, batch_chunks)
        
        # 메모리 정리 (주기적으로)
        if (i // batch_size) % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 통계 정보
    stats = indexer.get_stats()
    logger.info("Index statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # embeddings.npy 삭제 (디스크 공간 확보 - FAISS 인덱스 저장 전에)
    if embeddings_path and embeddings_path.exists():
        logger.info(f"Deleting embeddings file to free disk space before saving index: {embeddings_path}")
        embeddings_path.unlink()
        logger.info(f"✅ Embeddings file deleted! (Freed ~8.85 GB)")
    
    # 인덱스 저장
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss_index.bin"
    metadata_path = index_dir / "metadata.json"
    
    logger.info(f"Saving index to: {index_path}")
    indexer.save(index_path, metadata_path)
    
    logger.info(f"✅ Index saved successfully!")
    logger.info(f"Index file: {index_path}")
    logger.info(f"Metadata file: {metadata_path}")
    
    # Hugging Face에 업로드
    if upload_to_hf and hf_repo_id:
        upload_faiss_index_to_hf(index_path, metadata_path, hf_repo_id)
    
    return indexer


def test_search(indexer, embedder, test_query: str = "한국의 역사", k: int = 5):
    """인덱스 검증 (테스트 검색)"""
    logger.info(f"Test query: {test_query}")
    
    # 쿼리 임베딩 생성
    query_text = f"query: {test_query}"  # e5 모델의 query 포맷
    query_embedding = embedder.encode_texts([query_text], show_progress_bar=False)
    
    # 검색
    results = indexer.search(query_embedding[0], k=k)
    
    logger.info(f"\nTop {k} search results:")
    logger.info("="*80)
    for i, result in enumerate(results, 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"  Distance: {result['distance']:.4f}")
        logger.info(f"  Title: {result['metadata'].get('title', 'N/A')}")
        logger.info(f"  Page ID: {result['metadata'].get('page_id', 'N/A')}")
        
        # Header 경로
        header_path = []
        for level in ["Header 1", "Header 2", "Header 3"]:
            if level in result['metadata']:
                header_path.append(result['metadata'][level])
        if header_path:
            logger.info(f"  Header path: {' > '.join(header_path)}")
        
        content = result['metadata'].get('page_content', '')
        logger.info(f"  Content preview: {content[:200]}...")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index from chunked dataset")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Hugging Face dataset name (default: NLP-07-ODQA/kowiki-cleaned-chunked-markdown-400)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help=f"Embedding batch size (default: {EMBEDDING_BATCH_SIZE})"
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use FP16 precision for embeddings"
    )
    parser.add_argument(
        "--test-query",
        type=str,
        default="한국의 역사",
        help="Test query for index validation (default: '한국의 역사')"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test search"
    )
    parser.add_argument(
        "--faiss-batch-size",
        type=int,
        default=100000,
        help="Batch size for adding embeddings to FAISS index (default: 100000)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation and load from disk if exists"
    )
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload FAISS index to Hugging Face Hub"
    )
    parser.add_argument(
        "--upload-embeddings-to-hf",
        action="store_true",
        help="Upload embeddings to Hugging Face Hub"
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Hugging Face repository ID for uploading files (e.g., 'NLP-07-ODQA/kowiki-faiss-index')"
    )
    parser.add_argument(
        "--hf-embeddings-repo-id",
        type=str,
        default=None,
        help="Hugging Face repository ID for uploading embeddings (default: uses hf-repo-id or auto-generated)"
    )
    
    args = parser.parse_args()
    
    try:
        # 1. 환경 설정
        logger.info("="*80)
        logger.info("FAISS 임베딩 인덱스 구축 시작")
        logger.info("="*80)
        logger.info(f"Project root: {project_root}")
        logger.info(f"FAISS index directory: {FAISS_INDEX_DIR}")
        
        # 임베딩 저장 경로 설정
        embeddings_dir = project_root / "data" / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = embeddings_dir / "embeddings.npy"
        
        # 임베딩 생성 건너뛰기 여부 확인
        skip_embeddings = args.skip_embeddings or embeddings_path.exists()
        
        if skip_embeddings and embeddings_path.exists():
            logger.info("Skipping embedding generation (using existing embeddings)")
        elif skip_embeddings and not embeddings_path.exists():
            logger.info("Embeddings file not found locally. Will download from Hugging Face and use streaming mode.")
        
        # 2. Hugging Face에서 청킹된 데이터셋 로드 (임베딩 생성 시에만 필요)
        if not skip_embeddings:
            logger.info("\n" + "="*80)
            logger.info("Step 1: Loading chunked dataset from Hugging Face")
            logger.info("="*80)
            all_chunks = load_chunked_dataset_from_hf(args.dataset_name)
            
            # 3. Embedder 생성
            logger.info("\n" + "="*80)
            logger.info("Step 2: Creating embedder")
            logger.info("="*80)
            
            # GPU 메모리 정리 및 확인
            if EMBEDDING_DEVICE == "cuda" and torch.cuda.is_available():
                logger.info("Checking GPU memory before loading model...")
                free_memory = check_gpu_memory()
                
                if free_memory and free_memory < 2.0:  # 2GB 미만이면 정리 시도
                    logger.warning(f"GPU memory is low ({free_memory:.2f} GB). Attempting to clear cache...")
                    clear_gpu_memory()
                    free_memory = check_gpu_memory()
                    
                    if free_memory and free_memory < 2.0:
                        logger.error(f"Insufficient GPU memory ({free_memory:.2f} GB). Please free up GPU memory or use CPU.")
                        logger.error("You can try:")
                        logger.error("  1. Kill other processes using GPU: nvidia-smi to find PIDs, then kill them")
                        logger.error("  2. Use CPU instead: set EMBEDDING_DEVICE='cpu' in config")
                        logger.error("  3. Reduce batch size: --batch-size 128 or smaller")
                        raise RuntimeError(f"Insufficient GPU memory: {free_memory:.2f} GB available")
            
            logger.info(f"Model: {EMBEDDING_MODEL_NAME}")
            logger.info(f"Device: {EMBEDDING_DEVICE}, Batch size: {args.batch_size}")
            logger.info(f"Use FP16: {args.use_fp16}")
            
            embedder = create_embedder(
                model_name=EMBEDDING_MODEL_NAME,
                device=EMBEDDING_DEVICE,
                batch_size=args.batch_size,
                max_length=EMBEDDING_MAX_TOKENS,
                use_fp16=args.use_fp16
            )
            
            logger.info(f"Embedding dimension: {embedder.get_embedding_dim()}")
            
            # 4. 임베딩 생성
            logger.info("\n" + "="*80)
            logger.info("Step 3: Generating embeddings")
            logger.info("="*80)
            logger.info(f"Embeddings will be saved to: {embeddings_path}")
            embeddings = generate_embeddings(
                all_chunks, 
                embedder, 
                save_path=embeddings_path,
                load_if_exists=False  # 이미 skip_embeddings로 체크했으므로 False
            )
        else:
            # 임베딩 로드 또는 Hugging Face에서 다운로드
            use_streaming = False
            
            if not embeddings_path.exists():
                # 파일이 없으면 Hugging Face에서 다운로드하고 스트리밍 방식 사용
                logger.info("\n" + "="*80)
                logger.info("Step 1: Downloading embeddings from Hugging Face")
                logger.info("="*80)
                
                # 리포지토리 ID 설정
                embeddings_repo_id = args.hf_embeddings_repo_id
                if not embeddings_repo_id:
                    if args.hf_repo_id:
                        embeddings_repo_id = args.hf_repo_id
                    else:
                        if args.dataset_name:
                            dataset_name = args.dataset_name.split("/")[-1]
                            embeddings_repo_id = f"{HF_ORG}/{dataset_name}-embeddings"
                        else:
                            embeddings_repo_id = f"{HF_ORG}/kowiki-cleaned-chunked-markdown-400-embeddings"
                    logger.info(f"Using default embeddings repository ID: {embeddings_repo_id}")
                
                try:
                    download_embeddings_from_hf(embeddings_path, embeddings_repo_id)
                    use_streaming = True  # 다운로드 후 스트리밍 방식 사용
                except Exception as e:
                    logger.error(f"❌ Failed to download embeddings from Hugging Face: {e}")
                    raise
            
            if use_streaming:
                # 스트리밍 방식: 메모리 매핑 사용, FAISS 인덱스 구축 후 embeddings.npy 삭제
                logger.info("\n" + "="*80)
                logger.info("Step 2: Loading chunked dataset from Hugging Face (for metadata)")
                logger.info("="*80)
                all_chunks = load_chunked_dataset_from_hf(args.dataset_name)
                
                logger.info("\n" + "="*80)
                logger.info("Step 4: Building FAISS index (streaming mode)")
                logger.info("="*80)
                
                # Hugging Face 업로드 설정
                upload_to_hf = args.upload_to_hf
                hf_repo_id = args.hf_repo_id
                
                if upload_to_hf and not hf_repo_id:
                    # 기본 리포지토리 ID 생성
                    if args.dataset_name:
                        dataset_name = args.dataset_name.split("/")[-1]
                        hf_repo_id = f"{HF_ORG}/{dataset_name}-faiss-index"
                    else:
                        hf_repo_id = f"{HF_ORG}/kowiki-cleaned-chunked-markdown-400-faiss-index"
                    logger.info(f"Using default Hugging Face repo ID: {hf_repo_id}")
                
                indexer = build_faiss_index_from_hf_streaming(
                    all_chunks,
                    embeddings_repo_id,
                    embeddings_path,
                    batch_size=args.faiss_batch_size,
                    upload_to_hf=upload_to_hf,
                    hf_repo_id=hf_repo_id
                )
            else:
                # 일반 방식: 로컬 파일에서 로드
                logger.info("\n" + "="*80)
                logger.info("Step 1: Loading embeddings from disk")
                logger.info("="*80)
                embeddings = load_embeddings(embeddings_path)
                
                logger.info("\n" + "="*80)
                logger.info("Step 2: Loading chunked dataset from Hugging Face")
                logger.info("="*80)
                all_chunks = load_chunked_dataset_from_hf(args.dataset_name)
                
                logger.info("\n" + "="*80)
                logger.info("Step 4: Building FAISS index")
                logger.info("="*80)
                
                # Hugging Face 업로드 설정
                upload_to_hf = args.upload_to_hf
                hf_repo_id = args.hf_repo_id
                
                if upload_to_hf and not hf_repo_id:
                    # 기본 리포지토리 ID 생성
                    if args.dataset_name:
                        dataset_name = args.dataset_name.split("/")[-1]
                        hf_repo_id = f"{HF_ORG}/{dataset_name}-faiss-index"
                    else:
                        hf_repo_id = f"{HF_ORG}/kowiki-cleaned-chunked-markdown-400-faiss-index"
                    logger.info(f"Using default Hugging Face repo ID: {hf_repo_id}")
                
                # 임베딩을 Hugging Face에 업로드 (선택사항, 파일이 있을 때만)
                if args.upload_embeddings_to_hf and embeddings_path.exists():
                    logger.info("\n" + "="*80)
                    logger.info("Step 3.5: Uploading embeddings to Hugging Face")
                    logger.info("="*80)
                    
                    # 리포지토리 ID 설정
                    embeddings_repo_id = args.hf_embeddings_repo_id
                    if not embeddings_repo_id:
                        if args.hf_repo_id:
                            embeddings_repo_id = args.hf_repo_id
                        else:
                            if args.dataset_name:
                                dataset_name = args.dataset_name.split("/")[-1]
                                embeddings_repo_id = f"{HF_ORG}/{dataset_name}-embeddings"
                            else:
                                embeddings_repo_id = f"{HF_ORG}/kowiki-cleaned-chunked-markdown-400-embeddings"
                        logger.info(f"Using default embeddings repository ID: {embeddings_repo_id}")
                    
                    upload_embeddings_to_hf(embeddings_path, embeddings_repo_id)
                
                indexer = build_faiss_index(
                    embeddings, 
                    all_chunks, 
                    batch_size=args.faiss_batch_size,
                    upload_to_hf=upload_to_hf,
                    hf_repo_id=hf_repo_id,
                    embeddings_path=embeddings_path
                )
        
        # 6. 테스트 검색 (선택사항)
        if not args.skip_test:
            logger.info("\n" + "="*80)
            logger.info("Step 5: Testing index (test search)")
            logger.info("="*80)
            
            if not skip_embeddings:
                # Embedder가 이미 생성되어 있음
                test_search(indexer, embedder, args.test_query)
            else:
                # Embedder 생성 (테스트용)
                logger.info("Creating embedder for test search...")
                embedder = create_embedder(
                    model_name=EMBEDDING_MODEL_NAME,
                    device=EMBEDDING_DEVICE,
                    batch_size=args.batch_size,
                    max_length=EMBEDDING_MAX_TOKENS,
                    use_fp16=args.use_fp16
                )
                test_search(indexer, embedder, args.test_query)
        
        logger.info("\n" + "="*80)
        logger.info("✅ FAISS 인덱스 구축 완료!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("❌ Error occurred: " + str(e))
        logger.error("="*80)
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()