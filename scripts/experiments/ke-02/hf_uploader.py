"""Hugging Face Hub 업로드/다운로드 모듈"""

from pathlib import Path
from huggingface_hub import upload_file, create_repo, HfApi, hf_hub_download
import logging

logger = logging.getLogger(__name__)


def ensure_repo_exists(repo_id: str, repo_type: str = "dataset"):
    """리포지토리가 존재하는지 확인하고 없으면 생성"""
    api = HfApi()
    
    try:
        # 리포지토리 존재 여부 확인
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        logger.info(f"Repository {repo_id} already exists")
    except Exception:
        # 리포지토리가 없으면 생성
        logger.info(f"Repository {repo_id} does not exist. Creating...")
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            exist_ok=False
        )
        logger.info(f"✅ Repository {repo_id} created successfully!")


def download_embeddings_from_hf(
    embeddings_path: Path,
    repo_id: str,
    filename: str = "embeddings.npy",
    repo_type: str = "dataset"
):
    """Hugging Face Hub에서 임베딩 파일 다운로드"""
    logger.info(f"Downloading embeddings from Hugging Face: {repo_id}")
    
    # 디렉토리 생성
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 파일 다운로드
        logger.info(f"Downloading {filename} from {repo_id}...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(embeddings_path.parent),
            local_dir_use_symlinks=False
        )
        
        # 다운로드된 파일을 원하는 경로로 이동 (필요한 경우)
        if downloaded_path != str(embeddings_path):
            from shutil import move
            move(downloaded_path, embeddings_path)
        
        # 파일 크기 확인
        embeddings_size = embeddings_path.stat().st_size / 1e9
        logger.info(f"✅ Embeddings downloaded successfully! Size: {embeddings_size:.2f} GB")
        logger.info(f"✅ File saved to: {embeddings_path}")
        
        return embeddings_path
        
    except Exception as e:
        logger.error(f"❌ Error downloading embeddings from Hugging Face: {e}")
        raise


def upload_embeddings_to_hf(
    embeddings_path: Path,
    repo_id: str,
    repo_type: str = "dataset"
):
    """임베딩 파일을 Hugging Face Hub에 업로드"""
    logger.info(f"Uploading embeddings to Hugging Face: {repo_id}")
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    # 리포지토리 존재 확인 및 생성
    ensure_repo_exists(repo_id, repo_type)
    
    # 파일 크기 확인
    embeddings_size = embeddings_path.stat().st_size / 1e9
    
    logger.info(f"Embeddings file size: {embeddings_size:.2f} GB")
    
    try:
        # 임베딩 파일 업로드
        logger.info(f"Uploading embeddings file: {embeddings_path.name}...")
        upload_file(
            path_or_fileobj=str(embeddings_path),
            path_in_repo=embeddings_path.name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Upload embeddings"
        )
        logger.info(f"✅ Embeddings file uploaded successfully!")
        logger.info(f"✅ File uploaded to: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        logger.error(f"❌ Error uploading embeddings to Hugging Face: {e}")
        raise


def upload_faiss_index_to_hf(
    index_path: Path,
    metadata_path: Path,
    repo_id: str,
    repo_type: str = "dataset"
):
    """FAISS 인덱스를 Hugging Face Hub에 업로드"""
    logger.info(f"Uploading FAISS index to Hugging Face: {repo_id}")
    
    # 리포지토리 존재 확인 및 생성
    ensure_repo_exists(repo_id, repo_type)
    
    # 파일 크기 확인
    index_size = index_path.stat().st_size / 1e9
    metadata_size = metadata_path.stat().st_size / 1e6
    
    logger.info(f"Index file size: {index_size:.2f} GB")
    logger.info(f"Metadata file size: {metadata_size:.2f} MB")
    
    try:
        # 인덱스 파일 업로드
        logger.info(f"Uploading index file: {index_path.name}...")
        upload_file(
            path_or_fileobj=str(index_path),
            path_in_repo=index_path.name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Upload FAISS index"
        )
        logger.info(f"✅ Index file uploaded successfully!")
        
        # 메타데이터 파일 업로드
        logger.info(f"Uploading metadata file: {metadata_path.name}...")
        upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo=metadata_path.name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Upload FAISS metadata"
        )
        logger.info(f"✅ Metadata file uploaded successfully!")
        
        logger.info(f"✅ All files uploaded to: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        logger.error(f"❌ Error uploading to Hugging Face: {e}")
        raise

