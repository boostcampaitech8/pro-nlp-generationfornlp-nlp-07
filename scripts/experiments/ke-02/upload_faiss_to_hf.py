"""FAISS 인덱스를 Hugging Face Hub에 업로드하는 스크립트

이미 생성된 FAISS 인덱스 파일을 Hugging Face Hub에 업로드합니다.
"""

import sys
import os
from pathlib import Path
import logging
import argparse

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# 설정 임포트
from src.config.config import FAISS_INDEX_DIR, HF_ORG

# ke-02 모듈 임포트
ke02_path = project_root / "scripts" / "experiments" / "ke-02"
sys.path.insert(0, str(ke02_path))

from hf_uploader import upload_faiss_index_to_hf

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="Upload FAISS index to Hugging Face Hub"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help=f"FAISS index directory (default: {FAISS_INDEX_DIR})"
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'NLP-07-ODQA/kowiki-faiss-index')"
    )
    parser.add_argument(
        "--index-file",
        type=str,
        default="faiss_index.bin",
        help="FAISS index file name (default: 'faiss_index.bin')"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="metadata.json",
        help="Metadata file name (default: 'metadata.json')"
    )
    
    args = parser.parse_args()
    
    try:
        # 인덱스 디렉토리 설정
        if args.index_dir:
            index_dir = Path(args.index_dir)
        else:
            index_dir = FAISS_INDEX_DIR
        
        index_dir = Path(index_dir)
        
        # 파일 경로 설정
        index_path = index_dir / args.index_file
        metadata_path = index_dir / args.metadata_file
        
        # 파일 존재 확인
        if not index_path.exists():
            logger.error(f"❌ Index file not found: {index_path}")
            logger.error("Please check the file path and try again.")
            sys.exit(1)
        
        if not metadata_path.exists():
            logger.error(f"❌ Metadata file not found: {metadata_path}")
            logger.error("Please check the file path and try again.")
            sys.exit(1)
        
        # 파일 크기 확인
        index_size = index_path.stat().st_size / 1e9
        metadata_size = metadata_path.stat().st_size / 1e6
        
        logger.info("="*80)
        logger.info("FAISS 인덱스 Hugging Face 업로드")
        logger.info("="*80)
        logger.info(f"Index directory: {index_dir}")
        logger.info(f"Index file: {index_path}")
        logger.info(f"Index file size: {index_size:.2f} GB")
        logger.info(f"Metadata file: {metadata_path}")
        logger.info(f"Metadata file size: {metadata_size:.2f} MB")
        logger.info(f"Hugging Face repository: {args.hf_repo_id}")
        logger.info("="*80)
        
        # 업로드 실행
        upload_faiss_index_to_hf(
            index_path=index_path,
            metadata_path=metadata_path,
            repo_id=args.hf_repo_id
        )
        
        logger.info("\n" + "="*80)
        logger.info("✅ FAISS 인덱스 업로드 완료!")
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

