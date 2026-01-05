"""
HuggingFace FAISS 인덱스 로드 모듈
"""
import json
import faiss
from pathlib import Path
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from typing import Tuple, List, Dict, Any


def load_hf_faiss_index(
    repo_id: str = "NLP-07-ODQA/kowiki-faiss-index",
    local_dir: str = "./downloaded_faiss",
    use_mmap: bool = True
) -> Tuple[faiss.Index, List[Dict[str, Any]], SentenceTransformer]:
    """
    HuggingFace에서 FAISS 인덱스를 다운로드하고 로드합니다.
    
    Args:
        repo_id: HuggingFace 데이터셋 repo ID
        local_dir: 로컬 다운로드 디렉토리
        use_mmap: mmap 사용 여부 (기본값: True, 메모리 절약)
        
    Returns:
        tuple: (faiss_index, metadata_list, embedding_model)
            - faiss_index: FAISS 인덱스 객체
            - metadata_list: 메타데이터 리스트 (각 문서의 title, page_content 등)
            - embedding_model: E5 임베딩 모델
    """
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # 1. HuggingFace 데이터 다운로드
    print(f"HuggingFace에서 FAISS 인덱스 다운로드 중: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_path)
    )
    print(f"✅ 다운로드 완료: {local_path}")
    
    # 2. 파일 경로 설정
    index_path = local_path / "faiss_index.bin"
    metadata_path = local_path / "metadata.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS 인덱스 파일을 찾을 수 없습니다: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
    
    # 3. FAISS 인덱스 읽기 (mmap 사용)
    print(f"FAISS 인덱스 로딩 중: {index_path}")
    if use_mmap:
        print("  [mmap 모드] 디스크에서 직접 접근 (메모리 절약)")
        # mmap 플래그 사용: 메모리에 전체 로드하지 않고 디스크에서 직접 접근
        # 주의: IO_FLAG_READ_ONLY (언더스코어 하나 더 있음)
        index = faiss.read_index(
            str(index_path),
            faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        )
        print(f"✅ 인덱스 로드 완료 (mmap): {index.ntotal:,} 개의 문서")
        print(f"   RAM 사용량: ~100-500MB (디스크에서 직접 접근)")
    else:
        print("  [일반 모드] 전체를 메모리에 로드")
        index = faiss.read_index(str(index_path))
        print(f"✅ 인덱스 로드 완료: {index.ntotal:,} 개의 문서")
        print(f"   RAM 사용량: ~17.7GB")
    
    # 4. 메타데이터(JSON) 읽기
    print(f"메타데이터 로딩 중: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    print(f"✅ 메타데이터 로드 완료: {len(metadata_list):,} 개의 문서")
    
    # 5. 임베딩 모델 로드 (E5 모델)
    model_name = "intfloat/multilingual-e5-large-instruct"
    print(f"임베딩 모델 로딩 중: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    print("✅ 임베딩 모델 로드 완료")
    
    return index, metadata_list, embedding_model


if __name__ == "__main__":
    # 테스트 실행
    index, metadata_list, embedding_model = load_hf_faiss_index()
    
    # 검색 테스트
    query = "조선시대 과학 기술"
    query_text = f"query: {query}"  # E5 모델 필수 접두사
    
    print(f"\n검색 테스트: '{query}'")
    query_embedding = embedding_model.encode(
        [query_text],
        normalize_embeddings=True,  # 코사인 유사도 인덱스 사용했으니 필수
        convert_to_numpy=True
    )
    
    # FAISS 검색 (k=5)
    k = 5
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    # 결과 출력
    print(f"\n--- '{query}'에 대한 검색 결과 ---")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(metadata_list):
            res = metadata_list[idx]
            print(f"[{i+1}] 거리: {dist:.4f}")
            print(f"제목: {res.get('title', 'N/A')}")
            print(f"내용: {res.get('page_content', '')[:150]}...\n")

