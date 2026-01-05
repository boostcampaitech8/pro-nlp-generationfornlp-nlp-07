# FAISS 임베딩 인덱스 구축 모듈

이 모듈은 Hugging Face에서 청킹된 데이터셋을 로드하여 임베딩을 생성하고 FAISS 인덱스를 구축하는 기능을 제공합니다.

## 📁 파일 구조

### 주요 모듈

#### 1. `markdown_chunker.py`
**역할**: 마크다운 문서를 청킹하는 모듈

- **주요 함수**:
  - `chunk_markdown_document()`: 마크다운 문서를 헤더 기반으로 청킹
  - `prepare_embedding_text()`: 임베딩 생성을 위한 텍스트 포맷팅

- **특징**:
  - LangChain의 `MarkdownHeaderTextSplitter` 사용
  - 헤더 정보(Header 1, 2, 3)를 메타데이터로 보존
  - 512 토큰 제한을 고려한 청킹 크기 조정 (기본 400자)
  - title, page_id 등 원본 문서 정보 유지

#### 2. `embedder.py`
**역할**: 텍스트 임베딩 생성 모듈 (GPU 배치 처리 최적화)

- **주요 클래스**: `Embedder`
  - `encode_texts()`: 텍스트 리스트를 임베딩 벡터로 변환
  - `encode_documents()`: Document 객체 리스트를 임베딩 벡터로 변환

- **특징**:
  - `sentence-transformers` 기반
  - 기본 모델: `intfloat/multilingual-e5-large-instruct` (1024차원)
  - GPU 배치 처리 지원
  - FP16 정밀도 옵션 (메모리 절약 및 속도 향상)
  - OOM 에러 시 자동 배치 크기 감소

#### 3. `faiss_indexer.py`
**역할**: FAISS 인덱스 생성 및 관리 모듈

- **주요 클래스**: `FAISSIndexer`
  - `add_embeddings()`: 임베딩 벡터와 메타데이터를 인덱스에 추가
  - `add_documents()`: Document 객체를 인덱스에 추가
  - `search()`: 유사도 검색 수행
  - `save()`: 인덱스를 디스크에 저장
  - `load()`: 디스크에서 인덱스 로드

- **특징**:
  - 코사인 유사도 또는 L2 거리 지원
  - 메타데이터와 함께 저장/로드
  - 대용량 데이터 처리 최적화

#### 4. `hf_uploader.py`
**역할**: Hugging Face Hub 업로드/다운로드 유틸리티 모듈

- **주요 함수**:
  - `ensure_repo_exists()`: 리포지토리 존재 확인 및 자동 생성
  - `upload_embeddings_to_hf()`: 임베딩 파일(`embeddings.npy`) 업로드
  - `upload_faiss_index_to_hf()`: FAISS 인덱스 및 메타데이터 업로드
  - `download_embeddings_from_hf()`: Hugging Face에서 임베딩 파일 다운로드

- **특징**:
  - 리포지토리 자동 생성 (404 에러 방지)
  - 대용량 파일 업로드 지원
  - 파일 크기 확인 및 로깅

#### 5. `build_faiss_index.py`
**역할**: 메인 실행 스크립트 - 전체 워크플로우 통합

- **주요 기능**:
  1. Hugging Face에서 청킹된 데이터셋 로드
  2. 임베딩 생성 (GPU 배치 처리)
  3. 임베딩 저장 및 Hugging Face Hub 업로드 (선택사항)
  4. FAISS 인덱스 구축 (메모리 매핑 지원)
  5. 인덱스 저장 및 Hugging Face Hub 업로드 (선택사항)
  6. 인덱스 검증 (테스트 검색)

- **특징**:
  - 중간 결과물 자동 저장 (임베딩)
  - 메모리 매핑을 사용한 스트리밍 방식 지원 (디스크 공간 절약)
  - Hugging Face에서 임베딩 자동 다운로드
  - FAISS 인덱스 구축 후 `embeddings.npy` 자동 삭제 (디스크 공간 확보)
  - 배치 단위 처리로 메모리 효율성 확보
  - 프로그래스바 표시
  - GPU 메모리 관리

#### 6. `upload_faiss_to_hf.py`
**역할**: 이미 생성된 FAISS 인덱스를 Hugging Face Hub에 업로드하는 독립 스크립트

- **주요 기능**:
  - 로컬에 저장된 FAISS 인덱스 파일 업로드
  - 메타데이터 파일 업로드

- **사용 시나리오**:
  - `build_faiss_index.py`에서 업로드를 건너뛰고 나중에 업로드하고 싶을 때
  - 다른 환경에서 생성된 인덱스를 업로드할 때

## 🚀 실행 방법

### 기본 사용법

#### 1. 전체 프로세스 실행 (임베딩 생성 + 인덱스 구축)

```bash
python scripts/experiments/ke-02/build_faiss_index.py --use-fp16 --batch-size 512
```

#### 2. FAISS 인덱스 구축만 실행 (임베딩은 이미 생성됨)

```bash
python scripts/experiments/ke-02/build_faiss_index.py --skip-embeddings
```

#### 3. 임베딩을 Hugging Face Hub에 업로드

```bash
python scripts/experiments/ke-02/build_faiss_index.py \
    --upload-embeddings-to-hf \
    --hf-embeddings-repo-id "NLP-07-ODQA/kowiki-embeddings"
```

#### 4. Hugging Face에서 임베딩 다운로드 후 FAISS 인덱스 구축 (스트리밍 방식)

```bash
python scripts/experiments/ke-02/build_faiss_index.py \
    --skip-embeddings \
    --hf-embeddings-repo-id "NLP-07-ODQA/kowiki-embeddings" \
    --upload-to-hf \
    --hf-repo-id "NLP-07-ODQA/kowiki-faiss-index"
```

이 방식은:
- Hugging Face에서 `embeddings.npy`를 다운로드
- 메모리 매핑을 사용하여 전체를 메모리에 로드하지 않고 배치 단위로 읽음
- FAISS 인덱스 구축 후 `embeddings.npy`를 자동 삭제하여 디스크 공간 확보
- 최종적으로 FAISS 인덱스만 저장 (약 16.5 GB)

#### 5. 이미 생성된 FAISS 인덱스를 Hugging Face Hub에 업로드

```bash
python scripts/experiments/ke-02/upload_faiss_to_hf.py \
    --hf-repo-id "NLP-07-ODQA/kowiki-faiss-index"
```

### 명령줄 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--dataset-name` | Hugging Face 데이터셋 이름 | `NLP-07-ODQA/kowiki-cleaned-chunked-markdown-400` |
| `--batch-size` | 임베딩 생성 배치 크기 | `128` |
| `--use-fp16` | FP16 정밀도 사용 (GPU 메모리 절약) | `False` |
| `--faiss-batch-size` | FAISS 인덱스 추가 배치 크기 | `100000` |
| `--skip-embeddings` | 임베딩 생성 건너뛰기 (로컬 파일 또는 Hugging Face에서 다운로드) | `False` |
| `--upload-to-hf` | FAISS 인덱스를 Hugging Face Hub에 업로드 | `False` |
| `--upload-embeddings-to-hf` | 임베딩 파일을 Hugging Face Hub에 업로드 | `False` |
| `--hf-repo-id` | FAISS 인덱스용 Hugging Face 리포지토리 ID | 자동 생성 |
| `--hf-embeddings-repo-id` | 임베딩용 Hugging Face 리포지토리 ID | 자동 생성 |
| `--test-query` | 테스트 검색 쿼리 | `"한국의 역사"` |
| `--skip-test` | 테스트 검색 건너뛰기 | `False` |

### 실행 예시

#### 예시 1: FP16 사용하여 빠르게 임베딩 생성

```bash
python scripts/experiments/ke-02/build_faiss_index.py \
    --use-fp16 \
    --batch-size 512
```

#### 예시 2: 임베딩 생성 후 Hugging Face에 업로드

```bash
python scripts/experiments/ke-02/build_faiss_index.py \
    --use-fp16 \
    --batch-size 512 \
    --upload-embeddings-to-hf \
    --hf-embeddings-repo-id "NLP-07-ODQA/kowiki-embeddings" \
    --upload-to-hf \
    --hf-repo-id "NLP-07-ODQA/kowiki-faiss-index"
```

#### 예시 3: 기존 임베딩으로 인덱스만 재구축

```bash
python scripts/experiments/ke-02/build_faiss_index.py \
    --skip-embeddings \
    --faiss-batch-size 50000
```

#### 예시 4: Hugging Face에서 임베딩 다운로드 후 인덱스 구축 (디스크 공간 절약)

```bash
python scripts/experiments/ke-02/build_faiss_index.py \
    --skip-embeddings \
    --hf-embeddings-repo-id "NLP-07-ODQA/kowiki-embeddings" \
    --upload-to-hf \
    --hf-repo-id "NLP-07-ODQA/kowiki-faiss-index"
```

이 방식은 메모리 매핑을 사용하여 `embeddings.npy`를 전체 메모리에 로드하지 않고 배치 단위로 읽어 FAISS 인덱스를 구축합니다. 인덱스 구축 후 `embeddings.npy`가 자동으로 삭제되어 디스크 공간을 절약합니다.

#### 예시 5: 테스트 검색 없이 인덱스만 구축

```bash
python scripts/experiments/ke-02/build_faiss_index.py \
    --skip-embeddings \
    --skip-test
```

#### 예시 6: 독립 스크립트로 FAISS 인덱스 업로드

```bash
python scripts/experiments/ke-02/upload_faiss_to_hf.py \
    --hf-repo-id "NLP-07-ODQA/kowiki-faiss-index" \
    --index-dir "data/faiss_index"
```

## 📊 워크플로우

### 전체 프로세스 (임베딩 생성 포함)

```
1. 데이터셋 로드 (Hugging Face)
   ↓
2. Document 객체 변환
   ↓
3. 임베딩 생성 (GPU 배치 처리)
   ↓
4. 임베딩 저장 (data/embeddings/embeddings.npy)
   ↓
5. 임베딩을 Hugging Face Hub에 업로드 (선택사항)
   ↓
6. FAISS 인덱스 구축 (배치 단위)
   ↓
7. embeddings.npy 삭제 (디스크 공간 확보)
   ↓
8. 인덱스 저장 (data/faiss_index/)
   ↓
9. FAISS 인덱스를 Hugging Face Hub에 업로드 (선택사항)
   ↓
10. 인덱스 검증 (테스트 검색)
```

### 스트리밍 방식 (디스크 공간 절약)

```
1. Hugging Face에서 embeddings.npy 다운로드 (없는 경우)
   ↓
2. 데이터셋 로드 (Hugging Face) - 메타데이터용
   ↓
3. 메모리 매핑으로 embeddings.npy 로드 (전체 메모리 로드 없음)
   ↓
4. FAISS 인덱스 구축 (배치 단위로 임베딩 읽기)
   ↓
5. embeddings.npy 삭제 (디스크 공간 확보, ~8.85 GB)
   ↓
6. 인덱스 저장 (data/faiss_index/)
   ↓
7. FAISS 인덱스를 Hugging Face Hub에 업로드 (선택사항)
   ↓
8. 인덱스 검증 (테스트 검색)
```

**스트리밍 방식의 장점:**
- `embeddings.npy` (약 8.85 GB)를 전체 메모리에 로드하지 않음
- FAISS 인덱스 구축 후 `embeddings.npy` 자동 삭제로 디스크 공간 절약
- 최종적으로 FAISS 인덱스 (약 16.5 GB)만 저장
- 총 필요한 디스크 공간: 약 16.5 GB (임베딩 파일 제외)

### 중간 결과물 저장

- **임베딩**: `data/embeddings/embeddings.npy` (약 8.85 GB)
- **FAISS 인덱스**: `data/faiss_index/faiss_index.bin`
- **메타데이터**: `data/faiss_index/metadata.json`

## ⚙️ 설정

주요 설정은 `src/config/config.py`에서 관리됩니다:

- `EMBEDDING_MODEL_NAME`: 임베딩 모델명
- `EMBEDDING_BATCH_SIZE`: 기본 배치 크기
- `EMBEDDING_DEVICE`: 디바이스 ("cuda" 또는 "cpu")
- `EMBEDDING_MAX_TOKENS`: 최대 토큰 길이 (512)
- `FAISS_INDEX_DIR`: FAISS 인덱스 저장 디렉토리
- `HF_ORG`: Hugging Face 조직명

## 🔧 문제 해결

### GPU 메모리 부족

```bash
# 배치 크기 줄이기
python scripts/experiments/ke-02/build_faiss_index.py --batch-size 128

# 또는 CPU 사용 (느리지만 안전)
# config.py에서 EMBEDDING_DEVICE = "cpu"로 변경
```

### 디스크 공간 부족

**일반 방식:**
- FAISS 인덱스 파일: 약 16.5 GB
- 임베딩 파일: 약 8.85 GB
- 총 약 25 GB 이상의 여유 공간 필요

**스트리밍 방식 (권장):**
- Hugging Face에서 `embeddings.npy` 다운로드
- 메모리 매핑을 사용하여 전체 메모리 로드 없이 FAISS 인덱스 구축
- 인덱스 구축 후 `embeddings.npy` 자동 삭제
- 최종적으로 FAISS 인덱스 (약 16.5 GB)만 저장
- 총 필요한 디스크 공간: 약 16.5 GB + 다운로드 중 임시 공간 (~8.85 GB)

```bash
# 스트리밍 방식 사용
python scripts/experiments/ke-02/build_faiss_index.py \
    --skip-embeddings \
    --hf-embeddings-repo-id "NLP-07-ODQA/kowiki-embeddings"
```

### 기존 임베딩 재사용

**로컬 파일:**
- 임베딩 파일이 이미 있으면 자동으로 로드됩니다
- 파일 경로: `data/embeddings/embeddings.npy`
- `--skip-embeddings` 옵션으로 명시적으로 건너뛸 수 있음

**Hugging Face에서 다운로드:**
- 로컬에 `embeddings.npy`가 없으면 Hugging Face에서 자동 다운로드
- `--hf-embeddings-repo-id`로 리포지토리 ID 지정
- 다운로드 후 메모리 매핑 방식으로 FAISS 인덱스 구축

## 📝 참고사항

1. **임베딩 저장**: 임베딩은 자동으로 저장되며, 다음 실행 시 재사용됩니다.
2. **메타데이터**: 현재는 임베딩만 저장하고 메타데이터는 저장하지 않습니다. 따라서 `--skip-embeddings` 사용 시에도 데이터셋을 다시 로드해야 합니다.
3. **배치 처리**: 대용량 데이터 처리를 위해 배치 단위로 처리하며, 메모리 사용량을 최적화합니다.
4. **프로그래스바**: 각 단계에서 진행 상황을 표시합니다.
5. **메모리 매핑**: 스트리밍 방식에서는 NumPy의 메모리 매핑(`mmap_mode='r'`)을 사용하여 `embeddings.npy`를 전체 메모리에 로드하지 않고 배치 단위로 읽습니다.
6. **자동 삭제**: FAISS 인덱스 구축 후 `embeddings.npy`가 자동으로 삭제되어 디스크 공간을 절약합니다. (약 8.85 GB)
7. **코사인 유사도**: FAISS 인덱스는 코사인 유사도(`IndexFlatIP`)를 사용하며, 모든 임베딩 벡터는 정규화되어 저장됩니다. 쿼리 임베딩 생성 시에도 `normalize_embeddings=True`를 사용해야 합니다.

## 🔗 관련 파일

- 설정 파일: `src/config/config.py`
- Hugging Face 업로드 모듈: `scripts/experiments/ke-02/hf_uploader.py`
- FAISS 인덱스 업로드 스크립트: `scripts/experiments/ke-02/upload_faiss_to_hf.py`

## 📦 Hugging Face 리포지토리

### 임베딩 리포지토리
- **용도**: `embeddings.npy` 파일 저장
- **예시**: `NLP-07-ODQA/kowiki-embeddings`
- **파일**: `embeddings.npy` (약 8.85 GB)

### FAISS 인덱스 리포지토리
- **용도**: FAISS 인덱스 및 메타데이터 저장
- **예시**: `NLP-07-ODQA/kowiki-faiss-index`
- **파일**: 
  - `faiss_index.bin` (약 16.5 GB)
  - `metadata.json` (약 수백 MB)

### RAG 사용 시
RAG 시스템에서 FAISS 인덱스를 사용할 때는 **FAISS 인덱스 리포지토리**만 필요합니다. `embeddings.npy`는 FAISS 인덱스 구축에만 사용되며, RAG에서는 필요하지 않습니다.