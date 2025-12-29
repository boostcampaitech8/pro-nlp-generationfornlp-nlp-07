"""RAG dataset creator for Wikipedia data"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import Dataset
from huggingface_hub import HfApi, login

from .wikipedia_parser import parse_wikipedia_xml_stream, is_redirect, is_stub
from .wikipedia_cleaner import clean_wikitext
from src.config.config import HF_ORG, HF_TOKEN
from src.utils.hf_utils import get_hf_token, login_to_hf

# Global variables to track batches for streaming upload
_hf_batch_counters = {}
_hf_batch_files = {}  # Track uploaded batch file paths for final merging
_hf_local_batch_files = {}  # Track local temporary batch files for merging


def _upload_batch_to_hf(
    chunks: List[Dict],
    dataset_name: str,
    chunk_size: int,
    is_first_batch: bool = False,
):
    """Upload a batch of chunks to Hugging Face Hub using temporary files"""
    global _hf_batch_counters, _hf_batch_files, _hf_local_batch_files
    
    if not chunks:
        return
    
    config_name = f"chunks_{chunk_size}"
    
    # Initialize counters
    if config_name not in _hf_batch_counters:
        _hf_batch_counters[config_name] = 0
        _hf_batch_files[config_name] = []
        _hf_local_batch_files[config_name] = []
    
    _hf_batch_counters[config_name] += 1
    batch_num = _hf_batch_counters[config_name]
    
    # Create temporary Parquet file
    from tempfile import NamedTemporaryFile
    import os
    
    df = pd.DataFrame(chunks)
    
    with NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Save to temporary Parquet file
        df.to_parquet(tmp_path, compression='snappy', index=False)
        
        # Keep local copy for final merging (will be deleted after merge)
        _hf_local_batch_files[config_name].append(tmp_path)
        
        # Upload using HfApi
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Upload batch file
        batch_filename = f"data/{config_name}_batch_{batch_num}.parquet"
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=batch_filename,
            repo_id=dataset_name,
            repo_type="dataset",
            commit_message=f"Add batch {batch_num} for {config_name}",
        )
        _hf_batch_files[config_name].append(batch_filename)
        print(f"  ✓ Uploaded batch {batch_num}: {len(chunks):,} chunks (size: {chunk_size})")
    except Exception as e:
        # If upload fails, clean up local file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def create_rag_dataset(
    xml_path: str,
    chunk_sizes: List[int],
    output_dir: Path,
    overlap_ratio: float = 0.2,
    min_text_length: int = 200,
    namespace: int = 0,
    max_pages: Optional[int] = None,
    sample_rate: float = 1.0,
    min_korean_ratio: Optional[float] = None,
    upload_to_hf: bool = False,
    hf_dataset_name: Optional[str] = None,
    hf_dataset_version: Optional[str] = None,
    streaming_upload: bool = False,
    batch_size: int = 10000,
) -> Dict:
    """
    Create RAG dataset from Wikipedia XML dump
    
    Args:
        xml_path: Path to Wikipedia XML dump file
        chunk_sizes: List of chunk sizes to create (e.g., [500, 700, 1000])
        output_dir: Directory to save output files
        overlap_ratio: Overlap ratio between chunks (default: 0.2 = 20%)
        min_text_length: Minimum text length after cleaning (default: 200)
        namespace: Namespace to filter (0 = main articles, default: 0)
        max_pages: Maximum number of pages to process (None for all)
        sample_rate: Sampling rate (0.0 to 1.0)
        min_korean_ratio: Minimum Korean character ratio (None to disable)
        streaming_upload: If True, upload chunks to HF in batches without saving to disk
        batch_size: Number of chunks to accumulate before uploading (default: 10000)
        
    Returns:
        Dictionary with statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize statistics
    stats = {
        'total_pages': 0,
        'filtered_pages': 0,
        'chunks_by_size': defaultdict(int),
        'chunk_configs': {}
    }
    
    # Create chunk configurations
    chunk_configs = {}
    hf_initialized = False
    
    for chunk_size in chunk_sizes:
        overlap = int(chunk_size * overlap_ratio)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        chunk_configs[chunk_size] = {
            'splitter': splitter,
            'overlap': overlap,
            'chunks': [] if not streaming_upload else None,  # Don't store in memory if streaming
            'batch_chunks': [] if streaming_upload else None,  # For batch accumulation
        }
        stats['chunk_configs'][chunk_size] = {
            'chunk_size': chunk_size,
            'overlap': overlap,
            'overlap_ratio': overlap_ratio
        }
    
    # Initialize Hugging Face if streaming upload
    if streaming_upload and upload_to_hf:
        if not hf_dataset_name:
            raise ValueError("hf_dataset_name is required when streaming_upload=True")
        
        token = get_hf_token()
        if not token:
            raise ValueError("Hugging Face token not found. Please set HF_TOKEN in .env file.")
        
        login_to_hf(token)
        
        # Prepare dataset name
        if not hf_dataset_name.startswith(f"{HF_ORG}/"):
            hf_dataset_name = f"{HF_ORG}/{hf_dataset_name}"
        
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            api.dataset_info(hf_dataset_name)
            print(f"Dataset repository already exists: {hf_dataset_name}")
        except Exception:
            try:
                api.create_repo(
                    repo_id=hf_dataset_name,
                    repo_type="dataset",
                    private=False,
                    exist_ok=True
                )
                print(f"Created dataset repository: {hf_dataset_name}")
            except Exception as e:
                print(f"Failed to create repository: {e}")
                raise
        
        hf_initialized = True
        print(f"Streaming upload enabled: uploading in batches of {batch_size:,} chunks")
    
    # Process pages
    print(f"Processing Wikipedia XML dump: {xml_path}")
    print(f"Filters: namespace={namespace}, min_length={min_text_length}, "
          f"max_pages={max_pages}, sample_rate={sample_rate}")
    
    for page in tqdm(
        parse_wikipedia_xml_stream(xml_path, max_pages, sample_rate),
        desc="Processing pages"
    ):
        stats['total_pages'] += 1
        
        # Filter by namespace
        if page.get('ns', 0) != namespace:
            continue
        
        # Get page content
        title = page.get('title', '')
        text = page.get('text', '')
        page_id = page.get('page_id')
        
        # Skip redirects and empty pages
        if not text or is_redirect(text):
            continue
        
        # Clean text
        cleaned_text = clean_wikitext(text)
        
        # Filter by minimum length
        if len(cleaned_text) < min_text_length:
            continue
        
        # Filter by Korean ratio if specified
        if min_korean_ratio is not None:
            korean_count = len(re.findall(r'[가-힣]', cleaned_text))
            total_chars = len(cleaned_text)
            if total_chars > 0:
                korean_ratio = korean_count / total_chars
                if korean_ratio < min_korean_ratio:
                    continue
        
        # Skip stubs if they're too short after cleaning
        if is_stub(cleaned_text):
            continue
        
        stats['filtered_pages'] += 1
        
        # Extract sections from markdown text
        sections = _extract_sections_from_markdown(cleaned_text)
        
        # If no sections found, treat entire content as one section
        if not sections:
            sections = [('', cleaned_text)]
        
        # Create chunks for each chunk size
        for chunk_size, config in chunk_configs.items():
            splitter = config['splitter']
            global_chunk_idx = 0  # Track chunk index across all sections
            
            # Process each section separately
            for section_title, section_content in sections:
                # Skip empty sections
                if not section_content.strip():
                    continue
                
                # Create chunks within this section
                section_chunks = splitter.split_text(section_content)
                
                # Filter out chunks that are too short
                section_chunks = [chunk for chunk in section_chunks if len(chunk.strip()) >= min_text_length // 2]
                
                # Create chunk records for this section
                for section_chunk_idx, chunk_text in enumerate(section_chunks):
                    chunk_id = f"{title.replace(' ', '_')}_chunk{global_chunk_idx}"
                    
                    chunk_record = {
                        'content': chunk_text.strip(),
                        'title': title,
                        'section': section_title,  # Section title from markdown heading
                        'chunk_id': chunk_id,
                        'chunk_index': global_chunk_idx,
                        'chunk_size': chunk_size,
                        'page_id': page_id,
                    }
                    
                    if streaming_upload:
                        # Accumulate in batch
                        config['batch_chunks'].append(chunk_record)
                        stats['chunks_by_size'][chunk_size] += 1
                        
                        # Upload batch when it reaches batch_size
                        if len(config['batch_chunks']) >= batch_size:
                            _upload_batch_to_hf(
                                chunks=config['batch_chunks'],
                                dataset_name=hf_dataset_name,
                                chunk_size=chunk_size,
                                is_first_batch=(stats['chunks_by_size'][chunk_size] == len(config['batch_chunks']))
                            )
                            config['batch_chunks'] = []  # Clear batch
                    else:
                        # Store in memory for later saving
                        config['chunks'].append(chunk_record)
                        stats['chunks_by_size'][chunk_size] += 1
                    
                    global_chunk_idx += 1
    
    # Handle remaining batches for streaming upload
    if streaming_upload and upload_to_hf:
        print("\nUploading remaining batches...")
        for chunk_size, config in chunk_configs.items():
            if config['batch_chunks']:
                _upload_batch_to_hf(
                    chunks=config['batch_chunks'],
                    dataset_name=hf_dataset_name,
                    chunk_size=chunk_size,
                    is_first_batch=False
                )
                config['batch_chunks'] = []
        
        # Merge all batches into final datasets
        # Use streaming approach to minimize disk usage
        print("\nMerging batches into final datasets...")
        import os
        from datasets import concatenate_datasets
        
        for chunk_size in chunk_sizes:
            config_name = f"chunks_{chunk_size}"
            if config_name not in _hf_local_batch_files or not _hf_local_batch_files[config_name]:
                continue
            
            batch_files = [f for f in _hf_local_batch_files[config_name] if os.path.exists(f)]
            if not batch_files:
                continue
            
            print(f"Merging {len(batch_files)} batches for {config_name}...")
            
            # Stream batches one at a time to minimize memory/disk usage
            # Load and concatenate in chunks
            all_dataframes = []
            
            for i, batch_file in enumerate(batch_files):
                batch_df = pd.read_parquet(batch_file)
                all_dataframes.append(batch_df)
                
                # Clean up immediately after loading
                os.remove(batch_file)
                
                # Periodically merge to avoid memory issues
                if len(all_dataframes) >= 10:  # Merge every 10 batches
                    merged_df = pd.concat(all_dataframes, ignore_index=True)
                    all_dataframes = [merged_df]
                    print(f"  Merged {i+1}/{len(batch_files)} batches...")
            
            # Final merge
            if all_dataframes:
                final_df = pd.concat(all_dataframes, ignore_index=True)
                final_dataset = Dataset.from_pandas(final_df)
                
                # Push final merged dataset
                final_dataset.push_to_hub(
                    repo_id=hf_dataset_name,
                    config_name=config_name,
                    private=False,
                )
                print(f"  ✓ Merged and uploaded final dataset: {len(final_dataset):,} chunks (size: {chunk_size})")
                
                # Clear memory
                del final_df, final_dataset
            
            # Clear batch files lists
            _hf_batch_files[config_name] = []
            _hf_local_batch_files[config_name] = []
    
    # Save datasets (only if not streaming)
    if not streaming_upload:
        print("\nSaving datasets...")
        for chunk_size, config in chunk_configs.items():
            if not config['chunks']:
                print(f"Warning: No chunks created for size {chunk_size}")
                continue
            
            df = pd.DataFrame(config['chunks'])
            output_file = output_dir / f"chunks_{chunk_size}.parquet"
            df.to_parquet(output_file, compression='snappy', index=False)
            print(f"Saved {len(df)} chunks to {output_file}")
    
    # Save metadata
    metadata = {
        'statistics': {
            'total_pages': stats['total_pages'],
            'filtered_pages': stats['filtered_pages'],
            'chunks_by_size': dict(stats['chunks_by_size']),
        },
        'config': {
            'chunk_sizes': chunk_sizes,
            'overlap_ratio': overlap_ratio,
            'min_text_length': min_text_length,
            'namespace': namespace,
            'max_pages': max_pages,
            'sample_rate': sample_rate,
            'min_korean_ratio': min_korean_ratio,
        },
        'chunk_configs': stats['chunk_configs'],
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    # Upload to Hugging Face if requested
    if upload_to_hf:
        if not hf_dataset_name:
            raise ValueError("hf_dataset_name is required when upload_to_hf=True")
        
        print("\n" + "=" * 60)
        print("Uploading datasets to Hugging Face")
        print("=" * 60)
        
        upload_datasets_to_hf(
            output_dir=output_dir,
            dataset_name=hf_dataset_name,
            chunk_sizes=chunk_sizes,
            metadata=metadata,
            version=hf_dataset_version,
        )
    
    return stats


def create_dataset_card(
    dataset_name: str,
    chunk_sizes: List[int],
    metadata: Dict,
    version: Optional[str] = None,
) -> str:
    """Create dataset card (README.md) for Hugging Face"""
    
    stats = metadata.get('statistics', {})
    config = metadata.get('config', {})
    
    card = f"""---
license: cc-by-sa-4.0
task_categories:
- text-generation
- question-answering
language:
- ko
tags:
- korean
- wikipedia
- rag
- retrieval-augmented-generation
- csat
- nlp-competition
size_categories:
- 100K<n<1M
---

# {dataset_name}

## Dataset Description

이 데이터셋은 한국어 위키피디아 XML 덤프에서 추출하고 정제한 RAG(Retrieval-Augmented Generation)용 데이터셋입니다.

### 데이터 소스

- **원본**: 한국어 위키피디아 XML 덤프
- **네임스페이스**: {config.get('namespace', 0)} (일반 문서)
- **처리된 페이지 수**: {stats.get('filtered_pages', 0):,}개
- **전체 페이지 수**: {stats.get('total_pages', 0):,}개

### 데이터 정제 과정

1. **위키 마크업 제거**: `[[링크]]`, `{{템플릿}}`, `==제목==` 등 제거
2. **불필요한 섹션 제거**: '같이 보기', '외부 링크', '참고 문헌', '각주' 등 제거
3. **필터링**: 리다이렉트, 빈 페이지, 스텁 페이지 제거
4. **최소 길이**: {config.get('min_text_length', 200)}자 이상만 포함

### 청킹 전략

- **방법**: RecursiveCharacterTextSplitter (LangChain)
- **구분자 우선순위**: `["\\n\\n", "\\n", ".", " ", ""]`
- **Overlap 비율**: {config.get('overlap_ratio', 0.2) * 100:.0f}%

### 청크 크기별 통계

"""
    
    for chunk_size in chunk_sizes:
        count = stats.get('chunks_by_size', {}).get(chunk_size, 0)
        overlap = int(chunk_size * config.get('overlap_ratio', 0.2))
        card += f"- **{chunk_size}자**: {count:,}개 청크 (overlap: {overlap}자)\n"
    
    card += f"""
## 데이터 구조

각 데이터 포인트는 다음 필드를 포함합니다:

- `content`: 정제된 텍스트 청크
- `title`: 문서 제목
- `section`: 섹션 제목 (없으면 빈 문자열)
- `chunk_id`: 고유 청크 ID
- `chunk_index`: 문서 내 청크 순서
- `chunk_size`: 청크 크기 (500/700/1000)
- `page_id`: 위키피디아 페이지 ID

## 사용 예시

```python
from datasets import load_dataset

# 500자 청크 데이터셋 로드
dataset_500 = load_dataset("{dataset_name}", "chunks_500")

# 700자 청크 데이터셋 로드
dataset_700 = load_dataset("{dataset_name}", "chunks_700")

# 1000자 청크 데이터셋 로드
dataset_1000 = load_dataset("{dataset_name}", "chunks_1000")
```

## 데이터셋 생성 설정

- **Chunk Sizes**: {', '.join(map(str, chunk_sizes))}
- **Overlap Ratio**: {config.get('overlap_ratio', 0.2) * 100:.0f}%
- **Min Text Length**: {config.get('min_text_length', 200)}자
- **Sample Rate**: {config.get('sample_rate', 1.0)}
"""
    
    if config.get('max_pages'):
        card += f"- **Max Pages**: {config.get('max_pages'):,}\n"
    
    if config.get('min_korean_ratio'):
        card += f"- **Min Korean Ratio**: {config.get('min_korean_ratio') * 100:.1f}%\n"
    
    card += f"""
## 라이선스

이 데이터셋은 한국어 위키피디아의 CC BY-SA 4.0 라이선스를 따릅니다.

## 참고사항

- 이 데이터셋은 수능 국어/사회 과목 문제 풀이를 위한 RAG 시스템 구축을 목적으로 생성되었습니다.
- 각 청크 크기별로 별도의 데이터셋으로 제공됩니다.
- 청크 간 overlap이 포함되어 있어 문맥 단절을 방지합니다.
"""
    
    if version:
        card += f"\n## 버전\n\n- **현재 버전**: {version}\n"
    
    return card


def upload_datasets_to_hf(
    output_dir: Path,
    dataset_name: str,
    chunk_sizes: List[int],
    metadata: Dict,
    version: Optional[str] = None,
):
    """Upload datasets to Hugging Face Hub"""
    
    # Login to Hugging Face
    token = get_hf_token()
    if not token:
        raise ValueError("Hugging Face token not found. Please set HF_TOKEN in .env file.")
    
    login_to_hf(token)
    
    # Prepare dataset name
    if not dataset_name.startswith(f"{HF_ORG}/"):
        dataset_name = f"{HF_ORG}/{dataset_name}"
    
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.dataset_info(dataset_name)
        print(f"Dataset repository already exists: {dataset_name}")
    except Exception:
        try:
            api.create_repo(
                repo_id=dataset_name,
                repo_type="dataset",
                private=False,
                exist_ok=True
            )
            print(f"Created dataset repository: {dataset_name}")
        except Exception as e:
            print(f"Failed to create repository: {e}")
            raise
    
    # Create and upload dataset card
    card_content = create_dataset_card(dataset_name, chunk_sizes, metadata, version)
    card_path = output_dir / "README.md"
    card_path.write_text(card_content, encoding='utf-8')
    
    print(f"Uploading dataset card...")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=dataset_name,
        repo_type="dataset",
    )
    
    # Upload each chunk size dataset
    for chunk_size in chunk_sizes:
        parquet_file = output_dir / f"chunks_{chunk_size}.parquet"
        if not parquet_file.exists():
            print(f"Warning: {parquet_file} not found, skipping...")
            continue
        
        print(f"\nUploading {chunk_size} char chunks dataset...")
        
        # Load parquet and convert to Hugging Face Dataset
        df = pd.read_parquet(parquet_file)
        dataset = Dataset.from_pandas(df)
        
        # Upload to Hugging Face
        dataset.push_to_hub(
            repo_id=dataset_name,
            config_name=f"chunks_{chunk_size}",
            private=False,
        )
        
        print(f"✓ Uploaded {len(dataset):,} chunks (size: {chunk_size})")
    
    # Upload metadata
    metadata_file = output_dir / "metadata.json"
    if metadata_file.exists():
        print(f"\nUploading metadata...")
        api.upload_file(
            path_or_fileobj=str(metadata_file),
            path_in_repo="metadata.json",
            repo_id=dataset_name,
            repo_type="dataset",
        )
    
    dataset_url = f"https://huggingface.co/datasets/{dataset_name}"
    print(f"\n{'='*60}")
    print(f"Dataset uploaded successfully!")
    print(f"URL: {dataset_url}")
    print(f"{'='*60}")


def create_cleaned_dataset(
    xml_path: str,
    output_dir: Path,
    min_text_length: int = 200,
    namespace: int = 0,
    max_pages: Optional[int] = None,
    sample_rate: float = 1.0,
    min_korean_ratio: Optional[float] = None,
    upload_to_hf: bool = False,
    hf_dataset_name: Optional[str] = None,
    hf_dataset_version: Optional[str] = None,
) -> Dict:
    """
    Create cleaned dataset from Wikipedia XML dump (without chunking)
    
    Args:
        xml_path: Path to Wikipedia XML dump file
        output_dir: Directory to save output files
        min_text_length: Minimum text length after cleaning (default: 200)
        namespace: Namespace to filter (0 = main articles, default: 0)
        max_pages: Maximum number of pages to process (None for all)
        sample_rate: Sampling rate (0.0 to 1.0)
        min_korean_ratio: Minimum Korean character ratio (None to disable)
        upload_to_hf: Whether to upload to Hugging Face Hub
        hf_dataset_name: Hugging Face dataset name (e.g., "kowiki-cleaned")
        hf_dataset_version: Dataset version tag (e.g., "v1.0")
        
    Returns:
        Dictionary with statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize statistics
    stats = {
        'total_pages': 0,
        'filtered_pages': 0,
        'cleaned_pages': [],
        'text_lengths': [],
        'korean_ratios': [],
    }
    
    # Process pages
    print(f"Processing Wikipedia XML dump: {xml_path}")
    print(f"Filters: namespace={namespace}, min_length={min_text_length}, "
          f"max_pages={max_pages}, sample_rate={sample_rate}")
    
    for page in tqdm(
        parse_wikipedia_xml_stream(xml_path, max_pages, sample_rate),
        desc="Processing pages"
    ):
        stats['total_pages'] += 1
        
        # Filter by namespace
        if page.get('ns', 0) != namespace:
            continue
        
        # Get page content
        title = page.get('title', '')
        text = page.get('text', '')
        page_id = page.get('page_id')
        
        # Skip redirects and empty pages
        if not text or is_redirect(text):
            continue
        
        # Clean text
        cleaned_text = clean_wikitext(text)
        
        # Filter by minimum length
        if len(cleaned_text) < min_text_length:
            continue
        
        # Filter by Korean ratio if specified
        if min_korean_ratio is not None:
            korean_count = len(re.findall(r'[가-힣]', cleaned_text))
            total_chars = len(cleaned_text)
            if total_chars > 0:
                korean_ratio = korean_count / total_chars
                if korean_ratio < min_korean_ratio:
                    continue
                stats['korean_ratios'].append(korean_ratio)
        
        # Skip stubs if they're too short after cleaning
        if is_stub(cleaned_text):
            continue
        
        stats['filtered_pages'] += 1
        stats['text_lengths'].append(len(cleaned_text))
        
        # Store cleaned page
        page_record = {
            'content': cleaned_text,
            'title': title,
            'page_id': page_id,
            'text_length': len(cleaned_text),
        }
        stats['cleaned_pages'].append(page_record)
    
    # Save cleaned dataset
    print("\nSaving cleaned dataset...")
    df = pd.DataFrame(stats['cleaned_pages'])
    output_file = output_dir / "cleaned_pages.parquet"
    df.to_parquet(output_file, compression='snappy', index=False)
    print(f"Saved {len(df):,} cleaned pages to {output_file}")
    
    # Calculate statistics
    avg_text_length = sum(stats['text_lengths']) / len(stats['text_lengths']) if stats['text_lengths'] else 0
    avg_korean_ratio = sum(stats['korean_ratios']) / len(stats['korean_ratios']) if stats['korean_ratios'] else None
    
    # Save metadata
    metadata = {
        'statistics': {
            'total_pages': stats['total_pages'],
            'filtered_pages': stats['filtered_pages'],
            'avg_text_length': avg_text_length,
            'avg_korean_ratio': avg_korean_ratio,
        },
        'config': {
            'min_text_length': min_text_length,
            'namespace': namespace,
            'max_pages': max_pages,
            'sample_rate': sample_rate,
            'min_korean_ratio': min_korean_ratio,
        },
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    # Upload to Hugging Face if requested
    if upload_to_hf:
        if not hf_dataset_name:
            raise ValueError("hf_dataset_name is required when upload_to_hf=True")
        
        print("\n" + "=" * 60)
        print("Uploading cleaned dataset to Hugging Face")
        print("=" * 60)
        
        upload_cleaned_dataset_to_hf(
            output_dir=output_dir,
            dataset_name=hf_dataset_name,
            metadata=metadata,
            version=hf_dataset_version,
        )
    
    return {
        'total_pages': stats['total_pages'],
        'filtered_pages': stats['filtered_pages'],
        'avg_text_length': avg_text_length,
        'avg_korean_ratio': avg_korean_ratio,
    }


def create_cleaned_dataset_card(
    dataset_name: str,
    metadata: Dict,
    version: Optional[str] = None,
) -> str:
    """Create dataset card (README.md) for cleaned dataset"""
    
    stats = metadata.get('statistics', {})
    config = metadata.get('config', {})
    
    card = f"""---
license: cc-by-sa-4.0
task_categories:
- text-generation
- question-answering
language:
- ko
tags:
- korean
- wikipedia
- rag
- retrieval-augmented-generation
- csat
- nlp-competition
size_categories:
- 100K<n<1M
---

# {dataset_name}

## Dataset Description

이 데이터셋은 한국어 위키피디아 XML 덤프에서 추출하고 정제한 데이터셋입니다. 청킹 전 단계의 정제된 문서를 포함합니다.

### 데이터 소스

- **원본**: 한국어 위키피디아 XML 덤프
- **네임스페이스**: {config.get('namespace', 0)} (일반 문서)
- **처리된 페이지 수**: {stats.get('filtered_pages', 0):,}개
- **전체 페이지 수**: {stats.get('total_pages', 0):,}개

### 데이터 정제 과정

1. **위키 마크업 제거**: `[[링크]]`, `{{템플릿}}`, `==제목==` 등 제거
2. **불필요한 섹션 제거**: '같이 보기', '외부 링크', '참고 문헌', '각주' 등 제거
3. **필터링**: 리다이렉트, 빈 페이지, 스텁 페이지 제거
4. **최소 길이**: {config.get('min_text_length', 200)}자 이상만 포함

### 통계

- **평균 텍스트 길이**: {stats.get('avg_text_length', 0):.1f}자
"""
    
    if stats.get('avg_korean_ratio'):
        card += f"- **평균 한글 비율**: {stats.get('avg_korean_ratio', 0) * 100:.1f}%\n"
    
    card += f"""
## 데이터 구조

각 데이터 포인트는 다음 필드를 포함합니다:

- `content`: 정제된 텍스트 (전체 문서)
- `title`: 문서 제목
- `page_id`: 위키피디아 페이지 ID
- `text_length`: 정제된 텍스트 길이 (문자 수)

## 사용 예시

```python
from datasets import load_dataset

# 정제된 데이터셋 로드
dataset = load_dataset("{dataset_name}")

# 데이터 확인
print(dataset['train'][0])
```

## 데이터셋 생성 설정

- **Min Text Length**: {config.get('min_text_length', 200)}자
- **Sample Rate**: {config.get('sample_rate', 1.0)}
"""
    
    if config.get('max_pages'):
        card += f"- **Max Pages**: {config.get('max_pages'):,}\n"
    
    if config.get('min_korean_ratio'):
        card += f"- **Min Korean Ratio**: {config.get('min_korean_ratio') * 100:.1f}%\n"
    
    card += f"""
## 라이선스

이 데이터셋은 한국어 위키피디아의 CC BY-SA 4.0 라이선스를 따릅니다.

## 참고사항

- 이 데이터셋은 수능 국어/사회 과목 문제 풀이를 위한 RAG 시스템 구축을 목적으로 생성되었습니다.
- 청킹 전 단계의 정제된 문서를 포함합니다. 청킹이 필요한 경우 별도의 청킹 스크립트를 사용하세요.
"""
    
    if version:
        card += f"\n## 버전\n\n- **현재 버전**: {version}\n"
    
    return card


def upload_cleaned_dataset_to_hf(
    output_dir: Path,
    dataset_name: str,
    metadata: Dict,
    version: Optional[str] = None,
):
    """Upload cleaned dataset to Hugging Face Hub"""
    
    # Login to Hugging Face
    token = get_hf_token()
    if not token:
        raise ValueError("Hugging Face token not found. Please set HF_TOKEN in .env file.")
    
    login_to_hf(token)
    
    # Prepare dataset name
    if not dataset_name.startswith(f"{HF_ORG}/"):
        dataset_name = f"{HF_ORG}/{dataset_name}"
    
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.dataset_info(dataset_name)
        print(f"Dataset repository already exists: {dataset_name}")
    except Exception:
        try:
            api.create_repo(
                repo_id=dataset_name,
                repo_type="dataset",
                private=False,
                exist_ok=True
            )
            print(f"Created dataset repository: {dataset_name}")
        except Exception as e:
            print(f"Failed to create repository: {e}")
            raise
    
    # Create and upload dataset card
    card_content = create_cleaned_dataset_card(dataset_name, metadata, version)
    card_path = output_dir / "README.md"
    card_path.write_text(card_content, encoding='utf-8')
    
    print(f"Uploading dataset card...")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=dataset_name,
        repo_type="dataset",
    )
    
    # Upload cleaned dataset
    parquet_file = output_dir / "cleaned_pages.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Cleaned dataset file not found: {parquet_file}")
    
    print(f"\nUploading cleaned dataset...")
    
    # Load parquet and convert to Hugging Face Dataset
    df = pd.read_parquet(parquet_file)
    dataset = Dataset.from_pandas(df)
    
    # Upload to Hugging Face
    dataset.push_to_hub(
        repo_id=dataset_name,
        private=False,
    )
    
    print(f"✓ Uploaded {len(dataset):,} cleaned pages")
    
    # Upload metadata
    metadata_file = output_dir / "metadata.json"
    if metadata_file.exists():
        print(f"\nUploading metadata...")
        api.upload_file(
            path_or_fileobj=str(metadata_file),
            path_in_repo="metadata.json",
            repo_id=dataset_name,
            repo_type="dataset",
        )
    
    dataset_url = f"https://huggingface.co/datasets/{dataset_name}"
    print(f"\n{'='*60}")
    print(f"Dataset uploaded successfully!")
    print(f"URL: {dataset_url}")
    print(f"{'='*60}")
def _extract_sections_from_markdown(text: str) -> List[tuple]:
    """
    Extract sections from markdown text based on headings (#, ##, etc.)
    
    Args:
        text: Markdown formatted text
        
    Returns:
        List of (section_title, section_content) tuples
        First section may have empty title if there's no heading at the start
    """
    if not text:
        return []
    
    sections = []
    lines = text.split('\n')
    current_section_title = ''  # Empty for content before first heading
    current_section_content = []
    
    for line in lines:
        # Check if line is a markdown heading (#, ##, ###, etc.)
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if heading_match:
            # Save previous section if it has content
            if current_section_title or current_section_content:
                section_text = '\n'.join(current_section_content).strip()
                if section_text:  # Only add non-empty sections
                    sections.append((current_section_title, section_text))
            
            # Start new section
            current_section_title = heading_match.group(2).strip()
            current_section_content = []
        else:
            current_section_content.append(line)
    
    # Add last section
    if current_section_title or current_section_content:
        section_text = '\n'.join(current_section_content).strip()
        if section_text:  # Only add non-empty sections
            sections.append((current_section_title, section_text))
    
    return sections


def create_chunks_from_cleaned(
    cleaned_data_path: str,
    chunk_sizes: List[int],
    output_dir: Path,
    overlap_ratio: float = 0.2,
    min_chunk_length: int = 100,
    upload_to_hf: bool = False,
    hf_dataset_name: Optional[str] = None,
    hf_dataset_version: Optional[str] = None,
    streaming_upload: bool = False,
    batch_size: int = 10000,
) -> Dict:
    """
    Create chunked dataset from cleaned dataset
    
    Args:
        cleaned_data_path: Path to cleaned dataset Parquet file
        chunk_sizes: List of chunk sizes to create (e.g., [500, 700, 1000])
        output_dir: Directory to save output files
        overlap_ratio: Overlap ratio between chunks (default: 0.2 = 20%)
        min_chunk_length: Minimum chunk length (default: 100)
        upload_to_hf: Whether to upload to Hugging Face Hub
        hf_dataset_name: Hugging Face dataset name (e.g., "kowiki-rag-chunks")
        hf_dataset_version: Dataset version tag (e.g., "v1.0")
        streaming_upload: If True, upload chunks to HF in batches without saving to disk
        batch_size: Number of chunks to accumulate before uploading (default: 10000)
        
    Returns:
        Dictionary with statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cleaned dataset
    print(f"Loading cleaned dataset from {cleaned_data_path}...")
    df = pd.read_parquet(cleaned_data_path)
    print(f"Loaded {len(df):,} cleaned pages")
    
    # Initialize statistics
    stats = {
        'total_pages': len(df),
        'chunks_by_size': defaultdict(int),
        'chunk_configs': {}
    }
    
    # Create chunk configurations
    chunk_configs = {}
    hf_initialized = False
    
    for chunk_size in chunk_sizes:
        overlap = int(chunk_size * overlap_ratio)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        chunk_configs[chunk_size] = {
            'splitter': splitter,
            'overlap': overlap,
            'chunks': [] if not streaming_upload else None,
            'batch_chunks': [] if streaming_upload else None,
        }
        stats['chunk_configs'][chunk_size] = {
            'chunk_size': chunk_size,
            'overlap': overlap,
            'overlap_ratio': overlap_ratio
        }
    
    # Initialize Hugging Face if streaming upload
    if streaming_upload and upload_to_hf:
        if not hf_dataset_name:
            raise ValueError("hf_dataset_name is required when streaming_upload=True")
        
        token = get_hf_token()
        if not token:
            raise ValueError("Hugging Face token not found. Please set HF_TOKEN in .env file.")
        
        login_to_hf(token)
        
        # Prepare dataset name
        if not hf_dataset_name.startswith(f"{HF_ORG}/"):
            hf_dataset_name = f"{HF_ORG}/{hf_dataset_name}"
        
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            api.dataset_info(hf_dataset_name)
            print(f"Dataset repository already exists: {hf_dataset_name}")
        except Exception:
            try:
                api.create_repo(
                    repo_id=hf_dataset_name,
                    repo_type="dataset",
                    private=False,
                    exist_ok=True
                )
                print(f"Created dataset repository: {hf_dataset_name}")
            except Exception as e:
                print(f"Failed to create repository: {e}")
                raise
        
        hf_initialized = True
        print(f"Streaming upload enabled: uploading in batches of {batch_size:,} chunks")
    
    # Process pages
    print(f"\nCreating chunks from {len(df):,} pages...")
    print(f"Chunk sizes: {chunk_sizes}, Overlap ratio: {overlap_ratio}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing pages"):
        title = row['title']
        content = row['content']
        page_id = row.get('page_id')
        
        # Extract sections from markdown text
        sections = _extract_sections_from_markdown(content)
        
        # If no sections found, treat entire content as one section
        if not sections:
            sections = [('', content)]
        
        # Create chunks for each chunk size
        for chunk_size, config in chunk_configs.items():
            splitter = config['splitter']
            global_chunk_idx = 0  # Track chunk index across all sections
            
            # Process each section separately
            for section_title, section_content in sections:
                # Skip empty sections
                if not section_content.strip():
                    continue
                
                # Create chunks within this section
                section_chunks = splitter.split_text(section_content)
                
                # Filter out chunks that are too short
                section_chunks = [chunk for chunk in section_chunks if len(chunk.strip()) >= min_chunk_length]
                
                # Create chunk records for this section
                for section_chunk_idx, chunk_text in enumerate(section_chunks):
                    chunk_id = f"{title.replace(' ', '_')}_chunk{global_chunk_idx}"
                    
                    chunk_record = {
                        'content': chunk_text.strip(),
                        'title': title,
                        'section': section_title,  # Section title from markdown heading
                        'chunk_id': chunk_id,
                        'chunk_index': global_chunk_idx,
                        'chunk_size': chunk_size,
                        'page_id': page_id,
                    }
                    
                    if streaming_upload:
                        # Accumulate in batch
                        config['batch_chunks'].append(chunk_record)
                        stats['chunks_by_size'][chunk_size] += 1
                        
                        # Upload batch when it reaches batch_size
                        if len(config['batch_chunks']) >= batch_size:
                            _upload_batch_to_hf(
                                chunks=config['batch_chunks'],
                                dataset_name=hf_dataset_name,
                                chunk_size=chunk_size,
                                is_first_batch=(stats['chunks_by_size'][chunk_size] == len(config['batch_chunks']))
                            )
                            config['batch_chunks'] = []  # Clear batch
                    else:
                        # Store in memory for later saving
                        config['chunks'].append(chunk_record)
                        stats['chunks_by_size'][chunk_size] += 1
                    
                    global_chunk_idx += 1
    
    # Handle remaining batches for streaming upload
    if streaming_upload and upload_to_hf:
        print("\nUploading remaining batches...")
        for chunk_size, config in chunk_configs.items():
            if config['batch_chunks']:
                _upload_batch_to_hf(
                    chunks=config['batch_chunks'],
                    dataset_name=hf_dataset_name,
                    chunk_size=chunk_size,
                    is_first_batch=False
                )
                config['batch_chunks'] = []
        
        # Merge all batches into final datasets
        print("\nMerging batches into final datasets...")
        import os
        from datasets import concatenate_datasets
        
        for chunk_size in chunk_sizes:
            config_name = f"chunks_{chunk_size}"
            if config_name not in _hf_local_batch_files or not _hf_local_batch_files[config_name]:
                continue
            
            batch_files = [f for f in _hf_local_batch_files[config_name] if os.path.exists(f)]
            if not batch_files:
                continue
            
            print(f"Merging {len(batch_files)} batches for {config_name}...")
            
            # Stream batches one at a time to minimize memory/disk usage
            all_dataframes = []
            
            for i, batch_file in enumerate(batch_files):
                batch_df = pd.read_parquet(batch_file)
                all_dataframes.append(batch_df)
                
                # Clean up immediately after loading
                os.remove(batch_file)
                
                # Periodically merge to avoid memory issues
                if len(all_dataframes) >= 10:  # Merge every 10 batches
                    merged_df = pd.concat(all_dataframes, ignore_index=True)
                    all_dataframes = [merged_df]
                    print(f"  Merged {i+1}/{len(batch_files)} batches...")
            
            # Final merge
            if all_dataframes:
                final_df = pd.concat(all_dataframes, ignore_index=True)
                final_dataset = Dataset.from_pandas(final_df)
                
                # Push final merged dataset
                final_dataset.push_to_hub(
                    repo_id=hf_dataset_name,
                    config_name=config_name,
                    private=False,
                )
                print(f"  ✓ Merged and uploaded final dataset: {len(final_dataset):,} chunks (size: {chunk_size})")
                
                # Clear memory
                del final_df, final_dataset
            
            # Clear batch files lists
            _hf_batch_files[config_name] = []
            _hf_local_batch_files[config_name] = []
    
    # Save datasets (only if not streaming)
    if not streaming_upload:
        print("\nSaving datasets...")
        for chunk_size, config in chunk_configs.items():
            if not config['chunks']:
                print(f"Warning: No chunks created for size {chunk_size}")
                continue
            
            df_chunks = pd.DataFrame(config['chunks'])
            output_file = output_dir / f"chunks_{chunk_size}.parquet"
            df_chunks.to_parquet(output_file, compression='snappy', index=False)
            print(f"Saved {len(df_chunks):,} chunks to {output_file}")
    
    # Save metadata
    metadata = {
        'statistics': {
            'total_pages': stats['total_pages'],
            'chunks_by_size': dict(stats['chunks_by_size']),
        },
        'config': {
            'chunk_sizes': chunk_sizes,
            'overlap_ratio': overlap_ratio,
            'min_chunk_length': min_chunk_length,
            'cleaned_data_path': str(cleaned_data_path),
        },
        'chunk_configs': stats['chunk_configs'],
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    # Upload to Hugging Face if requested
    if upload_to_hf and not streaming_upload:
        if not hf_dataset_name:
            raise ValueError("hf_dataset_name is required when upload_to_hf=True")
        
        print("\n" + "=" * 60)
        print("Uploading datasets to Hugging Face")
        print("=" * 60)
        
        upload_datasets_to_hf(
            output_dir=output_dir,
            dataset_name=hf_dataset_name,
            chunk_sizes=chunk_sizes,
            metadata=metadata,
            version=hf_dataset_version,
        )
    
    return stats

