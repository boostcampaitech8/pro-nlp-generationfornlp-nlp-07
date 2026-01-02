"""마크다운 문서 청킹 모듈 (LangChain 기반)"""

from typing import List, Tuple
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)


def chunk_markdown_document(
    context: str,
    title: str,
    page_id: str,
    headers_to_split_on: List[Tuple[str, str]] = None,
    chunk_size: int = 400,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    마크다운 문서를 청킹하고 원본 문서 정보(title, page_id)를 메타데이터에 추가
    512 토큰 제한을 준수하도록 청킹 크기 조정
    
    Args:
        context: 마크다운 형식의 텍스트
        title: 문서 제목
        page_id: 페이지 ID
        headers_to_split_on: 헤더 레벨 설정 (기본값: [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")])
        chunk_size: 최대 청크 길이 (기본값: 400자, 512 토큰 제한 고려)
        chunk_overlap: 청크 오버랩 (기본값: 50자)
    
    Returns:
        청킹된 Document 리스트 (각각에 title, page_id, Header 정보 포함)
    """
    if headers_to_split_on is None:
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    
    # 빈 context 처리
    if not context or not context.strip():
        logger.warning(f"Empty context for title: {title}, page_id: {page_id}")
        return []
    
    # 1. 첫 문단 헤더 처리
    if not context.strip().startswith("#"):
        context = "# Introduction\n" + context
    
    # 2. MarkdownHeaderTextSplitter로 섹션별 분할
    try:
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_splits = markdown_splitter.split_text(context)
    except Exception as e:
        logger.error(f"Error splitting markdown for title: {title}, page_id: {page_id}: {e}")
        return []
    
    # 3. 각 청크에 원본 문서 정보 추가
    for split in md_splits:
        split.metadata["title"] = title if title else ""
        split.metadata["page_id"] = page_id if page_id else ""
    
    # 4. 긴 섹션은 RecursiveCharacterTextSplitter로 2차 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    try:
        final_splits = text_splitter.split_documents(md_splits)
    except Exception as e:
        logger.error(f"Error in recursive splitting for title: {title}, page_id: {page_id}: {e}")
        final_splits = md_splits
    
    # 5. 2차 분할된 청크에도 원본 정보 복사 (누락된 경우)
    for split in final_splits:
        if "title" not in split.metadata or not split.metadata["title"]:
            split.metadata["title"] = title if title else ""
        if "page_id" not in split.metadata or not split.metadata["page_id"]:
            split.metadata["page_id"] = page_id if page_id else ""
        
        # 청크 길이 검증
        if len(split.page_content) > chunk_size:
            logger.warning(
                f"Chunk exceeds chunk_size ({chunk_size}): "
                f"title={title}, page_id={page_id}, length={len(split.page_content)}"
            )
    
    return final_splits


def prepare_embedding_text(chunk: Document) -> str:
    """
    청크를 임베딩용 텍스트로 변환 (title + 헤더 경로 포함)
    512 토큰 제한을 고려하여 전체 길이를 400자 이내로 유지
    
    Args:
        chunk: LangChain Document 객체
    
    Returns:
        임베딩용 텍스트 (형식: "title > Header1 > Header2: {내용}")
    """
    # 헤더 경로 생성
    header_path = []
    for level in ["Header 1", "Header 2", "Header 3"]:
        if level in chunk.metadata and chunk.metadata[level]:
            header_path.append(chunk.metadata[level])
    
    title = chunk.metadata.get("title", "")
    content = chunk.page_content
    
    # prefix 생성
    if header_path:
        path_str = " > ".join(header_path)
        prefix = f"{title} > {path_str}: " if title else f"{path_str}: "
    else:
        prefix = f"{title}: " if title else ""
    
    # prefix 길이를 고려하여 content 조정 (최대 400자)
    max_content_len = 400 - len(prefix)
    if max_content_len <= 0:
        # prefix가 너무 긴 경우 축약
        if len(prefix) > 100:
            prefix = prefix[:100] + "...: "
            max_content_len = 400 - len(prefix)
        else:
            max_content_len = 300  # 최소한의 content 길이 보장
    
    if len(content) > max_content_len:
        content = content[:max_content_len]
    
    return prefix + content


def chunk_dataset(
    dataset,
    context_column: str = "context",
    title_column: str = "title",
    page_id_column: str = "page_id",
    headers_to_split_on: List[Tuple[str, str]] = None,
    chunk_size: int = 400,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    데이터셋 전체를 청킹
    
    Args:
        dataset: Hugging Face Dataset 객체
        context_column: context 컬럼명
        title_column: title 컬럼명
        page_id_column: page_id 컬럼명
        headers_to_split_on: 헤더 레벨 설정
        chunk_size: 최대 청크 길이
        chunk_overlap: 청크 오버랩
    
    Returns:
        모든 문서의 청킹된 Document 리스트
    """
    all_chunks = []
    
    for idx, row in enumerate(dataset):
        context = row.get(context_column, "")
        title = row.get(title_column, "")
        page_id = row.get(page_id_column, "")
        
        chunks = chunk_markdown_document(
            context=context,
            title=title,
            page_id=page_id,
            headers_to_split_on=headers_to_split_on,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        all_chunks.extend(chunks)
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1} documents, total chunks: {len(all_chunks)}")
    
    logger.info(f"Total documents processed: {len(dataset)}, total chunks: {len(all_chunks)}")
    return all_chunks


