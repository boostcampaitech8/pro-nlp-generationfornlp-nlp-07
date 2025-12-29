"""Wikipedia text cleaning utilities for RAG dataset creation"""

import re
import regex
from typing import List, Tuple
import html2text


def clean_wikitext(text: str, debug: bool = False) -> str:
    """
    Clean Wikipedia wikitext by removing markup and converting to markdown format
    
    Args:
        text: Raw wikitext content
        debug: If True, print text length at each step
        
    Returns:
        Cleaned text in markdown format ready for chunking
    """
    if not text:
        return ""
    
    if debug:
        print(f"[DEBUG] 초기 텍스트 길이: {len(text):,}자")
    
    # Remove redirect markers
    if text.strip().startswith('#넘겨주기') or text.strip().startswith('#REDIRECT'):
        if debug:
            print("[DEBUG] 리다이렉트 페이지로 감지됨")
        return ""
    
    # Remove unnecessary sections (같이 보기, 외부 링크, 참고 문헌, 각주, 주석 등)
    # 주의: 섹션 제목만 매칭하고, 다음 섹션 제목이나 문서 끝까지 제거
    sections_to_remove = r'같이\s+보기|외부\s+링크|참고\s+문헌|각주|주석|참고\s+사항|출처|바깥\s+고리'
    # 섹션 제목부터 다음 섹션 제목(==) 또는 문서 끝까지 제거
    text = re.sub(r'==\s*(' + sections_to_remove + r')\s*==.*?(?=\n==|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
    if debug:
        print(f"[DEBUG] 섹션 제거 후: {len(text):,}자")
    
    # Step 1: Remove HTML tags using html2text FIRST
    # html2text converts HTML to plain text, removing all HTML tags
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = False
    h.body_width = 0  # Don't wrap lines
    text = h.handle(text)
    if debug:
        print(f"[DEBUG] html2text 처리 후: {len(text):,}자")
    
    # Remove remaining HTML tags
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    if debug:
        print(f"[DEBUG] HTML 태그 제거 후: {len(text):,}자")
    
    # Remove special namespace links (파일, 분류 등)
    # 파일 링크 내부에 내부 링크가 있는 경우 그 텍스트를 추출
    def _process_file_link(match):
        link_content = match.group(0)  # 전체 링크: [[파일:...|...|[[주판]]...]]
        # 내부 링크 찾기: [[...]] (첫 번째 내부 링크)
        # 파일: 부분 이후에서 내부 링크 찾기
        inner_link_match = re.search(r'\[\[([^\]]+)\]\]', link_content)
        if inner_link_match:
            # 내부 링크에서 텍스트 추출
            return _extract_link_text(inner_link_match.group(1))
        return ""
    
    # 파일 링크 처리 (중첩된 링크를 처리하기 위해 반복)
    # 단계적으로 처리: 먼저 내부 링크가 없는 경우, 그 다음 내부 링크가 있는 경우
    prev_text = ""
    max_iterations = 10
    iteration = 0
    while prev_text != text and iteration < max_iterations:
        prev_text = text
        iteration += 1
        # 파일 링크 패턴: [[파일:...|...|[[내부링크]]...]]
        # 내부 링크가 있는 경우도 처리 (더 넓은 패턴)
        text = re.sub(r'\[\[(?:파일|File|Image|분류):[^\]]*(?:\[\[[^\]]+\]\][^\]]*)*\]\]', _process_file_link, text, flags=re.IGNORECASE)
    if debug:
        print(f"[DEBUG] 파일 링크 처리 후: {len(text):,}자")
    
    # Remove internal links but keep text: [[텍스트|링크]] -> 텍스트, [[텍스트]] -> 텍스트
    text = re.sub(r'\[\[([^\]]+)\]\]', lambda m: _extract_link_text(m.group(1)), text)
    if debug:
        print(f"[DEBUG] 내부 링크 처리 후: {len(text):,}자")

    # External links: [URL 텍스트] -> 텍스트 (remove link, keep text only)
    text = re.sub(r'\[https?://[^\s]+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)  # Remove links without text
    if debug:
        print(f"[DEBUG] 외부 링크 처리 후: {len(text):,}자")
    
    # Extract content from templates: {{템플릿|파라미터|...}} -> 파라미터
    # 재귀적 패턴으로 중첩된 {{ }} 처리
    # (?R)은 재귀적으로 패턴을 매칭하여 중첩된 중괄호를 처리
    text = regex.sub(r'\{\{((?:[^{}]++|\{(?:[^{}]++|\{[^{}]*+\})*+\}|(?R))*+)\}\}', 
                lambda m: _extract_template_content(m.group(1), debug), text)
    if debug:
        print(f"[DEBUG] 템플릿 제거 후 : {len(text):,}자")
    
    # Remove template parameters and style attributes
    # 주의: 줄 전체가 파이프 패턴인 경우만 제거 (본문 보존)
    text = re.sub(r'^\|[^|\n]+\|$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\|[^|\n]+=.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?:style|display)\s*[:=]\s*[^|\n}]+', '', text, flags=re.IGNORECASE)
    if debug:
        print(f"[DEBUG] 템플릿 파라미터 제거 후: {len(text):,}자")
    
    # Remove image patterns (sizes and file names)
    image_ext = r'\.(jpg|jpeg|png|gif|svg|webp|bmp|tiff|tif|ico|jfif|heic|heif)'
    text = re.sub(r'\d+x?\d*px', '', text)
    # 이미지 파일명만 제거 (공백이나 특수문자로 구분된 경우만)
    # 예: "file.jpg" 또는 "file.jpg|thumb" 형태만 제거
    text = re.sub(r'(?:^|\s)([^\s\[\]{}|<>]+' + image_ext + r')(?:\|[^\s\[\]{}|<>]*)?(?=\s|$)', '', text, flags=re.IGNORECASE | re.MULTILINE)
    if debug:
        print(f"[DEBUG] 이미지 패턴 제거 후: {len(text):,}자")
    
    # Remove remaining braces and patterns
    text = re.sub(r'\{\{+|\}\}+', '', text)
    text = re.sub(r'더\s+보기\.\.\.', '', text)
    if debug:
        print(f"[DEBUG] 중괄호 제거 후: {len(text):,}자")
    
    
    # Convert to markdown format
    # Headings: ==제목== -> # 제목, ===제목=== -> ## 제목 (with newlines before and after)
    # 연속된 "="의 개수 -1만큼 "#"을 붙인다
    # Process headings from level 6 down to level 2 to avoid conflicts
    for level in range(6, 1, -1):
        equals = '=' * level
        def _convert_heading(match):
            heading_text = match.group(1).strip()
            # Markdown level = wiki level - 1
            markdown_level = level - 1  # e.g., 2 -> 1 (#), 3 -> 2 (##)
            return '\n' + '#' * min(markdown_level, 6) + ' ' + heading_text + '\n'
        # Match with optional spaces: == 제목 == or ==제목==
        pattern = re.escape(equals) + r'\s*([^=\n]+?)\s*' + re.escape(equals)
        text = re.sub(pattern, _convert_heading, text)
    if debug:
        print(f"[DEBUG] 제목 변환 후: {len(text):,}자")
    
    # Bold/italic: '''굵게''' -> **굵게**, ''기울임'' -> *기울임*
    text = re.sub(r"'''(.+?)'''", r'**\1**', text)
    text = re.sub(r"''(.+?)''", r'*\1*', text)
    if debug:
        print(f"[DEBUG] 볼드/이탤릭 변환 후: {len(text):,}자")
    
    # Remove tables: {|...|}
    # 테이블은 줄 시작에 {|가 있는 경우만 제거
    text = re.sub(r'^\{\|.*?\|\}$', '', text, flags=re.MULTILINE | re.DOTALL)
    if debug:
        print(f"[DEBUG] 테이블 제거 후: {len(text):,}자")
    
    # Final cleanup: whitespace and formatting
    text = text.replace('\t', ' ')
    # 각 줄의 앞뒤 공백 제거 (빈 줄은 유지)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single
    text = text.strip()
    if debug:
        print(f"[DEBUG] 최종 정리 후: {len(text):,}자")
    
    return text


def _extract_link_text(link_content: str) -> str:
    """Extract text from wiki link: [[링크|텍스트]] -> 텍스트 or [[링크]] -> 링크 """
    if '|' in link_content:
        return link_content.split('|')[-1]
    return link_content


def _extract_template_content(template_content: str, debug: bool = False) -> str:
    """
    Extract content from wiki template: {{템플릿|내용}} -> 내용
    
    Examples:
        {{수학 변수|eπ}} -> eπ
        {{수학|1=''e'' + ''π''}} -> ''e'' + ''π''
        {{템플릿명|파라미터1|파라미터2}} -> 파라미터2 (마지막 파라미터)
    """
    if not template_content:
        return ""
    
    if debug:
        print("--------------------------------")
        print("[DEBUG] template_content : ", template_content)
        print("--------------------------------")

    # 파이프(|)로 분리
    parts = template_content.split('|')
    template_name = parts[0].strip() if parts else ""
    
    # Image frame, Image, File 등의 이미지/파일 관련 템플릿은 완전히 제거
    template_names_voc = [
        'Image frame', 'Image', 'File', '파일', '이미지', '그림',
        'ISBN', 'Music', '인용', '웹', '참고']
    if any(template_name.lower() in name.lower() for name in template_names_voc):
        return ""
    
    if len(parts) == 1:
        # 파이프가 없으면 템플릿 이름만 반환
        return parts[0].strip()
    
    # 마지막 파라미터 반환
    return parts[-1].strip()