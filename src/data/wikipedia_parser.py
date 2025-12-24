"""Wikipedia XML dump parser utilities"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Iterator, Tuple
from collections import Counter, defaultdict
import re
from pathlib import Path
from tqdm import tqdm


def parse_wikipedia_xml_stream(
    xml_path: str,
    max_pages: Optional[int] = None,
    sample_rate: float = 1.0
) -> Iterator[Dict]:
    """
    Stream parse Wikipedia XML dump file
    
    Args:
        xml_path: Path to XML file
        max_pages: Maximum number of pages to parse (None for all)
        sample_rate: Sampling rate (0.0 to 1.0)
        
    Yields:
        Dictionary containing page information
    """
    import random
    
    page_count = 0
    
    # Use iterparse for memory-efficient parsing
    print(f"XML 파일 열기 중... (파일 크기: {Path(xml_path).stat().st_size / (1024**3):.2f} GB)")
    context = ET.iterparse(xml_path, events=('start', 'end'))
    context = iter(context)
    print("XML 파서 초기화 중...")
    event, root = next(context)
    print("파싱 시작...")
    
    current_page = {}
    
    for event, elem in context:
        if event == 'end':
            if elem.tag == '{http://www.mediawiki.org/xml/export-0.11/}title':
                current_page['title'] = elem.text or ''
            elif elem.tag == '{http://www.mediawiki.org/xml/export-0.11/}ns':
                current_page['ns'] = int(elem.text) if elem.text else 0
            elif elem.tag == '{http://www.mediawiki.org/xml/export-0.11/}id':
                if 'page_id' not in current_page:
                    current_page['page_id'] = int(elem.text) if elem.text else None
            elif elem.tag == '{http://www.mediawiki.org/xml/export-0.11/}text':
                current_page['text'] = elem.text or ''
                current_page['text_bytes'] = int(elem.get('bytes', 0)) if elem.get('bytes') else len(elem.text or '')
            elif elem.tag == '{http://www.mediawiki.org/xml/export-0.11/}timestamp':
                current_page['timestamp'] = elem.text or ''
            elif elem.tag == '{http://www.mediawiki.org/xml/export-0.11/}page':
                # Page complete
                if current_page and random.random() <= sample_rate:
                    yield current_page.copy()
                    page_count += 1
                    if max_pages and page_count >= max_pages:
                        break
                current_page = {}
                # Clear element to free memory
                root.clear()
    
    # Clean up
    root.clear()


def analyze_wikitext_tags(text: str) -> Dict[str, int]:
    """
    Analyze wikitext tags in page text
    
    Args:
        text: Wikitext content
        
    Returns:
        Dictionary with tag counts
    """
    if not text:
        return {}
    
    patterns = {
        'internal_links': r'\[\[([^\]]+)\]\]',
        'external_links': r'\[https?://[^\s]+\s+([^\]]+)\]',
        'templates': r'\{\{([^}]+)\}\}',
        'categories': r'\[\[분류:([^\]]+)\]\]',
        'files': r'\[\[파일:([^\]]+)\]\]',
        'headings': r'^={1,6}(.+?)={1,6}$',
        'bold': r"'''(.+?)'''",
        'italic': r"''(.+?)''",
        'references': r'<ref[^>]*>.*?</ref>',
        'tables': r'\{\|.*?\|\}',
    }
    
    counts = {}
    for tag_name, pattern in patterns.items():
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        counts[tag_name] = len(matches)
    
    return counts


def analyze_language_content(text: str) -> Dict[str, int]:
    """
    Analyze language content (Korean, English, numbers, symbols)
    
    Args:
        text: Text content
        
    Returns:
        Dictionary with character type counts
    """
    if not text:
        return {
            'korean': 0,
            'english': 0,
            'numbers': 0,
            'symbols': 0,
            'spaces': 0,
            'total': 0
        }
    
    korean = len(re.findall(r'[가-힣]', text))
    english = len(re.findall(r'[a-zA-Z]', text))
    numbers = len(re.findall(r'[0-9]', text))
    spaces = len(re.findall(r'\s', text))
    total = len(text)
    symbols = total - korean - english - numbers - spaces
    
    return {
        'korean': korean,
        'english': english,
        'numbers': numbers,
        'symbols': symbols,
        'spaces': spaces,
        'total': total
    }


def is_redirect(text: str) -> bool:
    """Check if page is a redirect"""
    if not text:
        return False
    return text.strip().startswith('#넘겨주기') or text.strip().startswith('#REDIRECT')


def is_stub(text: str) -> bool:
    """Check if page is a stub (very short)"""
    if not text:
        return True
    # Remove wikitext markup for length check
    clean_text = re.sub(r'\[\[[^\]]+\]\]', '', text)
    clean_text = re.sub(r'\{\{[^}]+\}\}', '', clean_text)
    return len(clean_text.strip()) < 100


def count_words(text: str) -> int:
    """Count words in text (Korean and English)"""
    if not text:
        return 0
    # Remove wikitext markup
    clean_text = re.sub(r'\[\[[^\]]+\]\]', ' ', text)
    clean_text = re.sub(r'\{\{[^}]+\}\}', ' ', clean_text)
    clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
    # Split by whitespace
    words = clean_text.split()
    return len(words)


def get_namespace_name(ns: int) -> str:
    """Get namespace name from namespace number"""
    namespace_map = {
        0: '문서',
        1: '토론',
        2: '사용자',
        3: '사용자토론',
        4: '위키백과',
        5: '위키백과토론',
        6: '파일',
        7: '파일토론',
        10: '틀',
        11: '틀토론',
        12: '도움말',
        13: '도움말토론',
        14: '분류',
        15: '분류토론',
        100: '포털',
        101: '포털토론',
        102: '위키프로젝트',
        103: '위키프로젝트토론',
    }
    return namespace_map.get(ns, f'기타({ns})')


def collect_statistics(
    xml_path: str,
    max_pages: Optional[int] = None,
    sample_rate: float = 1.0
) -> Dict:
    """
    Collect comprehensive statistics from Wikipedia XML dump
    
    Args:
        xml_path: Path to XML file
        max_pages: Maximum number of pages to analyze
        sample_rate: Sampling rate (0.0 to 1.0)
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_pages': 0,
        'namespace_counts': Counter(),
        'redirect_count': 0,
        'stub_count': 0,
        'empty_count': 0,
        'text_lengths': [],
        'word_counts': [],
        'title_lengths': [],
        'tag_counts': defaultdict(int),
        'language_stats': {
            'korean': 0,
            'english': 0,
            'numbers': 0,
            'symbols': 0,
            'total': 0
        },
        'sample_pages': [],
    }
    
    
    for page in tqdm(parse_wikipedia_xml_stream(xml_path, max_pages, sample_rate), 
                desc="파싱 중..."):
        stats['total_pages'] += 1
        
        # Namespace
        ns = page.get('ns', 0)
        stats['namespace_counts'][ns] += 1
        
        # Title length
        title = page.get('title', '')
        stats['title_lengths'].append(len(title))
        
        # Text analysis
        text = page.get('text', '')
        
        if not text or len(text.strip()) == 0:
            stats['empty_count'] += 1
            continue 
        
        # Redirect check
        if is_redirect(text):
            stats['redirect_count'] += 1
        
        # Stub check
        if is_stub(text):
            stats['stub_count'] += 1
        
        # Text length
        text_len = len(text)
        stats['text_lengths'].append(text_len)
        
        # Word count
        word_count = count_words(text)
        stats['word_counts'].append(word_count)
        
        # Wikitext tags
        tag_counts = analyze_wikitext_tags(text)
        for tag, count in tag_counts.items():
            stats['tag_counts'][tag] += count
        
        # Language analysis
        lang_stats = analyze_language_content(text)
        for key in stats['language_stats']:
            if key in lang_stats:
                stats['language_stats'][key] += lang_stats[key]
        
        # Collect samples (first 10 non-empty pages)
        if len(stats['sample_pages']) < 10 and not is_redirect(text):
            stats['sample_pages'].append({
                'title': title,
                'ns': ns,
                'text_preview': text[:500] + '...' if len(text) > 500 else text,
                'text_length': len(text),
            })
    
    return stats

