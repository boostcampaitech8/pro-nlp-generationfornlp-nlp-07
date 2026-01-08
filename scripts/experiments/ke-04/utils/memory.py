"""메모리 최적화 유틸리티"""
import gc
import torch
from typing import Optional, Dict


def cleanup_memory(aggressive: bool = False):
    """메모리 정리 함수"""
    # Python 가비지 컬렉션
    gc.collect()
    
    # GPU 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            # 더 공격적인 정리 (느릴 수 있음)
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()


def get_memory_usage() -> Optional[Dict[str, float]]:
    """현재 메모리 사용량 확인"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return {
            "allocated": allocated,
            "reserved": reserved,
            "max_allocated": max_allocated
        }
    return None

