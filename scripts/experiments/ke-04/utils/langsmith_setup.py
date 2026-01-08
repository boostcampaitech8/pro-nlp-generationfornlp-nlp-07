"""LangSmith 설정 유틸리티"""
import os
import logging
from langchain_core.tracers import LangChainTracer

# LangSmith SDK for custom tracing
try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

# LangSmith SDK 디버깅 활성화
os.environ.setdefault("LANGSMITH_DEBUG", "true")

# 로깅 레벨 설정 (LangSmith SDK 내부 로그 확인용)
logging.getLogger("langsmith").setLevel(logging.DEBUG)


def setup_langsmith():
    """LangSmith Tracer 초기화"""
    # LangChain이 환경 변수에서 자동으로 읽음
    # LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT
    try:
        tracer = LangChainTracer()
        callbacks = [tracer]
        
        # LangSmith Client 초기화 (커스텀 이벤트 로깅용)
        client = None
        if LANGSMITH_AVAILABLE:
            try:
                # 환경 변수 확인
                api_key = os.getenv("LANGCHAIN_API_KEY")
                tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "").lower()
                project = os.getenv("LANGCHAIN_PROJECT", "default")
                
                print(f"🔍 LangSmith 설정 확인:")
                print(f"   LANGCHAIN_API_KEY: {'설정됨' if api_key else '❌ 설정되지 않음'}")
                print(f"   LANGCHAIN_TRACING_V2: {tracing_v2}")
                print(f"   LANGCHAIN_PROJECT: {project}")
                
                if not api_key:
                    print("⚠️ LANGCHAIN_API_KEY가 설정되지 않았습니다. LangSmith 트래킹이 비활성화됩니다.")
                    return tracer, callbacks, None
                
                if tracing_v2 not in ("true", "1", "yes"):
                    print("⚠️ LANGCHAIN_TRACING_V2가 활성화되지 않았습니다. LangSmith 트래킹이 비활성화됩니다.")
                    return tracer, callbacks, None
                
                # LangSmith Client 초기화 (디버깅 활성화)
                # debug=True로 설정하면 더 자세한 로그 출력
                client = Client()
                print(f"✅ LangSmith Client 초기화 성공")
                
            except Exception as e:
                print(f"⚠️ LangSmith Client 초기화 실패: {e}")
                print(f"   예외 타입: {type(e).__name__}")
                import traceback
                print(f"   상세 에러:\n{traceback.format_exc()}")
                client = None
        
        return tracer, callbacks, client
    except Exception as e:
        print(f"⚠️ LangSmith 초기화 실패: {e}")
        print("   LangSmith 트래킹 없이 진행합니다.")
        import traceback
        print(f"   상세 에러:\n{traceback.format_exc()}")
        return None, None, None

