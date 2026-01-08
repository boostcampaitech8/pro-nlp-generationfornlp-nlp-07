"""Agentic QA 시스템 구축 함수"""
import torch
from typing import Dict, Any
from unsloth import FastLanguageModel

from components import (
    MultipleChoiceLLM, FAISSRetriever, ParagraphAnalyzer, QueryGenerator,
    SearchResultFilter, IntegratedReasoner, ChoiceValidator, ChoiceEvaluator,
    FinalReasoner
)
from agentic_rag_chain import AgenticRAGChain
from utils.memory import cleanup_memory
from utils.langsmith_setup import setup_langsmith
from load_faiss_index import load_hf_faiss_index


def build_agentic_qa_system(
    model_name: str = "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    faiss_repo_id: str = "NLP-07-ODQA/kowiki-faiss-index",
    local_faiss_dir: str = "./downloaded_faiss",
    enable_tracing: bool = True,
    use_mmap: bool = True
) -> Dict[str, Any]:
    """Agentic QA 시스템 구축 (선택지별 RAG + 2단계 판단)"""
    print("=" * 60)
    print("Agentic RAG 시스템 구축 시작 (선택지별 RAG + 2단계 판단)")
    print("=" * 60)
    
    # 1. FAISS 인덱스 로드 (mmap 사용)
    print("\n[1/6] FAISS 인덱스 로딩...")
    index, metadata_list, embedding_model = load_hf_faiss_index(
        repo_id=faiss_repo_id,
        local_dir=local_faiss_dir,
        use_mmap=use_mmap
    )
    cleanup_memory()
    
    # 2. LLM 모델 로드
    print("\n[2/6] LLM 모델 로딩...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=8192,
        dtype=torch.float16,
        load_in_4bit=True
    )
    
    # 추론 모드 활성화
    FastLanguageModel.for_inference(model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    llm = MultipleChoiceLLM(model, tokenizer)
    cleanup_memory()
    
    # 3. LangSmith 설정
    print("\n[3/6] LangSmith 설정...")
    if enable_tracing:
        tracer, callbacks, langsmith_client = setup_langsmith()
    else:
        tracer, callbacks, langsmith_client = None, None, None
    
    # 4. 컴포넌트 초기화
    print("\n[4/6] 컴포넌트 초기화...")
    retriever = FAISSRetriever(index, metadata_list, embedding_model)
    paragraph_analyzer = ParagraphAnalyzer(llm)
    query_generator = QueryGenerator(llm)
    search_filter = SearchResultFilter()
    integrated_reasoner = IntegratedReasoner(llm)
    choice_validator = ChoiceValidator()
    choice_evaluator = ChoiceEvaluator(llm)
    final_reasoner = FinalReasoner(llm)
    
    # 5. Agentic RAG Chain 구축
    print("\n[5/6] Agentic RAG Chain 구축...")
    agentic_chain = AgenticRAGChain(
        paragraph_analyzer=paragraph_analyzer,
        query_generator=query_generator,
        retriever=retriever,
        search_filter=search_filter,
        integrated_reasoner=integrated_reasoner,
        choice_validator=choice_validator,
        choice_evaluator=choice_evaluator,
        final_reasoner=final_reasoner,
        llm=llm,
        tracer=tracer,
        langsmith_client=langsmith_client
    )
    
    print("\n✅ Agentic RAG 시스템 구축 완료!")
    
    return {
        "chain": agentic_chain,
        "llm": llm,
        "retriever": retriever,
        "callbacks": callbacks
    }

