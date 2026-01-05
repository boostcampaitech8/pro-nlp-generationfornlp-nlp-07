"""
Agentic RAG 핵심 로직
- Thinking 모드 완전히 비활성화
- 메모리 최적화 전략 적용
- LangSmith 트래킹
"""
import os
import gc
import torch
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.tracers import LangChainTracer
from pydantic import ConfigDict, Field

from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

from load_faiss_index import load_hf_faiss_index

# 환경 변수 로드
load_dotenv()

# LangSmith 설정 (LangChain이 자동으로 읽음)
# LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT는 .env 파일에 설정


# ========== 메모리 최적화 유틸리티 ==========

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


# ========== LangSmith 설정 ==========

def setup_langsmith():
    """LangSmith Tracer 초기화"""
    # LangChain이 환경 변수에서 자동으로 읽음
    # LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT
    try:
        tracer = LangChainTracer()
        callbacks = [tracer]
        return tracer, callbacks
    except Exception as e:
        print(f"⚠️ LangSmith 초기화 실패: {e}")
        print("   LangSmith 트래킹 없이 진행합니다.")
        return None, None


# ========== FAISS Retriever (메모리 최적화) ==========

class FAISSRetriever(BaseRetriever):
    """HuggingFace FAISS 인덱스를 사용하는 Retriever (메모리 최적화)"""
    
    index: Any = Field(exclude=True)
    metadata_list: List[Dict[str, Any]] = Field(exclude=True)
    embedding_model: SentenceTransformer = Field(exclude=True)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, index, metadata_list, embedding_model):
        super().__init__(index=index, metadata_list=metadata_list, embedding_model=embedding_model)
        # Pydantic 필드가 아닌 일반 인스턴스 변수로 정의 (언더스코어 사용 가능)
        self._model_on_gpu = False
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """검색 수행 (동적 GPU 로딩)"""
        # 임베딩 모델을 GPU로 일시 이동
        if not self._model_on_gpu:
            if hasattr(self.embedding_model, 'model'):
                self.embedding_model.model = self.embedding_model.model.to('cuda')
            else:
                # SentenceTransformer의 경우
                for module in self.embedding_model.modules():
                    if hasattr(module, 'to'):
                        module.to('cuda')
            self._model_on_gpu = True
        
        try:
            # 검색 수행
            query_text = f"query: {query}"  # E5 모델 필수 접두사
            query_embedding = self.embedding_model.encode(
                [query_text],
                normalize_embeddings=True,  # 코사인 유사도 인덱스 사용했으니 필수
                convert_to_numpy=True
            )
            
            # FAISS 검색
            distances, indices = self.index.search(
                query_embedding.astype('float32'), 
                k=10  # 기본값
            )
            
            # 결과 생성
            docs = []
            for idx in indices[0]:
                if idx < len(self.metadata_list):
                    metadata = self.metadata_list[idx]
                    docs.append(Document(
                        page_content=metadata.get('page_content', ''),
                        metadata={
                            'title': metadata.get('title', 'N/A'),
                            'page_id': metadata.get('page_id', ''),
                            'index': int(idx)
                        }
                    ))
            
            # 임베딩 벡터 즉시 삭제
            del query_embedding, distances, indices
            
        finally:
            # 임베딩 모델을 CPU로 다시 이동 (메모리 절약)
            if self._model_on_gpu:
                if hasattr(self.embedding_model, 'model'):
                    self.embedding_model.model = self.embedding_model.model.to('cpu')
                else:
                    for module in self.embedding_model.modules():
                        if hasattr(module, 'to'):
                            module.to('cpu')
                self._model_on_gpu = False
                cleanup_memory()
        
        return docs


# ========== LLM Wrapper (Thinking 모드 비활성화) ==========

class MultipleChoiceLLM:
    """객관식 문제를 위한 LLM 래퍼 (Thinking 모드 비활성화)"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.last_probs = None
        self.last_answer = None
        self.last_confidence = None
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0
    ) -> str:
        """생성 수행 (Thinking 모드 비활성화)"""
        # Qwen3 모델의 경우 enable_thinking=False 설정
        messages = [{"role": "user", "content": prompt}]
        
        # chat_template 적용 (enable_thinking=False)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Thinking 모드 비활성화
        )
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 디코딩
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        # 메모리 정리
        del inputs, outputs
        cleanup_memory()
        
        return response.strip()
    
    def predict_choice(
        self,
        prompt: str,
        num_choices: int = 5
    ) -> Dict[str, Any]:
        """선택지 예측 (logits 기반)"""
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            
            # 선택지 토큰 ID 추출
            choice_tokens = [str(i) for i in range(1, num_choices + 1)]
            choice_token_ids = []
            for choice in choice_tokens:
                token_id = self.tokenizer.encode(choice, add_special_tokens=False)
                if token_id:
                    choice_token_ids.append(token_id[0])
                else:
                    # 폴백: vocab에서 직접 찾기
                    vocab = self.tokenizer.get_vocab()
                    if choice in vocab:
                        choice_token_ids.append(vocab[choice])
            
            if not choice_token_ids:
                # 최종 폴백: 1~5를 직접 인코딩
                choice_token_ids = [
                    self.tokenizer.encode(str(i), add_special_tokens=False)[0]
                    for i in range(1, num_choices + 1)
                ]
            
            # Logits 추출 및 확률 계산
            choice_logits = logits[0, choice_token_ids]
            choice_probs = torch.nn.functional.softmax(choice_logits, dim=-1)
            
            self.last_probs = {
                choice: prob.item() 
                for choice, prob in zip(choice_tokens, choice_probs)
            }
            
            predicted_idx = torch.argmax(choice_probs).item()
            self.last_answer = choice_tokens[predicted_idx]
            self.last_confidence = choice_probs[predicted_idx].item()
        
        # 메모리 정리
        del inputs, outputs, logits
        cleanup_memory()
        
        return {
            'probs': self.last_probs,
            'answer': self.last_answer,
            'confidence': self.last_confidence
        }


# ========== 1단계: 지문 분석 ==========

class ParagraphAnalyzer:
    """지문만으로 답을 구할 수 있는지 분석"""
    
    def __init__(self, llm: MultipleChoiceLLM):
        self.llm = llm
    
    def analyze(
        self,
        paragraph: str,
        question: str,
        choices: List[str],
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """지문 분석"""
        prompt = f"""다음 지문과 질문을 읽고, 지문만으로 답을 구할 수 있는지 판단하세요.

지문:
{paragraph}

질문:
{question}

선택지:
{chr(10).join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])}

지문만으로 답을 구할 수 있으면 "예"를, 배경지식이 필요하면 "아니오"를 답하세요.
답변 형식: "예" 또는 "아니오"로만 답하세요."""
        
        response = self.llm.generate(prompt, max_new_tokens=100)
        
        can_answer = "예" in response or "yes" in response.lower() or "가능" in response
        
        return {
            'can_answer_without_search': can_answer,
            'reasoning': response
        }


# ========== 2단계: 검색 쿼리 생성 ==========

class QueryGenerator:
    """검색 쿼리 생성"""
    
    def __init__(self, llm: MultipleChoiceLLM):
        self.llm = llm
    
    def generate_query(
        self,
        paragraph: str,
        question: str,
        choices: List[str],
        previous_queries: Optional[List[str]] = None,
        callbacks: Optional[List] = None
    ) -> str:
        """검색 쿼리 생성"""
        previous_info = ""
        if previous_queries:
            previous_info = f"\n\n이전 검색 쿼리:\n{chr(10).join([f'- {q}' for q in previous_queries])}\n\n위 쿼리로 충분한 정보를 얻지 못했습니다. 다른 관점에서 쿼리를 생성하세요."
        
        prompt = f"""다음 지문과 질문을 읽고, 위키피디아에서 검색할 쿼리를 생성하세요.

지문:
{paragraph[:500]}  # 일부만 표시

질문:
{question}

선택지:
{chr(10).join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])}
{previous_info}

위키피디아에서 검색할 키워드나 질문을 한 문장으로 작성하세요.
검색 쿼리만 답하세요 (설명 없이)."""
        
        query = self.llm.generate(prompt, max_new_tokens=50)
        return query.strip()


# ========== 검색 결과 필터링 ==========

class SearchResultFilter:
    """검색 결과 필터링 및 리랭킹"""
    
    def filter_and_rerank(
        self,
        docs: List[Document],
        query: str,
        top_k: int = 5,
        callbacks: Optional[List] = None
    ) -> List[Document]:
        """검색 결과 필터링 및 리랭킹"""
        # 간단한 필터링: 관련성 점수 기반 (거리 기반)
        # 실제로는 Cross-encoder 리랭커를 사용할 수 있지만 메모리 고려하여 생략
        
        # 문서 길이 기반 필터링 (너무 짧거나 긴 문서 제외)
        filtered_docs = []
        for doc in docs:
            content_len = len(doc.page_content)
            if 100 <= content_len <= 5000:  # 적절한 길이
                filtered_docs.append(doc)
        
        # 상위 k개만 반환
        return filtered_docs[:top_k]


# ========== 3단계: 통합 추론 ==========

class IntegratedReasoner:
    """지문 + 위키 검색 결과 통합 추론"""
    
    def __init__(self, llm: MultipleChoiceLLM):
        self.llm = llm
    
    def reason(
        self,
        paragraph: str,
        question: str,
        choices: List[str],
        search_results: List[Document],
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """통합 추론"""
        # 검색 결과 포맷팅
        rag_context = "\n\n".join([
            f"[{i+1}] {doc.metadata.get('title', 'N/A')}\n{doc.page_content[:500]}"
            for i, doc in enumerate(search_results)
        ])
        
        choices_text = "\n".join([
            f"{i+1}. {choice}" for i, choice in enumerate(choices)
        ])
        
        prompt = f"""다음 참고 자료는 문제 해결을 돕기 위한 배경 지식입니다. 이 자료는 위키피디아에서 검색된 내용으로, 문제의 지문과 직접 관련된 역사적 사실, 사회적 개념, 경제 원리, 정치 제도, 지리적 정보, 심리학 이론 등을 보충합니다.

참고 자료를 활용할 때는 다음을 유념하세요:
1. 원래 지문이 최우선 판단 기준입니다. 참고 자료는 지문의 맥락을 이해하거나 불분명한 개념을 보충하는 용도로만 사용하세요.
2. 참고 자료와 지문 내용이 충돌하면 반드시 지문을 따르세요.
3. 수능/KMMLU/KLUE MRC 문제는 지문 기반 추론이 핵심입니다. 선택지 판단 시 지문의 논리 흐름과 근거를 우선 분석하고, 참고 자료는 보조적으로만 활용하세요.
4. 각 선택지가 지문 및 참고 자료의 근거에 부합하는지 하나씩 검토한 뒤, 가장 타당한 하나만 고르세요.

참고 자료:
{rag_context}

---

지문을 읽고 질문의 답을 구하세요.

지문:
{paragraph}

질문:
{question}

선택지:
{choices_text}

1~{len(choices)} 중에 하나를 정답으로 고르세요.
정답:"""
        
        result = self.llm.predict_choice(prompt, num_choices=len(choices))
        
        return {
            'answer': result['answer'],
            'confidence': result['confidence'],
            'probs': result['probs'],
            'reasoning': '통합 추론 완료'
        }


# ========== 선택지 검증 ==========

class ChoiceValidator:
    """모든 선택지가 충분히 검증되었는지 확인"""
    
    def validate_all_choices(
        self,
        reasoning_result: Dict[str, Any],
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """선택지 검증"""
        # 간단한 검증: 확률이 충분히 높으면 검증 완료
        confidence = reasoning_result.get('confidence', 0.0)
        threshold = 0.6  # 임계값
        
        all_validated = confidence >= threshold
        
        return {
            'all_validated': all_validated,
            'confidence': confidence,
            'threshold': threshold
        }


# ========== Agentic RAG Chain ==========

class AgenticRAGChain:
    """전체 Agentic RAG 워크플로우"""
    
    def __init__(
        self,
        paragraph_analyzer: ParagraphAnalyzer,
        query_generator: QueryGenerator,
        retriever: FAISSRetriever,
        search_filter: SearchResultFilter,
        integrated_reasoner: IntegratedReasoner,
        choice_validator: ChoiceValidator,
        llm: MultipleChoiceLLM,
        tracer: Optional[LangChainTracer] = None
    ):
        self.paragraph_analyzer = paragraph_analyzer
        self.query_generator = query_generator
        self.retriever = retriever
        self.search_filter = search_filter
        self.integrated_reasoner = integrated_reasoner
        self.choice_validator = choice_validator
        self.llm = llm
        self.tracer = tracer
    
    def solve(
        self,
        paragraph: str,
        question: str,
        choices: List[str],
        callbacks: Optional[List] = None,
        max_search_iterations: int = 3
    ) -> Dict[str, Any]:
        """전체 워크플로우 실행"""
        # 메모리 사용량 로깅
        mem_before = get_memory_usage()
        if mem_before:
            print(f"[메모리] 시작: {mem_before['allocated']:.2f}GB")
        
        # LangSmith 트래킹 (간단한 로깅으로 대체)
        # 실제 LangSmith 통합은 LangChain의 자동 트래킹에 의존
        workflow_run = None
        
        try:
            # 1단계: 지문 분석
            analysis = self.paragraph_analyzer.analyze(
                paragraph, question, choices, callbacks=callbacks
            )
            
            cleanup_memory()  # 지문 분석 후 정리
            
            if analysis['can_answer_without_search']:
                # 지문만으로 답 가능
                result = self.llm.predict_choice(
                    f"{paragraph}\n\n{question}\n\n{chr(10).join([f'{i+1}. {c}' for i, c in enumerate(choices)])}\n\n1~{len(choices)} 중에 하나를 정답으로 고르세요.\n정답:",
                    num_choices=len(choices)
                )
                
                cleanup_memory()
                return result
            
            # 2단계: 검색 필요
            search_results = []
            iteration = 0
            
            while iteration < max_search_iterations:
                # 검색 쿼리 생성
                query = self.query_generator.generate_query(
                    paragraph, question, choices,
                    previous_queries=[q for q, _ in search_results],
                    callbacks=callbacks
                )
                
                cleanup_memory()  # 쿼리 생성 후 정리
                
                # FAISS 검색
                # invoke 메서드를 사용하면 LangChain이 자동으로 run_manager 생성
                docs = self.retriever.invoke(query)
                
                cleanup_memory()  # 검색 후 정리
                
                # 필터링 및 리랭킹
                filtered_docs = self.search_filter.filter_and_rerank(
                    docs, query, top_k=5, callbacks=callbacks
                )
                
                # 원본 docs 삭제
                del docs
                cleanup_memory()  # 필터링 후 정리
                
                search_results.append((query, filtered_docs))
                
                # 3단계: 통합 추론
                reasoning = self.integrated_reasoner.reason(
                    paragraph, question, choices, filtered_docs, callbacks=callbacks
                )
                
                cleanup_memory()  # 추론 후 정리
                
                # 검증: 모든 선택지가 충분히 검증되었는가?
                validation = self.choice_validator.validate_all_choices(
                    reasoning, callbacks=callbacks
                )
                
                cleanup_memory()  # 검증 후 정리
                
                if validation['all_validated']:
                    break
                
                iteration += 1
            
            # 최종 답변
            final_result = reasoning
            
            cleanup_memory()  # 최종 답변 생성 후 정리
            
            # 메모리 사용량 로깅
            mem_after = get_memory_usage()
            if mem_after:
                print(f"[메모리] 종료: {mem_after['allocated']:.2f}GB")
            
            return final_result
            
        except Exception as e:
            print(f"[오류] Agentic RAG 실행 중 오류 발생: {e}")
            raise


# ========== 시스템 구축 함수 ==========

def build_agentic_qa_system(
    model_name: str = "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    faiss_repo_id: str = "NLP-07-ODQA/kowiki-faiss-index",
    local_faiss_dir: str = "./downloaded_faiss",
    enable_tracing: bool = True,
    use_mmap: bool = True
) -> Dict[str, Any]:
    """Agentic QA 시스템 구축"""
    print("=" * 60)
    print("Agentic RAG 시스템 구축 시작")
    print("=" * 60)
    
    # 1. FAISS 인덱스 로드 (mmap 사용)
    print("\n[1/5] FAISS 인덱스 로딩...")
    index, metadata_list, embedding_model = load_hf_faiss_index(
        repo_id=faiss_repo_id,
        local_dir=local_faiss_dir,
        use_mmap=use_mmap
    )
    cleanup_memory()
    
    # 2. LLM 모델 로드
    print("\n[2/5] LLM 모델 로딩...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
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
    print("\n[3/5] LangSmith 설정...")
    if enable_tracing:
        tracer, callbacks = setup_langsmith()
    else:
        tracer, callbacks = None, None
    
    # 4. 컴포넌트 초기화
    print("\n[4/5] 컴포넌트 초기화...")
    retriever = FAISSRetriever(index, metadata_list, embedding_model)
    paragraph_analyzer = ParagraphAnalyzer(llm)
    query_generator = QueryGenerator(llm)
    search_filter = SearchResultFilter()
    integrated_reasoner = IntegratedReasoner(llm)
    choice_validator = ChoiceValidator()
    
    # 5. Agentic RAG Chain 구축
    print("\n[5/5] Agentic RAG Chain 구축...")
    agentic_chain = AgenticRAGChain(
        paragraph_analyzer=paragraph_analyzer,
        query_generator=query_generator,
        retriever=retriever,
        search_filter=search_filter,
        integrated_reasoner=integrated_reasoner,
        choice_validator=choice_validator,
        llm=llm,
        tracer=tracer
    )
    
    print("\n✅ Agentic RAG 시스템 구축 완료!")
    
    return {
        "chain": agentic_chain,
        "llm": llm,
        "retriever": retriever,
        "callbacks": callbacks
    }

