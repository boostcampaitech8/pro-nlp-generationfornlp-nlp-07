# ============================================================
# mcq_retrieval.py - 2-Stage Retrieval (Title → Content) + Question-based Retrieval
# ============================================================

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings

from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from datasets import load_dataset
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from typing import List, Any, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
from pydantic import ConfigDict, Field
from pathlib import Path
import pickle
import re
import gc
import torch
import faiss
from tqdm import tqdm
import pandas as pd
import json
import ast
import time

# ========== 0단계: 문서 로드 (전체 - 청킹 없음) ==========
def load_wikipedia_documents(cache_dir="./cache"):
    """
    청킹 없이 원본 문서만 로드
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    cache_file = cache_path / "full_docs.pkl"
    
    if cache_file.exists():
        print(f"캐시에서 문서 로딩: {cache_file}")
        with open(cache_file, 'rb') as f:
            docs = pickle.load(f)
        print(f"로드 완료: {len(docs):,}개 문서")
        return docs
    
    print("한국어 위키피디아 데이터 로딩 중...")
    dataset = load_dataset("NLP-07-ODQA/kowiki-cleaned", split="train")
    
    # 품질 필터링만
    print("품질 필터링 중...")
    docs = []
    for item in tqdm(dataset, desc="Filtering"):
        content = item['content']
        title = item['title']
        
        if (200 <= len(content) <= 10000 and 
            not any(p in title for p in ['위키백과:', '분류:', '틀:', '목록'])):
            docs.append(Document(
                page_content=content,
                metadata={
                    'title': title,
                    'page_id': item['page_id']
                }
            ))
    
    print(f"필터링 완료: {len(docs):,}개 문서")
    
    with open(cache_file, 'wb') as f:
        pickle.dump(docs, f)
    
    return docs



# ========== 0-1단계: 제목 FAISS 인덱스 생성 (매우 빠름) ==========
def build_title_index(docs, cache_dir="./cache"):
    """
    문서 제목만 임베딩 (더 빠르게)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    title_index_path = cache_path / "title_context_faiss_index"
    
    if title_index_path.exists():
        print(f"제목 FAISS 인덱스 이미 존재: {title_index_path}")
        return
    
    print("=" * 60)
    print("Phase 1: 제목 임베딩 & FAISS 인덱스 생성")
    print("=" * 60)
    
    print("임베딩 모델 로딩...")
    model = SentenceTransformer(
        "dragonkue/BGE-m3-ko", 
        device='cuda',
        model_kwargs={'dtype': torch.float16}
    )
    
    print("제목 + 첫 문단 추출 중...")
    search_texts = []
    for doc in docs:
        title = doc.metadata['title']
        content_preview = doc.page_content[:200]
        search_text = f"{title}\n{content_preview}"
        search_texts.append(search_text)
    
    print(f"임베딩 중... (총 {len(search_texts):,}개)")
    embeddings_array = model.encode(
        search_texts,
        batch_size=1024,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    
    print(f"임베딩 완료: shape={embeddings_array.shape}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("FAISS 인덱스 구축 중...")
    
    dimension = embeddings_array.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array.astype('float32'))
    
    index_to_id = {i: str(i) for i in range(len(docs))}
    docstore = InMemoryDocstore({str(i): docs[i] for i, doc in enumerate(docs)})
    
    embed_wrapper = HuggingFaceEmbeddings(
        model_name="dragonkue/BGE-m3-ko",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS(
        embedding_function=embed_wrapper,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_id
    )
    
    print(f"FAISS 인덱스 저장 중: {title_index_path}")
    vectorstore.save_local(str(title_index_path))
    print("제목 FAISS 인덱스 저장 완료!")
    
    del embeddings_array
    del vectorstore
    del embed_wrapper
    gc.collect()
    torch.cuda.empty_cache()
    print()


def _call(
    self,
    prompt: str,
    stop: Optional[List[str]] = None,
    run_manager: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    import time
    
    t0 = time.time()
    num_choices = self._extract_num_choices(prompt)
    
    t1 = time.time()
    inputs = self.tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True,
        max_length=4096
    ).to(self.model.device)
    
    t2 = time.time()
    with torch.no_grad():
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        
        choice_tokens = [str(i) for i in range(1, num_choices + 1)]
        choice_token_ids = [
            self.tokenizer.encode(choice, add_special_tokens=False)[0] 
            for choice in choice_tokens
        ]
        
        choice_logits_tensor = logits[0, choice_token_ids]
        choice_probs_tensor = torch.nn.functional.softmax(choice_logits_tensor, dim=-1)
        
        self.last_probs = {choice: prob.item() for choice, prob in zip(choice_tokens, choice_probs_tensor)}
        
        predicted_idx = torch.argmax(choice_probs_tensor).item()
        predicted_answer = choice_tokens[predicted_idx]
        self.last_answer = predicted_answer
        self.last_confidence = choice_probs_tensor[predicted_idx].item()
    
    # 메모리 정리
    del inputs
    del outputs
    del logits
    torch.cuda.empty_cache()

    t3 = time.time()
    
    print(f"  [LLM] 토크나이즈: {t2-t1:.2f}s | Forward: {t3-t2:.2f}s | 총: {t3-t0:.2f}s")
    
    return predicted_answer


class MultipleChoiceLogitLLM(LLM):
    """
    객관식 문제의 선택지 확률 추출
    - probs: 모든 선택지 확률
    - answer: 최종 답안
    - confidence: 최종 답안의 확률
    """
    model: Any = Field(exclude=True)
    tokenizer: Any = Field(exclude=True)
    verbose: bool = False  

    last_probs: Optional[Dict[str, float]] = Field(default=None, exclude=True)
    last_answer: Optional[str] = Field(default=None, exclude=True)
    last_confidence: Optional[float] = Field(default=None, exclude=True)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def _llm_type(self) -> str:
        return "multiple_choice_confidence"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        t0 = time.time()
        num_choices = self._extract_num_choices(prompt)
        
        t1 = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
        
        t2 = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            
            choice_tokens = [str(i) for i in range(1, num_choices + 1)]
            choice_token_ids = [
                self.tokenizer.encode(choice, add_special_tokens=False)[0] 
                for choice in choice_tokens
            ]
            
            choice_logits_tensor = logits[0, choice_token_ids]
            choice_probs_tensor = torch.nn.functional.softmax(choice_logits_tensor, dim=-1)
            
            self.last_probs = {choice: prob.item() for choice, prob in zip(choice_tokens, choice_probs_tensor)}
            
            predicted_idx = torch.argmax(choice_probs_tensor).item()
            predicted_answer = choice_tokens[predicted_idx]
            self.last_answer = predicted_answer
            self.last_confidence = choice_probs_tensor[predicted_idx].item()
        
        t3 = time.time()
        
        if self.verbose:  # verbose 모드일 때만 출력
            print(f"  [LLM] 토크나이즈: {t2-t1:.2f}s | Forward: {t3-t2:.2f}s | 총: {t3-t0:.2f}s")
        
        return predicted_answer
    
    def _extract_num_choices(self, prompt: str) -> int:
        match = re.search(r'1~(\d+)\s*중에', prompt)
        if match:
            return int(match.group(1))
        return 5
    
    def get_probs(self) -> Optional[Dict[str, float]]:
        """모든 선택지의 확률 반환 {'1': 0.15, '2': 0.45, ...}"""
        return self.last_probs
    
    def get_answer(self) -> Optional[str]:
        """예측된 답안 반환 ('1', '2', ...)"""
        return self.last_answer
    
    def get_confidence(self) -> Optional[float]:
        """최종 답안의 확률 반환 (0~1)"""
        return self.last_confidence
    
    def get_result(self) -> Dict[str, Any]:
        """전체 결과 반환 (probs, answer, confidence)"""
        return {
            'probs': self.last_probs,           # 모든 선택지 확률
            'answer': self.last_answer,          # 최종 답안
            'confidence': self.last_confidence   # 최종 답안의 확률
        }


# ========== 1단계: 2-Stage Retriever ==========
class TwoStageRetriever(BaseRetriever):
    title_vectorstore: FAISS
    all_docs: List[Document]
    top_k_docs: int = 30
    top_k_chunks: int = 10
    verbose: bool = False  # 추가
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        t0 = time.time()
        relevant_docs = self.title_vectorstore.similarity_search(query, k=self.top_k_docs)
        t1 = time.time()

        torch.cuda.empty_cache()
        
        t2 = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", ".", " "]
        )
        
        chunked_docs = []
        for doc in relevant_docs:
            chunks = text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)
        t3 = time.time()
        
        t4 = time.time()
        bm25_retriever = BM25Retriever.from_documents(chunked_docs)
        bm25_retriever.k = self.top_k_chunks
        final_docs = bm25_retriever.invoke(query)
        t5 = time.time()
        
        if self.verbose:  # verbose 모드일 때만 출력
            print(f"      [임베딩검색: {t1-t0:.2f}s | 청킹: {t3-t2:.2f}s | BM25: {t5-t4:.2f}s | 합계: {t5-t0:.2f}s]")
        
        return final_docs


# ========== 2단계: Question + Option Retriever ==========
class QuestionAndOptionRetriever(BaseRetriever):
    base_retriever: TwoStageRetriever
    top_k_for_question: int = 5
    top_k_per_option: int = 3
    max_workers: int = 5
    verbose: bool = False
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def _retrieve_for_question(
        self,
        question: str,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = self.base_retriever.invoke(
            question,
            config={"callbacks": run_manager.get_child()}
        )
        
        for doc in docs[:self.top_k_for_question]:
            doc.metadata['source_type'] = 'question'
            doc.metadata['related_choice'] = 0
        
        return docs[:self.top_k_for_question]
    
    def _retrieve_for_option(
        self, 
        question: str, 
        choice: str, 
        idx: int,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        option_query = f"{question}\n정답 후보: {choice}"
        docs = self.base_retriever.invoke(
            option_query, 
            config={"callbacks": run_manager.get_child()}
        )
        
        for doc in docs[:self.top_k_per_option]:
            doc.metadata['related_choice'] = idx + 1
            doc.metadata['choice_text'] = choice
            doc.metadata['source_type'] = 'option'
        
        return docs[:self.top_k_per_option]
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        parts = query.split("|||")
        question = parts[0]
        choices = parts[1:]
        
        all_docs = []
        
        t0 = time.time()
        if self.verbose:
            print(f"    [질문 검색]")
        question_docs = self._retrieve_for_question(question, run_manager)
        all_docs.extend(question_docs)
        t1 = time.time()
        if self.verbose:
            print(f"    질문 검색 시간: {t1-t0:.2f}초\n")
        
        t2 = time.time()
        if self.verbose:
            print(f"    [선택지 검색 {len(choices)}개]")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._retrieve_for_option, 
                    question, 
                    choice, 
                    idx, 
                    run_manager
                )
                for idx, choice in enumerate(choices)
            ]
            
            for idx, future in enumerate(futures, 1):
                docs = future.result()
                all_docs.extend(docs)
        t3 = time.time()
        if self.verbose:
            print(f"    선택지 검색 시간: {t3-t2:.2f}초\n")
        
        unique_docs = self._remove_duplicates(all_docs)
        
        return unique_docs
    
    def _remove_duplicates(self, docs: List[Document]) -> List[Document]:
        seen = set()
        unique = []
        for doc in docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique.append(doc)
        return unique



# ========== 3단계: Retrieval 파이프라인 구축 ==========
def build_retrieval_pipeline(docs, embedding_model, use_reranker=False, cache_dir="./cache"):
    """
    2-Stage Retrieval 파이프라인 (질문 + 선택지 + Re-ranking)
    embedding_model: 미리 로드된 임베딩 모델
    """
    print("=" * 60)
    print("Phase 2: Retrieval 파이프라인 구축")
    print("=" * 60)
    
    cache_path = Path(cache_dir)
    title_index_path = cache_path / "title_context_faiss_index"
    
    print(f"제목 FAISS 인덱스 로딩: {title_index_path}")
    title_vectorstore = FAISS.load_local(
        str(title_index_path),
        embedding_model,  # 외부에서 받은 임베딩 모델 사용
        allow_dangerous_deserialization=True
    )
    
    # 2-Stage Retriever
    two_stage_retriever = TwoStageRetriever(
        title_vectorstore=title_vectorstore,
        all_docs=docs,
        top_k_docs=30,
        top_k_chunks=10
    )
    
    # 질문 + 선택지 Retriever
    question_option_retriever = QuestionAndOptionRetriever(
        base_retriever=two_stage_retriever,
        top_k_for_question=3,
        top_k_per_option=1,
        max_workers=5
    )
    
    if use_reranker:
        print("Cross-Encoder Re-ranker 로딩 (CPU)...")
        cross_encoder = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-large",
            model_kwargs={'device': 'cpu'}
        )
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=10)
        
        final_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=question_option_retriever
        )
        print("Retrieval 파이프라인 구축 완료! (Reranker 활성화)")
    else:
        final_retriever = question_option_retriever
        print("Retrieval 파이프라인 구축 완료! (Reranker 비활성화)")
    
    print()
    return final_retriever





# ========== 4단계: QA Chain ==========
def format_mcq_for_retrieval(question: str, choices: List[str]) -> str:
    return "|||".join([question] + choices)



def build_qa_chain(final_retriever, reader_llm):
    """
    QA Chain 생성 (RAG + 원래 지문)
    """
    prompt_template = """다음 참고 자료는 문제 해결을 돕기 위한 배경 지식입니다. 이 자료는 위키피디아에서 검색된 내용으로, 문제의 지문과 직접 관련된 역사적 사실, 사회적 개념, 경제 원리, 정치 제도, 지리적 정보, 심리학 이론 등을 보충합니다.
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
{original_paragraph}


질문:
{question}


선택지:
{choices}


{choice_range} 중에 하나를 정답으로 고르세요.
정답:"""
    
    prompt = PromptTemplate(
        input_variables=["rag_context", "original_paragraph", "question", "choices", "choice_range"],
        template=prompt_template
    )
    
    last_retrieved_docs = {"docs": None}
    
    def format_docs(docs):
        """RAG 검색 문서"""
        last_retrieved_docs["docs"] = docs

        if not docs:
            return "(관련 자료 없음)"
        return "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
    
    # Custom Chain
    chain = (
        {
            "rag_context": lambda x: format_docs(final_retriever.invoke(x["retrieval_query"])),
            "original_paragraph": lambda x: x.get("paragraph", ""),
            "question": lambda x: x["question"],
            "choices": lambda x: x["choices"],
            "choice_range": lambda x: x["choice_range"]
        }
        | prompt
        | reader_llm
        | StrOutputParser()
    )
    
    return {
        "chain": chain,
        "retriever": final_retriever,
        "get_last_docs": lambda: last_retrieved_docs["docs"]
    }



def solve_mcq(question: str, choices: List[str], qa_system, paragraph=""):
    """
    수능 문제 풀이 실행
    
    Args:
        question: 질문
        choices: 선택지 리스트 (개수 유동적)
        chain_and_retriever: (chain, retriever) 튜플
        paragraph: 원래 문제의 지문
    """
    chain = qa_system["chain"]
    get_last_docs = qa_system["get_last_docs"]
    
    # Retrieval용 쿼리
    retrieval_query = format_mcq_for_retrieval(question, choices)
    
    # 선택지 포맷
    choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
    
    # 선택지 범위 자동 계산
    choice_range = f"1~{len(choices)}"
    
    # Chain 실행
    result_text = chain.invoke({
        "retrieval_query": retrieval_query,
        "paragraph": paragraph,
        "question": question,
        "choices": choices_text,
        "choice_range": choice_range  # 1~5, 1~4, 1~3 등 자동
    })

    source_documents = get_last_docs()

    return {
        "result": result_text,
        "source_documents": source_documents,
        "query": question
    }




def analyze_retrieval_results(result, choices):
    """
    RAG 검색 결과 상세 분석
    """
    print("\n" + "=" * 80)
    print("RAG 검색 결과 분석")
    print("=" * 80)
    
    print(f"\n모델 답변: {result['result']}")
    print(f"총 검색 문서: {len(result['source_documents'])}개")
    
    # 질문 기반 문서
    question_docs = [
        doc for doc in result['source_documents']
        if doc.metadata.get('source_type') == 'question'
    ]
    print(f"\n[질문 기반 검색]: {len(question_docs)}개")
    for i, doc in enumerate(question_docs, 1):
        print(f"  [{i}] {doc.metadata.get('title', 'N/A')[:50]}")
    
    # 선택지별 분석
    print(f"\n[선택지별 검색]:")
    for choice_idx in range(1, 6):
        related_docs = [
            doc for doc in result['source_documents'] 
            if doc.metadata.get('related_choice') == choice_idx
        ]
        
        print(f"\n{'='*80}")
        print(f"선택지 {choice_idx}: {choices[choice_idx-1]}")
        print(f"검색된 문서: {len(related_docs)}개")
        print(f"{'='*80}")
        
        for i, doc in enumerate(related_docs, 1):
            print(f"\n  [{i}] 제목: {doc.metadata.get('title', 'N/A')}")
            print(f"      내용: {doc.page_content[:150].replace(chr(10), ' ')}...")
    
    # 전체 Top 5
    print(f"\n{'='*80}")
    print("전체 검색 Top 5 문서")
    print(f"{'='*80}")
    
    for i, doc in enumerate(result['source_documents'][:5], 1):
        print(f"\n[{i}위] 제목: {doc.metadata.get('title', 'N/A')}")
        print(f"     유형: {doc.metadata.get('source_type', 'N/A')}")
        print(f"     선택지: {doc.metadata.get('related_choice', 'N/A')}")
        print(f"     내용: {doc.page_content[:200].replace(chr(10), ' ')}...")



# ========== 메인 실행 ==========
if __name__ == "__main__":
    import time
    
    # Phase 1: 제목 임베딩
    docs = load_wikipedia_documents()
    build_title_index(docs)
    
    # Phase 1.5: 임베딩 모델 로드 (한 번만!)
    print("=" * 60)
    print("임베딩 모델 로드 (재사용)")
    print("=" * 60)
    embedding_model = HuggingFaceEmbeddings(
        model_name="dragonkue/BGE-m3-ko",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("임베딩 모델 로드 완료!\n")
    
    # Phase 2: Retrieval 파이프라인 구축 (임베딩 모델 전달)
    final_retriever = build_retrieval_pipeline(docs, embedding_model)
    
    # Phase 3: Reader LLM 로드
    print("=" * 60)
    print("Phase 3: Reader LLM 모델 로드 (HuggingFace)")
    print("=" * 60)

    model_name = "NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_15"
    exp_name = "qwen2.5-32b-it-cot15_chain-v1"
    
    print(f"모델 로딩 중: {model_name}")
    
    try:
        print("  [시도 1] LoRA Adapter 자동 로드...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )
        print("  ✓ LoRA Adapter + Base Model 자동 로드 성공")
        
    except Exception as e:
        print(f"  [시도 1 실패: {str(e)[:100]}]")
        print("  [시도 2] 머지된 모델로 로드...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )
        print("  ✓ 머지된 모델 로드 성공")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    reader_llm = MultipleChoiceLogitLLM(
        model=model,
        tokenizer=tokenizer
    )
    
    print("Reader LLM 로드 완료!\n")

    print("\n" + "="*60)
    print("모델 배치 디버깅")
    print("="*60)
    if hasattr(model, 'hf_device_map'):
        cpu_layers = [k for k, v in model.hf_device_map.items() if str(v) == 'cpu']
        if cpu_layers:
            print(f"⚠️  CPU에 배치된 레이어: {len(cpu_layers)}개")
            print(f"예시: {cpu_layers[:5]}")
        else:
            print("✓ 모든 레이어가 GPU에 있습니다")
    else:
        print("device_map 정보 없음")
    
    # Phase 4: QA Chain 구축
    qa_system = build_qa_chain(final_retriever, reader_llm)
    
    # Phase 5: 문제 풀이
    print("=" * 60)
    print("문제 풀이 시작")
    print("=" * 60)
    
    csv_path = "../../../../data/test/test.csv"

    print(f"\nCSV 파일 로딩: {csv_path}")
    df = pd.read_csv(csv_path, converters={'problems': ast.literal_eval})
    print(f"총 {len(df)}개 문제 로드 완료")

    submission_results = []
    detail_results = []

    print("\n추론 시작...")
    start_total = time.time()

    # 메인 루프 수정
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="추론 진행"):
        iter_start = time.time()
        
        problem_id = row['id']
        paragraph = row['paragraph']
        problems_data = row['problems']
        question_plus = row.get('question_plus', '')
        
        question = problems_data.get('question', '')
        choices = problems_data.get('choices', [])
        
        try:
            # 첫 번째 문제만 verbose
            if idx == 0:
                # final_retriever 타입 확인
                retriever_type = type(final_retriever).__name__
                
                if retriever_type == 'ContextualCompressionRetriever':
                    # Reranker 있음: ContextualCompressionRetriever -> QuestionAndOptionRetriever -> TwoStageRetriever
                    final_retriever.base_retriever.verbose = True  # QuestionAndOptionRetriever
                    final_retriever.base_retriever.base_retriever.verbose = True  # TwoStageRetriever
                else:
                    # Reranker 없음: QuestionAndOptionRetriever -> TwoStageRetriever
                    final_retriever.verbose = True  # QuestionAndOptionRetriever
                    final_retriever.base_retriever.verbose = True  # TwoStageRetriever
                
                reader_llm.verbose = True
                
                print(f"\n{'='*60}")
                print(f"[첫 번째 문제 상세 분석] ID: {problem_id}")
                print(f"{'='*60}")
            else:
                retriever_type = type(final_retriever).__name__
                
                if retriever_type == 'ContextualCompressionRetriever':
                    final_retriever.base_retriever.verbose = False
                    final_retriever.base_retriever.base_retriever.verbose = False
                else:
                    final_retriever.verbose = False
                    final_retriever.base_retriever.verbose = False
                
                reader_llm.verbose = False
            
            t0 = time.time()
            result = solve_mcq(
                question=question,
                choices=choices,
                qa_system=qa_system,
                paragraph=paragraph
            )
            t1 = time.time()
            
            if idx == 0:
                print(f"  [Retrieval 전체]: {t1-t0:.2f}초")
            
            result_data = reader_llm.get_result()
            t2 = time.time()
            
            submission_results.append({
                'id': problem_id,
                'answer': result_data['answer']
            })
            
            detail_results.append({
                'id': problem_id,
                'prediction': str(result_data['answer']),
                'probabilities': result_data['probs'],
                'confidence': result_data['confidence'],
                'num_choices': len(choices),
                'num_source_docs': len(result['source_documents'])
            })
            
            if idx == 0:
                iter_time = time.time() - iter_start
                estimated_total = iter_time * len(df) / 3600
                
                print(f"\n답안: {result_data['answer']} | 신뢰도: {result_data['confidence']:.3f}")
                print(f"첫 문제 소요 시간: {iter_time:.1f}초")
                print(f"예상 전체 소요 시간: {estimated_total:.2f}시간\n")
            
            # 메모리 정리 (매 문제마다!)
            if idx % 10 == 0:  # 10개마다 강제 정리
                gc.collect()
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n[오류] ID {problem_id}: {str(e)}")
            submission_results.append({'id': problem_id, 'answer': 0})
            detail_results.append({'id': problem_id, 'error': str(e)})

    print("\n추론 완료!")

    # 최종 통계
    total_time = time.time() - start_total
    avg_time = total_time / len(df)
    print(f"\n최종 통계:")
    print(f"  총 소요 시간: {total_time/3600:.2f}시간")
    print(f"  문제당 평균: {avg_time:.1f}초")

    answer_counts = Counter([str(r['prediction']) for r in detail_results if 'prediction' in r])
    answer_distribution = {str(i): answer_counts.get(str(i), 0) for i in range(1, 6)}

    valid_confidences = [r['confidence'] for r in detail_results if 'confidence' in r]
    avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0

    # 결과 저장
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    submission_df = pd.DataFrame(submission_results)
    submission_path = output_dir / f"{exp_name}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\n제출 파일 저장: {submission_path}")
    print(f"형식:\n{submission_df.head()}")

    detail_path = output_dir / f"{exp_name}_detailed.json"
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_name': exp_name,
            'total_samples': len(df),
            'predictions': detail_results,
            'summary': {  # ✅ summary 추가
                'answer_distribution': answer_distribution,
                'average_confidence': avg_confidence,
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"세부 정보 파일 저장: {detail_path}")

    print(f"\n[Summary]")
    print(f"  답변 분포: {answer_distribution}")
    print(f"  평균 신뢰도: {avg_confidence:.4f}")

    print("\n" + "=" * 60)
    print(f"전체 처리 완료! (성공: {len(submission_results)}개)")
    print("=" * 60)

    DEBUG_MODE = False
    if DEBUG_MODE:
        paragraph = "이것은 테스트 지문입니다. 지문 내용이 여기에 들어갑니다."
        question = "이것은 테스트 질문입니다. 이 문제의 정답은 4번입니다. 그럼에도 불구하고, 문제를 풀고 정답을 고르세요."
        choices =  [
            '1번은 정답이 아닙니다.', 
            '2번은 오답입니다.', 
            '3번은 정답이 아닐 수도 있습니다.', 
            '4번은 정답일 수도 있습니다.', 
            '5번은 정답이 아닌지 모릅니다.',
        ]
        
        result = solve_mcq(question, choices, chain_and_retriever, paragraph)
        
        result_data = reader_llm.get_result()
        
        print(f"\n최종 답안: {result_data['answer']}번")
        print(f"Confidence: {result_data['confidence']:.4f} ({result_data['confidence']*100:.2f}%)")
        
        print(f"\n전체 선택지 확률 분포:")
        for choice, prob in result_data['probs'].items():
            marker = " ← 선택됨" if choice == result_data['answer'] else ""
            print(f"  {choice}번: {prob:.4f} ({prob*100:.2f}%){marker}")
        
        print(f"\n참조 문서 수: {len(result['source_documents'])}")
        
        # 질문 기반 문서 수
        question_docs = [
            d for d in result['source_documents'] 
            if d.metadata.get('source_type') == 'question'
        ]
        print(f"질문 기반: {len(question_docs)}개")
        
        # 선택지별 문서 수
        for choice_idx in range(1, 6):
            related_docs = [
                doc for doc in result['source_documents'] 
                if doc.metadata.get('related_choice') == choice_idx
            ]
            print(f"선택지 {choice_idx}: {len(related_docs)}개 문서")
        
        # 상세 분석
        analyze_retrieval_results(result, choices)
        
        save_data = {
            'question': question,
            'paragraph': paragraph,
            'choices': choices,
            'answer': result_data['answer'],           # 최종 답안
            'confidence': result_data['confidence'],   # 최종 답안 확률
            'probs': result_data['probs'],             # 전체 선택지 확률
            'num_source_docs': len(result['source_documents'])
        }
        
        print(f"\n저장 데이터:")
        print(save_data)

