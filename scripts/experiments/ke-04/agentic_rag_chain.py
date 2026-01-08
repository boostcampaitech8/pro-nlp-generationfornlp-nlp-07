"""Agentic RAG Chain - 전체 워크플로우"""
import os
from typing import List, Dict, Any, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.tracers import LangChainTracer

from components import (
    ParagraphAnalyzer, QueryGenerator, FAISSRetriever, SearchResultFilter,
    IntegratedReasoner, ChoiceValidator, ChoiceEvaluator, FinalReasoner,
    MultipleChoiceLLM
)
from utils.memory import cleanup_memory, get_memory_usage
from utils.langsmith_setup import LANGSMITH_AVAILABLE

# LangSmith SDK for @traceable decorator
try:
    import langsmith as ls
    from langsmith import traceable, get_current_run_tree
    LANGCHAIN_TRACEABLE_AVAILABLE = True
except ImportError:
    LANGCHAIN_TRACEABLE_AVAILABLE = False
    traceable = lambda **kwargs: lambda f: f  # No-op decorator
    get_current_run_tree = lambda: None


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
        choice_evaluator: ChoiceEvaluator,
        final_reasoner: FinalReasoner,
        llm: MultipleChoiceLLM,
        tracer: Optional[LangChainTracer] = None,
        langsmith_client: Optional[Any] = None
    ):
        self.paragraph_analyzer = paragraph_analyzer
        self.query_generator = query_generator
        self.retriever = retriever
        self.search_filter = search_filter
        self.integrated_reasoner = integrated_reasoner
        self.choice_validator = choice_validator
        self.choice_evaluator = choice_evaluator
        self.final_reasoner = final_reasoner
        self.llm = llm
        self.tracer = tracer
        self.langsmith_client = langsmith_client
    
    @traceable(name="agentic_rag_solve")
    def solve(
        self,
        paragraph: str,
        question: str,
        choices: List[str],
        callbacks: Optional[List] = None,
        max_search_iterations: int = 3,
        problem_id: Optional[str] = None,
        langsmith_extra: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """전체 워크플로우 실행 (선택지별 RAG + 2단계 판단)"""
        # 메모리 사용량 로깅
        mem_before = get_memory_usage()
        if mem_before:
            print(f"[메모리] 시작: {mem_before['allocated']:.2f}GB")
        
        # @traceable이 자동으로 parent run 생성
        # 현재 run tree 가져오기
        current_run_tree = get_current_run_tree()
        
        # problem_id를 metadata에 추가
        if current_run_tree and problem_id:
            try:
                # RunTree의 metadata 설정
                if hasattr(current_run_tree, 'extra'):
                    if 'metadata' not in current_run_tree.extra:
                        current_run_tree.extra['metadata'] = {}
                    current_run_tree.extra['metadata']['problem_id'] = str(problem_id)
                elif hasattr(current_run_tree, 'metadata'):
                    current_run_tree.metadata = current_run_tree.metadata or {}
                    current_run_tree.metadata['problem_id'] = str(problem_id)
                print(f"🔍 LangSmith trace 시작: problem_id={problem_id}")
            except Exception as e:
                print(f"⚠️ metadata 설정 실패: {e}")
        
        result = None
        try:
            # 1단계: 지문 분석
            # LangChain callbacks를 통해 자동으로 trace됨
            analysis = self._analyze_paragraph_with_trace(
                paragraph, question, choices, callbacks, current_run_tree
            )
            
            cleanup_memory()  # 지문 분석 후 정리
            
            if analysis['can_answer_without_search']:
                # 지문만으로 답 가능
                result = self._direct_answer_with_trace(
                    paragraph, question, choices, current_run_tree
                )
                
                # 지문만으로 답한 경우는 choice_evaluations가 없음
                result['choice_evaluations'] = None
                result['method'] = 'direct_answer'
                
                cleanup_memory()
                return result
            
            # 2단계: 선택지별 RAG 및 개별 판단
            choice_results = {}  # {"1": {query, docs, evaluation}, ...}
            correct_choices = []  # 명확히 맞는 선택지들
            incorrect_choices = []  # 명확히 틀린 선택지들
            ambiguous_choices = {}  # 애매한 선택지들
            
            print(f"\n[선택지별 RAG 시작] 총 {len(choices)}개 선택지")
            
            for i, choice in enumerate(choices):
                choice_num = str(i + 1)
                print(f"\n[선택지 {choice_num}] 처리 중...")
                
                # 선택지별 RAG 처리 (nested trace)
                choice_result = self._process_choice_rag_with_trace(
                    paragraph, question, choice, choice_num,
                    callbacks, current_run_tree
                )
                
                query = choice_result['query']
                filtered_docs = choice_result['filtered_docs']
                evaluation = choice_result['evaluation']
                
                cleanup_memory()  # 판단 후 정리
                
                # 결과 저장
                choice_results[choice_num] = {
                    'choice': choice,
                    'query': query,
                    'docs': filtered_docs,
                    'evaluation': evaluation
                }
                
                # @traceable이 자동으로 trace 종료 처리
                
                # 판단 결과에 따라 분류
                judgment = evaluation.get('judgment', 'ambiguous')
                confidence = evaluation.get('confidence', 0.5)
                
                if judgment == "correct" and confidence >= 0.8:
                    correct_choices.append(choice_num)
                    print(f"  → 명확히 맞음 (신뢰도: {confidence:.2f})")
                elif judgment == "incorrect" and confidence <= 0.2:
                    incorrect_choices.append(choice_num)
                    print(f"  → 명확히 틀림 (신뢰도: {confidence:.2f})")
                else:
                    ambiguous_choices[choice_num] = choice_results[choice_num]
                    print(f"  → 애매함 (신뢰도: {confidence:.2f})")
            
            cleanup_memory()  # 모든 선택지 처리 후 정리
            
            # 3단계: 최종 판단
            print(f"\n[최종 판단]")
            print(f"  명확히 맞음: {len(correct_choices)}개 {correct_choices}")
            print(f"  명확히 틀림: {len(incorrect_choices)}개 {incorrect_choices}")
            print(f"  애매함: {len(ambiguous_choices)}개 {list(ambiguous_choices.keys())}")
            
            # 각 선택지별 evaluation 결과 정리
            choice_evaluations = {}
            for choice_num, choice_data in choice_results.items():
                evaluation = choice_data['evaluation']
                choice_evaluations[choice_num] = {
                    'choice': choice_data['choice'],
                    'judgment': evaluation.get('judgment', 'ambiguous'),
                    'confidence': evaluation.get('confidence', 0.5),
                    'reasoning': evaluation.get('reasoning', '')
                }
            
            # 최종 판단 (필요시)
            needs_final_judgment = (len(correct_choices) > 1 or len(ambiguous_choices) > 0)
            
            if len(correct_choices) == 1:
                # 명확히 맞는 선택지가 1개면 그것이 정답
                final_answer = correct_choices[0]
                result = {
                    'answer': final_answer,
                    'confidence': choice_results[final_answer]['evaluation']['confidence'],
                    'probs': {final_answer: 1.0},
                    'reasoning': f'명확히 맞는 선택지: {final_answer}',
                    'choice_evaluations': choice_evaluations
                }
            elif len(correct_choices) > 1:
                # 명확히 맞는 선택지가 여러 개면 애매한 선택지로 취급하여 최종 판단
                print(f"  → 명확히 맞는 선택지가 여러 개이므로 최종 판단 필요")
                for choice_num in correct_choices:
                    ambiguous_choices[choice_num] = choice_results[choice_num]
                final_result = self.final_reasoner.reason_ambiguous_choices(
                    paragraph, question, ambiguous_choices, callbacks=callbacks
                )
                final_result['choice_evaluations'] = choice_evaluations
                result = final_result
            elif len(ambiguous_choices) > 0:
                # 애매한 선택지들만 최종 판단
                final_result = self.final_reasoner.reason_ambiguous_choices(
                    paragraph, question, ambiguous_choices, callbacks=callbacks
                )
                final_result['choice_evaluations'] = choice_evaluations
                result = final_result
            else:
                # 모든 선택지가 명확히 틀림 (이상한 경우)
                # 가장 덜 틀린 선택지를 선택
                min_incorrect_confidence = 1.0
                best_choice = None
                for choice_num in incorrect_choices:
                    conf = choice_results[choice_num]['evaluation']['confidence']
                    if conf < min_incorrect_confidence:
                        min_incorrect_confidence = conf
                        best_choice = choice_num
                
                result = {
                    'answer': best_choice if best_choice else "1",
                    'confidence': 1.0 - min_incorrect_confidence,
                    'probs': {best_choice: 1.0 - min_incorrect_confidence} if best_choice else {},
                    'reasoning': '모든 선택지가 명확히 틀림으로 판단됨',
                    'choice_evaluations': choice_evaluations
                }
            
            cleanup_memory()  # 최종 답변 생성 후 정리
            
            # 메모리 사용량 로깅
            mem_after = get_memory_usage()
            if mem_after:
                print(f"[메모리] 종료: {mem_after['allocated']:.2f}GB")
            
            # @traceable이 자동으로 output 저장 및 종료
            # 명시적으로 run tree 종료 (필요시)
            if current_run_tree:
                try:
                    # output을 명시적으로 설정
                    if hasattr(current_run_tree, 'end'):
                        current_run_tree.end(outputs=result)
                    elif hasattr(current_run_tree, 'patch'):
                        current_run_tree.patch(outputs=result)
                except Exception as e:
                    print(f"⚠️ run tree 종료 실패 (무시): {e}")
            
            return result
            
        except Exception as e:
            print(f"[오류] Agentic RAG 실행 중 오류 발생: {e}")
            # 에러 발생 시 run tree 종료
            if current_run_tree:
                try:
                    if hasattr(current_run_tree, 'end'):
                        current_run_tree.end(error=str(e))
                    elif hasattr(current_run_tree, 'patch'):
                        current_run_tree.patch(error=str(e))
                except:
                    pass
            raise
    
    def _analyze_paragraph_with_trace(
        self,
        paragraph: str,
        question: str,
        choices: List[str],
        callbacks: Optional[List],
        parent_run_tree: Optional[Any]
    ) -> Dict[str, Any]:
        """지문 분석 (LangChain callbacks를 통해 자동 trace)"""
        # LangChain callbacks를 통해 자동으로 trace됨
        return self.paragraph_analyzer.analyze(
            paragraph, question, choices, callbacks=callbacks
        )
    
    def _direct_answer_with_trace(
        self,
        paragraph: str,
        question: str,
        choices: List[str],
        parent_run_tree: Optional[Any]
    ) -> Dict[str, Any]:
        """지문만으로 답변 (LangChain callbacks를 통해 자동 trace)"""
        # LangChain callbacks를 통해 자동으로 trace됨
        return self.llm.predict_choice(
            f"{paragraph}\n\n{question}\n\n{chr(10).join([f'{i+1}. {c}' for i, c in enumerate(choices)])}\n\n1~{len(choices)} 중에 하나를 정답으로 고르세요.\n정답:",
            num_choices=len(choices)
        )
    
    def _process_choice_rag_with_trace(
        self,
        paragraph: str,
        question: str,
        choice: str,
        choice_num: str,
        callbacks: Optional[List],
        parent_run_tree: Optional[Any]
    ) -> Dict[str, Any]:
        """선택지별 RAG 처리 (LangChain callbacks를 통해 자동 trace)"""
        # LangChain callbacks를 통해 자동으로 trace됨
        
        # 2-1. 선택지별 쿼리 생성
        query = self.query_generator.generate_query_for_choice(
            paragraph, question, choice, choice_num,
            previous_queries=None,
            callbacks=callbacks
        )
        
        cleanup_memory()  # 쿼리 생성 후 정리
        
        # 2-2. 선택지별 검색
        docs = self.retriever.invoke(
            query,
            config={"callbacks": callbacks} if callbacks else {}
        )
        
        cleanup_memory()  # 검색 후 정리
        
        # 2-3. 필터링 및 리랭킹
        filtered_docs = self.search_filter.filter_and_rerank(
            docs, query, top_k=5, callbacks=callbacks
        )
        
        # 원본 docs 삭제
        del docs
        cleanup_memory()  # 필터링 후 정리
        
        # 2-4. 선택지별 개별 판단
        evaluation = self.choice_evaluator.evaluate_choice(
            paragraph, question, choice, choice_num,
            filtered_docs, callbacks=callbacks
        )
        
        return {
            'query': query,
            'filtered_docs': filtered_docs,
            'evaluation': evaluation
        }

