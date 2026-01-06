"""
Agentic RAG 추론 실행 파일
- Thinking 모드 완전히 비활성화
- 메모리 최적화 적용
- LangSmith 트래킹
"""
import torch
import pandas as pd
import json
import os
from ast import literal_eval
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import sys
from dotenv import load_dotenv
import argparse
import time

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 현재 디렉토리 추가 (같은 폴더의 모듈 import를 위해)
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from agentic_rag import build_agentic_qa_system, cleanup_memory, get_memory_usage

load_dotenv()

# 명령줄 인자 파싱
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean 값이 아닙니다!')

parser = argparse.ArgumentParser(description="Agentic RAG Inference Script")
parser.add_argument('--exp_name', type=str, required=True, help='실험 이름')
parser.add_argument('--model_name', type=str, default='unsloth/Qwen3-14B-unsloth-bnb-4bit', help='모델 이름')
parser.add_argument('--faiss_repo_id', type=str, default='NLP-07-ODQA/kowiki-faiss-index', help='FAISS 인덱스 repo ID')
parser.add_argument('--local_faiss_dir', type=str, default='./downloaded_faiss', help='로컬 FAISS 디렉토리')
parser.add_argument('--max_search_iterations', type=int, default=3, help='최대 재검색 횟수')
parser.add_argument('--enable_tracing', type=str2bool, default=True, help='LangSmith 트래킹 활성화')
parser.add_argument('--use_mmap', type=str2bool, default=True, help='FAISS 인덱스 mmap 사용 (메모리 절약)')
parser.add_argument('--test_data', type=str, default='test.csv', help='테스트 데이터 파일 이름')

args = parser.parse_args()

# LangSmith 프로젝트 이름을 exp_name으로 설정
# 이렇게 하면 각 실험을 별도 프로젝트로 관리할 수 있음
os.environ["LANGCHAIN_PROJECT"] = args.exp_name
print(f"📊 LangSmith 프로젝트: {args.exp_name}")

# 상수
CAMPER_ID = "T8001"
EXP_NAME = args.exp_name
HF_ORG = "NLP-07-ODQA"

# 경로
SUBMISSION_DIR = project_root / "submissions" / CAMPER_ID
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# 데이터 파일 경로
DATA_FILES = {
    "test": str(project_root / "data" / "test" / args.test_data),
}

# 데이터 파싱 함수
def parse_data(example):
    problems = literal_eval(example['problems'])
    return {
        'id': example['id'],
        'paragraph': example['paragraph'],
        'question': problems['question'],
        'question_plus': problems.get('question_plus', None),
        'choices': problems['choices'],
        'answer': problems.get('answer', None),
    }

# ========== 메인 실행 ==========

print("=" * 60)
print("Agentic RAG 추론 시작")
print("=" * 60)

# 1. Agentic QA 시스템 구축
print("\n[단계 1] Agentic QA 시스템 구축...")
qa_system = build_agentic_qa_system(
    model_name=args.model_name,
    faiss_repo_id=args.faiss_repo_id,
    local_faiss_dir=args.local_faiss_dir,
    enable_tracing=args.enable_tracing,
    use_mmap=args.use_mmap
)

agentic_chain = qa_system["chain"]
callbacks = qa_system.get("callbacks", None)

# 2. 테스트 데이터 로드
print("\n[단계 2] 테스트 데이터 로드...")
test_dataset = load_dataset('csv', data_files={'test': DATA_FILES['test']})['test']

# Unnamed: 0 컬럼 제거 (있는 경우)
if 'Unnamed: 0' in test_dataset.column_names:
    test_dataset = test_dataset.remove_columns(['Unnamed: 0'])

print(f"✅ 테스트 데이터 {len(test_dataset)}개 로드 완료")

# 데이터 파싱
test_dataset = test_dataset.map(parse_data, remove_columns=['problems'], desc="데이터 파싱")
print("✅ 데이터 파싱 완료")

# 3. 추론 시작
print("\n[단계 3] 추론 시작...")
print("=" * 60)

predictions = []
detailed_results = []

start_total = time.time()

# 메모리 상태 확인
mem_initial = get_memory_usage()
if mem_initial:
    print(f"\n[초기 메모리] {mem_initial['allocated']:.2f}GB allocated, {mem_initial['reserved']:.2f}GB reserved")

for idx, row in tqdm(enumerate(test_dataset), total=len(test_dataset), desc="추론 진행"):
    iter_start = time.time()
    
    problem_id = row['id']
    paragraph = row['paragraph']
    question = row['question']
    choices = row['choices']
    question_plus = row.get('question_plus', '')
    
    try:
        # Agentic RAG로 문제 풀이
        result = agentic_chain.solve(
            paragraph=paragraph,
            question=question,
            choices=choices,
            callbacks=callbacks,
            max_search_iterations=args.max_search_iterations
        )
        
        # 결과 저장
        predictions.append({
            'id': problem_id,
            'answer': result['answer']
        })
        
        detailed_results.append({
            'id': problem_id,
            'prediction': str(result['answer']),
            'probabilities': result.get('probs', {}),
            'confidence': result.get('confidence', 0.0),
            'num_choices': len(choices)
        })
        
        # 첫 문제 상세 정보 출력
        if idx == 0:
            iter_time = time.time() - iter_start
            estimated_total = iter_time * len(test_dataset) / 3600
            
            print(f"\n[첫 문제 결과]")
            print(f"  ID: {problem_id}")
            print(f"  답안: {result['answer']}")
            print(f"  신뢰도: {result.get('confidence', 0.0):.3f}")
            print(f"  소요 시간: {iter_time:.1f}초")
            print(f"  예상 전체 소요 시간: {estimated_total:.2f}시간")
            
            # 메모리 상태 확인
            mem_current = get_memory_usage()
            if mem_current:
                print(f"  현재 메모리: {mem_current['allocated']:.2f}GB")
        
        # 주기적 메모리 정리 (10개마다)
        if (idx + 1) % 10 == 0:
            cleanup_memory(aggressive=True)
            mem_current = get_memory_usage()
            if mem_current:
                print(f"\n[메모리 정리] {idx+1}개 처리 후: {mem_current['allocated']:.2f}GB")
        
    except Exception as e:
        print(f"\n[오류] ID {problem_id}: {str(e)}")
        predictions.append({
            'id': problem_id,
            'answer': '1'  # 기본값
        })
        detailed_results.append({
            'id': problem_id,
            'error': str(e)
        })

print("\n✅ 추론 완료!")

# 4. 결과 저장
print("\n[단계 4] 결과 저장...")

# 최종 통계
total_time = time.time() - start_total
avg_time = total_time / len(test_dataset)

print(f"\n최종 통계:")
print(f"  총 소요 시간: {total_time/3600:.2f}시간 ({total_time:.1f}초)")
print(f"  문제당 평균: {avg_time:.1f}초")

# 답변 분포
answer_counts = {}
for r in detailed_results:
    if 'prediction' in r:
        pred = r['prediction']
        answer_counts[pred] = answer_counts.get(pred, 0) + 1

answer_distribution = {str(i): answer_counts.get(str(i), 0) for i in range(1, 6)}
print(f"  답변 분포: {answer_distribution}")

# 신뢰도 통계
valid_confidences = [r['confidence'] for r in detailed_results if 'confidence' in r]
if valid_confidences:
    avg_confidence = sum(valid_confidences) / len(valid_confidences)
    print(f"  평균 신뢰도: {avg_confidence:.4f}")

# CSV 저장
submission_df = pd.DataFrame(predictions)
submission_path = SUBMISSION_DIR / f"{EXP_NAME}.csv"
submission_df.to_csv(submission_path, index=False)
print(f"\n✅ 제출 파일 저장: {submission_path}")
print(f"형식:\n{submission_df.head()}")

# JSON 저장 (상세 정보)
detail_path = SUBMISSION_DIR / f"{EXP_NAME}_detailed.json"
with open(detail_path, 'w', encoding='utf-8') as f:
    json.dump({
        'experiment_name': EXP_NAME,
        'total_samples': len(test_dataset),
        'predictions': detailed_results,
        'summary': {
            'answer_distribution': answer_distribution,
            'average_confidence': avg_confidence if valid_confidences else 0.0,
            'total_time_seconds': total_time,
            'avg_time_per_sample': avg_time
        }
    }, f, ensure_ascii=False, indent=2)
print(f"✅ 상세 정보 파일 저장: {detail_path}")

# 최종 메모리 상태
mem_final = get_memory_usage()
if mem_final:
    print(f"\n[최종 메모리] {mem_final['allocated']:.2f}GB allocated, {mem_final['reserved']:.2f}GB reserved")
    if mem_initial:
        print(f"[메모리 증가] {mem_final['allocated'] - mem_initial['allocated']:.2f}GB")

print("\n" + "=" * 60)
print(f"전체 처리 완료! (성공: {len(predictions)}개)")
print("=" * 60)

