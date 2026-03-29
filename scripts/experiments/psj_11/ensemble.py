import pandas as pd
from collections import Counter
import os

# 1. 파일들이 있는 폴더 경로 설정
base_path = "data/train/ensemble_data"

# 2. 앙상블할 파일 이름 리스트
file_names = [
    "submission_deepseek_distill_qwen32_v3.csv",
    "submission_ens_ENS_softvote_aligned_3 (1).csv",
    "submission_ens_ENS_softvote_aligned_5.csv",
    "submission_ens_ENS_softvote_aligned_7.csv"
]

# 전체 경로 생성
file_paths = [os.path.join(base_path, f) for f in file_names]

# 3. 데이터 로드 및 앙상블 수행
dfs = [pd.read_csv(f) for f in file_paths]

# 기준이 될 첫 번째 파일의 ID 가져오기
ensemble_df = dfs[0][['id']].copy()

# 각 파일의 정답(answer)을 pred_0, pred_1... 열에 추가
for i, df in enumerate(dfs):
    ensemble_df[f'pred_{i}'] = df['answer']

# 다수결 투표 함수 (Hard Voting)
def get_majority_vote(row):
    # 현재 행의 모든 예측값 가져오기
    predictions = [row[f'pred_{i}'] for i in range(len(dfs))]
    
    # 가장 많이 나온 값(최빈값) 찾기
    # most_common(1)은 [(값, 빈도수)] 형태를 반환하므로 [0][0]으로 값만 추출
    # 동점일 경우 Counter는 먼저 발견된 요소를 우선함 -> 즉, 리스트 앞쪽 파일에 가중치가 살짝 있는 셈
    vote_result = Counter(predictions).most_common(1)[0][0]
    return vote_result

# 투표 실행
ensemble_df['answer'] = ensemble_df.apply(get_majority_vote, axis=1)

# 4. 최종 결과 저장
final_submission = ensemble_df[['id', 'answer']]
save_path = os.path.join(base_path, "final_ensemble_submission.csv") # 같은 폴더에 저장
final_submission.to_csv(save_path, index=False)

print(f"앙상블 완료! 결과 파일이 저장되었습니다: {save_path}")
print(final_submission.head())