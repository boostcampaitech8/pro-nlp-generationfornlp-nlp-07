"""
정답 위치 셔플링을 통한 데이터 증강 스크립트

- 4지선다: 4개 버전 생성 (정답을 1, 2, 3, 4번 위치에 배치)
- 5지선다: 5개 버전 생성 (정답을 1, 2, 3, 4, 5번 위치에 배치)
- 2031개 → 약 9,000개로 증강
"""

import pandas as pd
import ast
from tqdm import tqdm


def shuffle_answer_position(problems_dict, target_position):
    """
    정답을 target_position으로 이동시키는 함수

    Args:
        problems_dict: {'question': str, 'choices': list, 'answer': int}
        target_position: 정답을 이동시킬 위치 (1-based, 1~4 또는 1~5)

    Returns:
        새로운 problems_dict
    """
    question = problems_dict['question']
    choices = problems_dict['choices'].copy()
    current_answer = problems_dict['answer']  # 1-based index

    # 현재 정답 위치 (0-based로 변환)
    current_idx = current_answer - 1
    target_idx = target_position - 1

    # 정답 선택지 추출
    correct_choice = choices[current_idx]

    # 정답을 제거하고 나머지 선택지만 남김
    other_choices = choices[:current_idx] + choices[current_idx+1:]

    # target_idx 위치에 정답 삽입
    new_choices = other_choices[:target_idx] + [correct_choice] + other_choices[target_idx:]

    return {
        'question': question,
        'choices': new_choices,
        'answer': target_position  # 1-based
    }


def augment_dataset(input_csv, output_csv):
    """
    전체 데이터셋에 대해 정답 위치 셔플링 수행

    Args:
        input_csv: 입력 CSV 파일 경로
        output_csv: 출력 CSV 파일 경로
    """
    print(f"데이터 로드 중: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"원본 데이터: {len(df)}개")

    augmented_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="정답 위치 셔플링"):
        # problems 컬럼 파싱
        problems_dict = ast.literal_eval(row['problems'])

        num_choices = len(problems_dict['choices'])

        # 각 위치에 정답을 배치한 버전 생성
        for target_pos in range(1, num_choices + 1):
            new_problems = shuffle_answer_position(problems_dict, target_pos)

            # 새로운 행 생성
            new_row = row.copy()
            new_row['problems'] = str(new_problems)

            # id에 위치 정보 추가
            new_row['id'] = f"{row['id']}_ans{target_pos}"

            augmented_data.append(new_row)

    # 새 데이터프레임 생성
    augmented_df = pd.DataFrame(augmented_data)

    print(f"\n증강 완료: {len(df)} → {len(augmented_df)}개 ({len(augmented_df)/len(df):.1f}배)")

    # 통계 출력
    print("\n=== 증강 통계 ===")
    original_4choice = sum(1 for idx in range(len(df)) 
                          if len(ast.literal_eval(df.iloc[idx]['problems'])['choices']) == 4)
    original_5choice = len(df) - original_4choice

    print(f"원본 4지선다: {original_4choice}개 → {original_4choice * 4}개")
    print(f"원본 5지선다: {original_5choice}개 → {original_5choice * 5}개")

    # 저장
    print(f"\n저장 중: {output_csv}")
    augmented_df.to_csv(output_csv, index=False)
    print("완료!")

    return augmented_df


def verify_augmentation(df, num_samples=3):
    """
    증강 결과 검증 (원본 샘플 하나의 모든 버전 확인)

    Args:
        df: 증강된 데이터프레임
        num_samples: 확인할 샘플 수
    """
    print("\n=== 증강 결과 검증 ===")

    # 원본 id 추출 (예: generation-for-nlp-425_ans1 → generation-for-nlp-425)
    df['original_id'] = df['id'].str.rsplit('_', n=1).str[0]

    # 첫 num_samples개의 원본 id 선택
    original_ids = df['original_id'].unique()[:num_samples]

    for orig_id in original_ids:
        print(f"\n--- {orig_id} ---")
        samples = df[df['original_id'] == orig_id]

        for idx, row in samples.iterrows():
            problems = ast.literal_eval(row['problems'])
            print(f"  {row['id']}: 정답 {problems['answer']}번 - {problems['choices'][problems['answer']-1]}")


if __name__ == "__main__":
    import argparse

    ### 경로에 맞게 수정
    dataPath = "/data/ephemeral/home/T8091/phk-04/data/train"

    parser = argparse.ArgumentParser(description="정답 위치 셔플링을 통한 데이터 증강")
    parser.add_argument("--input", type=str, default="train.csv", 
                       help="입력 CSV 파일 경로 (기본값: train.csv)")
    parser.add_argument("--output", type=str, default="train_augmented.csv",
                       help="출력 CSV 파일 경로 (기본값: train_augmented.csv)")
    parser.add_argument("--verify", action="store_true",
                       help="증강 결과 검증 수행")

    args = parser.parse_args()

    # 데이터 증강 실행
    augmented_df = augment_dataset(f"{dataPath}/{args.input}", f"{dataPath}/{args.output}")

    # 검증 (옵션)
    if args.verify:
        verify_augmentation(augmented_df, num_samples=5)