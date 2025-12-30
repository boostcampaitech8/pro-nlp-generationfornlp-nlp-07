import pandas as pd


DATA_PATH = "../../../data/train/"
# ========== 설정 부분 ==========
# 입력 파일 경로
INPUT_FILE = f"{DATA_PATH}/train.csv"

# 출력 파일 경로
OUTPUT_FILE = f"{DATA_PATH}/train_clean.csv"

# 제외할 ID 리스트
EXCLUDE_IDS = [
    'generation-for-nlp-498', 'generation-for-nlp-502', 'generation-for-nlp-510',
    'generation-for-nlp-526', 'generation-for-nlp-530', 'generation-for-nlp-531',
    'generation-for-nlp-538', 'generation-for-nlp-556', 'generation-for-nlp-578',
    'generation-for-nlp-682', 'generation-for-nlp-728', 'generation-for-nlp-757',
    'generation-for-nlp-934', 'generation-for-nlp-940', 'generation-for-nlp-969',
    'generation-for-nlp-986', 'generation-for-nlp-991', 'generation-for-nlp-1088',
    'generation-for-nlp-1191', 'generation-for-nlp-1209', 'generation-for-nlp-1333',
    'generation-for-nlp-1367', 'generation-for-nlp-1398', 'generation-for-nlp-1548',
    'generation-for-nlp-1555', 'generation-for-nlp-1563', 'generation-for-nlp-1621',
    'generation-for-nlp-1717', 'generation-for-nlp-1898', 'generation-for-nlp-1908',
    'generation-for-nlp-1938', 'generation-for-nlp-1943', 'generation-for-nlp-2008',
    'generation-for-nlp-2090', 'generation-for-nlp-2125', 'generation-for-nlp-2443',
    'generation-for-nlp-2453', 'generation-for-nlp-2532', 'generation-for-nlp-2541',
    'generation-for-nlp-2547', 'generation-for-nlp-2563', 'generation-for-nlp-2600',
    'generation-for-nlp-2625', 'generation-for-nlp-2681', 'generation-for-nlp-2768',
    'generation-for-nlp-2776', 'generation-for-nlp-2779', 'generation-for-nlp-2827',
    'generation-for-nlp-2831', 'generation-for-nlp-2843', 'generation-for-nlp-2870',
    'generation-for-nlp-2878'
]
# ===============================


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("CSV 필터링 스크립트 시작")
    print("=" * 60)

    # CSV 파일 읽기
    print(f"\n입력 파일 읽는 중: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # 원본 데이터 정보
    original_count = len(df)
    print(f"원본 데이터 개수: {original_count}")

    # 제외할 ID 정보
    print(f"\n제외할 ID 개수: {len(EXCLUDE_IDS)}")
    if len(EXCLUDE_IDS) <= 10:
        print(f"제외할 ID: {EXCLUDE_IDS}")
    else:
        print(f"제외할 ID (처음 10개): {EXCLUDE_IDS[:10]}")
        print(f"... 외 {len(EXCLUDE_IDS) - 10}개")

    # 특정 ID를 제외한 데이터 필터링
    print(f"\n필터링 중...")
    df_filtered = df[~df['id'].isin(EXCLUDE_IDS)]

    # 필터링 결과
    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count

    print(f"\n필터링 후 데이터 개수: {filtered_count}")
    print(f"실제 제거된 데이터 개수: {removed_count}")

    # 새로운 CSV 파일로 저장
    print(f"\n저장 중: {OUTPUT_FILE}")
    df_filtered.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("=" * 60)
    print("✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()