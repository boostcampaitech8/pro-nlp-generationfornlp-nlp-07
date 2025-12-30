import pandas as pd
from pathlib import Path

# 경로 설정
data_path = Path("../../../data/train/train.csv")
output_dir = Path("../../../data/train/train_chunk/")

# 출력 디렉토리 생성
output_dir.mkdir(parents=True, exist_ok=True)

# 데이터 로드
df = pd.read_csv(data_path)
chunk_size = 30

print("=" * 80)
print("데이터 청크 분할")
print("=" * 80)
print(f"원본 파일: {data_path}")
print(f"저장 경로: {output_dir}")
print(f"전체 샘플: {len(df)}개")
print(f"청크 크기: {chunk_size}개")
print("=" * 80)

# 청크 분할 및 저장
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk_num = i // chunk_size + 1
    
    # 파일명: train_chunk1.csv, train_chunk2.csv, ...
    output_file = output_dir / f"train_chunk{chunk_num}.csv"
    chunk.to_csv(output_file, index=False)
    
    print(f"✅ train_chunk{chunk_num}.csv: {len(chunk)} samples")

total_batches = (len(df) - 1) // chunk_size + 1
print("=" * 80)
print(f"✅ 완료! 총 {total_batches}개 청크 파일 생성")
print(f"📁 위치: {output_dir.absolute()}")
