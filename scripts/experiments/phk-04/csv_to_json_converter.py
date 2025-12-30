import pandas as pd
import json

FILE_NAME = 'probs_NLP-07-ODQA_Qwen2.5-32B-Instruct-bnb-4bit_15.csv'

# 1. 파일 읽기
csv_df = pd.read_csv(FILE_NAME)

# JSON 파일에서 num_choices 정보 로드
with open('../../../submissions/qwen3-32b-qlora-v6.1_detailed.json', 'r') as f:
    json_data = json.load(f)

# 2. JSON을 id를 키로 하는 딕셔너리로 변환
json_dict = {pred['id']: pred for pred in json_data['predictions']}

# 3. CSV 행을 JSON 형식으로 변환하는 함수
def csv_row_to_json(row):
    sample_id = row['id']

    # JSON에서 num_choices 정보 가져오기 (없으면 기본값 5)
    num_choices = json_dict.get(sample_id, {}).get('num_choices', 5)

    # probabilities 딕셔너리 생성
    probabilities = {}
    for i in range(1, num_choices + 1):
        probabilities[str(i)] = float(row[f'p{i}'])

    # JSON 구조 생성
    return {
        "id": sample_id,
        "prediction": str(int(row['pred_base'])),
        "probabilities": probabilities,
        "confidence": float(row['top1p']),
        "num_choices": num_choices
    }

# 4. 전체 데이터 변환
predictions = [csv_row_to_json(row) for _, row in csv_df.iterrows()]

# 5. 최종 JSON 구조
output = {
    "experiment_name": FILE_NAME.split('.csv')[0],
    "total_samples": len(predictions),
    "predictions": predictions
}

# 6. JSON 파일로 저장
with open(FILE_NAME.split('.csv')[0] + '.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"변환 완료: {len(predictions)}개 샘플")
