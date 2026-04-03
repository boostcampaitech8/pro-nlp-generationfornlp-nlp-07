# 실험 스크립트 가이드

이 폴더는 다양한 하이퍼파라미터 실험을 위한 스크립트들을 포함합니다.

## 실험 스크립트 목록

### 1. `memberA/train_bert_model.py`
- **목적**: 다른 모델(BERT)로 실험하기
- **변경 사항**: 모델명 변경 및 모델별 설정 조정
- **실행**: `python scripts/experiments/memberA/train_bert_model.py`

### 2. `memberB/exp_custom_all.py`
- **목적**: 여러 하이퍼파라미터를 동시에 변경하기 (예제)
- **변경 사항**: 학습률, epoch, batch size, weight decay, LoRA 파라미터 등
- **실행**: `python scripts/experiments/memberB/exp_custom_all.py`

## 실험 스크립트 작성 방법

### 기본 구조

```python
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 필요한 import
from src.config.config import (
    DEFAULT_MODEL_NAME, TRAIN_DATA_PATH, RANDOM_SEED,
    MAX_TOKEN_LENGTH, TEST_SIZE
)
# ... 기타 import ...

# 실험 설정 (config 기본값 오버라이드)
LEARNING_RATE = 5e-5  # 예시
OUTPUT_DIR = project_root / "outputs" / "exp_name"

def main():
    # 1. 데이터 로딩
    # 2. 모델 로딩
    # 3. LoRA config 설정
    # 4. Trainer 생성 (하이퍼파라미터 오버라이드)
    # 5. 학습 실행

if __name__ == "__main__":
    main()
```

### 오버라이드 가능한 파라미터

#### Trainer 파라미터
- `learning_rate`: 학습률
- `num_train_epochs`: 학습 epoch 수
- `per_device_train_batch_size`: 배치 크기
- `weight_decay`: Weight decay
- `lr_scheduler_type`: Learning rate scheduler 타입
- 기타 `create_trainer()` 함수의 파라미터

#### LoRA 파라미터
- `r`: LoRA rank
- `lora_alpha`: LoRA alpha
- `lora_dropout`: LoRA dropout
- `target_modules`: 타겟 모듈 리스트

### 출력 디렉토리

각 실험은 고유한 출력 디렉토리를 사용해야 합니다:
```python
OUTPUT_DIR = project_root / "outputs" / "exp_unique_name"
```

## 실행 방법

```bash
# 가상환경 활성화
source .venv/bin/activate

# 실험 실행
python scripts/experiments/memberB/exp_custom_all.py
```

## Hugging Face 업로드

학습 완료 후 모델을 Hugging Face에 업로드하려면:

```python
from src.utils.hf_utils import upload_model_to_hf

upload_model_to_hf(
    checkpoint_path=str(OUTPUT_DIR),
    model_name="NLP-07-ODQA/your-model-name-v1",
    experiment_name="your-experiment-name"
)
```

## 주의사항

1. 각 실험은 고유한 `OUTPUT_DIR`을 사용해야 합니다
2. 실험명은 명확하고 구분 가능하게 작성하세요
3. 변경한 하이퍼파라미터를 주석으로 명시하세요
4. 실험 결과는 `experiments/` 폴더에 기록하세요

