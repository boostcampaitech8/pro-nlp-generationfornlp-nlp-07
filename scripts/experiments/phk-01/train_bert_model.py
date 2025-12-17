"""
팀원A의 BERT 모델 실험 스크립트

이 파일은 다른 모델로 실험하는 방법을 보여주는 예시입니다.
각 팀원은 이 파일을 참고하여 자신만의 실험 스크립트를 만들 수 있습니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import (
    TRAIN_DATA_PATH,
    TEST_SIZE,
    RANDOM_SEED,
    MAX_TOKEN_LENGTH,
)
from src.data.loader import load_data, flatten_dataset
from src.data.preprocessor import prepare_training_data
from src.data.tokenizer import tokenize_dataset, filter_by_length
from src.models.model_loader import load_model, load_tokenizer
from src.models.lora_config import get_lora_config
from src.models.chat_template import setup_chat_template
from src.training.trainer import create_trainer, train
from src.utils.seed import set_seed
from datasets import Dataset
import pandas as pd

# 환경 변수 로드
set_seed(RANDOM_SEED)

# 팀원A의 모델 설정
MODEL_NAME = "klue/bert-base"  # 또는 다른 모델
OUTPUT_DIR = project_root / "outputs" / "bert_memberA"
EXPERIMENT_NAME = "bert-base-lora-v1"


def main():
    """Main training function for memberA's BERT experiment"""
    # 1. 데이터 로드 (공통 모듈 재사용)
    print("Loading data...")
    train_df = load_data(str(TRAIN_DATA_PATH))
    df = flatten_dataset(train_df)
    print(f"Loaded {len(df)} samples")
    
    # 2. 데이터 전처리 (공통 모듈 재사용)
    print("Preparing training data...")
    processed_dataset = prepare_training_data(df)
    dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))
    
    # 3. 모델 로드 (모델명만 변경)
    print(f"Loading model: {MODEL_NAME}")
    model = load_model(MODEL_NAME)
    tokenizer = load_tokenizer(MODEL_NAME)
    
    # Setup chat template (BERT는 다른 template이 필요할 수 있음)
    # tokenizer = setup_chat_template(tokenizer, model_name=MODEL_NAME)
    
    # 4. 토큰화 (공통 모듈 재사용)
    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Filter by length
    print(f"Filtering by max length: {MAX_TOKEN_LENGTH}")
    tokenized_dataset = filter_by_length(tokenized_dataset, MAX_TOKEN_LENGTH)
    
    # Split dataset
    print(f"Splitting dataset (test_size={TEST_SIZE})...")
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    
    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['test']
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # 5. LoRA 설정 (필요시 조정)
    # BERT의 경우 target_modules가 다를 수 있음
    peft_config = get_lora_config(
        r=8,  # 팀원A가 다른 값으로 실험
        lora_alpha=16,
        target_modules=['query', 'key', 'value']  # BERT에 맞게 조정
    )
    
    # 6. 훈련 (공통 모듈 재사용)
    print("Creating trainer...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        output_dir=str(OUTPUT_DIR),
        # 팀원A의 하이퍼파라미터
        learning_rate=3e-5,
        num_train_epochs=5,
    )
    
    # 7. 훈련 실행
    print("Starting training...")
    train(trainer)
    print("Training completed!")
    
    # 8. Hugging Face에 업로드 (선택사항)
    # from src.utils.hf_utils import upload_model_to_hf
    # model_name = f"NLP-07-ODQA/memberA-{EXPERIMENT_NAME}"
    # upload_model_to_hf(
    #     checkpoint_path=str(OUTPUT_DIR),
    #     model_name=model_name,
    #     experiment_name=EXPERIMENT_NAME
    # )


if __name__ == "__main__":
    main()

