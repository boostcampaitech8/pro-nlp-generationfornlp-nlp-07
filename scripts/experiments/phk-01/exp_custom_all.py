"""
실험 예제: 여러 하이퍼파라미터를 동시에 변경하기

이 예제는 학습률, epoch, LoRA 파라미터 등 여러 설정을 동시에 변경하는 방법을 보여줍니다.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import (
    DEFAULT_MODEL_NAME,
    TRAIN_DATA_PATH,
    RANDOM_SEED,
    MAX_TOKEN_LENGTH,
    TEST_SIZE,
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

# 실험 설정 - 여러 하이퍼파라미터 동시 변경
LEARNING_RATE = 3e-5  # 기본값 2e-5 대신 3e-5
NUM_TRAIN_EPOCHS = 5  # 기본값 3 대신 5
PER_DEVICE_TRAIN_BATCH_SIZE = 2  # 기본값 1 대신 2 (GPU 메모리가 충분한 경우)
WEIGHT_DECAY = 0.05  # 기본값 0.01 대신 0.05

# LoRA 설정
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# 출력 디렉토리
OUTPUT_DIR = project_root / "outputs" / "exp_custom_all"


def main():
    """Main training function with multiple custom hyperparameters"""
    # Set random seed
    set_seed(RANDOM_SEED)
    
    # Load data
    print("Loading data...")
    train_df = load_data(str(TRAIN_DATA_PATH))
    df = flatten_dataset(train_df)
    print(f"Loaded {len(df)} samples")
    
    # Prepare training data
    print("Preparing training data...")
    processed_dataset = prepare_training_data(df)
    dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))
    
    # Load model and tokenizer
    print(f"Loading model: {DEFAULT_MODEL_NAME}")
    model = load_model(DEFAULT_MODEL_NAME)
    tokenizer = load_tokenizer(DEFAULT_MODEL_NAME)
    
    # Setup chat template
    tokenizer = setup_chat_template(tokenizer, model_name=DEFAULT_MODEL_NAME)
    
    # Tokenize dataset
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
    
    # Get LoRA config with custom parameters
    print(f"Creating LoRA config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    peft_config = get_lora_config(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )
    
    # Create trainer with multiple custom hyperparameters
    print(f"Creating trainer with custom settings:")
    print(f"  - learning_rate: {LEARNING_RATE}")
    print(f"  - num_train_epochs: {NUM_TRAIN_EPOCHS}")
    print(f"  - per_device_train_batch_size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  - weight_decay: {WEIGHT_DECAY}")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        output_dir=str(OUTPUT_DIR),
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        # 다른 파라미터는 config 기본값 사용
    )
    
    # Train
    print("Starting training...")
    train(trainer)
    print("Training completed!")
    print(f"Checkpoints saved to: {OUTPUT_DIR}")
    
    # Hugging Face 업로드 예시 (주석 해제하여 사용)
    # from src.utils.hf_utils import upload_model_to_hf
    # model_name = "NLP-07-ODQA/gemma-ko-2b-lora-custom-v1"
    # upload_model_to_hf(
    #     checkpoint_path=str(OUTPUT_DIR),
    #     model_name=model_name,
    #     experiment_name="custom-all-params"
    # )


if __name__ == "__main__":
    main()

