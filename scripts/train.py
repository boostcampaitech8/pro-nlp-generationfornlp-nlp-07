"""Training script"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    TRAIN_DATA_PATH,
    RANDOM_SEED,
    MAX_TOKEN_LENGTH,
    TEST_SIZE,
    HF_MODEL_NAME,
    HF_ORG,
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


def main():
    """Main training function"""
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
    
    # Get LoRA config
    peft_config = get_lora_config()
    
    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        output_dir=str(DEFAULT_OUTPUT_DIR),
    )
    
    # Train
    print("Starting training...")
    train(trainer)
    print("Training completed!")
    
    # Upload all checkpoints to Hugging Face (if HF_MODEL_NAME is set)
    if HF_MODEL_NAME:
        try:
            from src.utils.hf_utils import upload_all_checkpoints_to_hf
            
            model_name = f"{HF_ORG}/{HF_MODEL_NAME}"
            print(f"\nUploading all checkpoints to Hugging Face: {model_name}")
            uploaded_urls = upload_all_checkpoints_to_hf(
                output_dir=str(DEFAULT_OUTPUT_DIR),
                model_name=model_name,
                experiment_name=HF_MODEL_NAME,
            )
            if uploaded_urls:
                print(f"\nSuccessfully uploaded {len(uploaded_urls)} checkpoints:")
                for url in uploaded_urls:
                    print(f"  - {url}")
        except Exception as e:
            print(f"\nWarning: Failed to upload to Hugging Face: {e}")
            print("Check your HF_TOKEN in .env file and try again.")
    else:
        print("\nHF_MODEL_NAME not set. Skipping Hugging Face upload.")


if __name__ == "__main__":
    main()

