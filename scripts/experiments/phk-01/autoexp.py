"""
MemberA's Qwen/Llama Experiment Script

This script performs both training and inference in a single pipeline.
It is configured for Qwen/Llama based models.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset

# Add project root to path
# Assuming this script is in scripts/experiments/memberA/
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TEST_SIZE,
    RANDOM_SEED,
    MAX_TOKEN_LENGTH,
    HF_ORG,
)
from src.data.loader import load_data, flatten_dataset
from src.data.preprocessor import prepare_training_data, prepare_inference_data
from src.data.tokenizer import tokenize_dataset, filter_by_length
from src.models.model_loader import load_model, load_tokenizer, load_checkpoint
from src.models.lora_config import get_lora_config
from src.models.chat_template import setup_chat_template
from src.training.trainer import create_trainer, train
from src.inference.predictor import predict_batch
from src.inference.submission import create_submission
from src.utils.seed import set_seed

# Defaults (can be overridden by CLI args)
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507" 
DEFAULT_EXPERIMENT_NAME = "qwen3-4b-lora-v1"

# LoRA Target Modules for Qwen/Llama
# Common targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen/Llama Experiment")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model name")
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME, help="Experiment name")
    return parser.parse_args()

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()
    MODEL_NAME = args.model_name
    EXPERIMENT_NAME = args.experiment_name
    OUTPUT_DIR = project_root / "outputs" / "T8091"
    
    # 1. Setup
    print(f"Starting Experiment: {EXPERIMENT_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Training Phase
    # ---------------------------------------------------------
    print("\n=== Starting Training Phase ===")
    
    # Load Data
    print("Loading training data...")
    train_df = load_data(str(TRAIN_DATA_PATH))
    df = flatten_dataset(train_df)
    print(f"Loaded {len(df)} samples")

    # Preprocess
    print("Preparing training data...")
    processed_dataset = prepare_training_data(df)
    dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))

    # Load Model & Tokenizer
    print(f"Loading model: {MODEL_NAME}")
    model = load_model(MODEL_NAME)
    tokenizer = load_tokenizer(MODEL_NAME)

    # Setup Chat Template (Critical for Instruct models)
    print("Setting up chat template...")
    tokenizer = setup_chat_template(tokenizer, model_name=MODEL_NAME)

    # Tokenize
    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Filter
    print(f"Filtering by max length: {MAX_TOKEN_LENGTH}")
    tokenized_dataset = filter_by_length(tokenized_dataset, MAX_TOKEN_LENGTH)

    # Split (incorporating validation as per train.py)
    print(f"Splitting dataset (test_size={TEST_SIZE})...")
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    
    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['test']
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # LoRA Config
    print(f"Configuring LoRA with targets: {LORA_TARGET_MODULES}")
    peft_config = get_lora_config(
        target_modules=LORA_TARGET_MODULES,
        r=16,           # Increased R for better expressivity with larger models
        lora_alpha=32,
        lora_dropout=0.05
    )

    # Create Trainer
    print("Creating trainer...")
    # Passing custom hyperparameters here
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-4,  # QLoRA/LoRA often uses higher LR like 2e-4
        num_train_epochs=3,   # Adjustable
        per_device_train_batch_size=2, # Adjust based on GPU memory
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4, # Simulate larger batch size
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    # Run Training
    print("Starting training...")
    train(trainer)
    print("Training phase completed!")
    
    # Run Validation explicitly to show metrics
    print("Running final validation...")
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    # Free up memory (optional but good practice before inference if in same process)
    # import torch
    # del model, trainer
    # torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Hugging Face Upload Phase
    # ---------------------------------------------------------
    print("\n=== Starting Hugging Face Upload Phase ===")
    try:
        from src.utils.hf_utils import upload_all_checkpoints_to_hf
        
        # Use EXPERIMENT_NAME as the HF model name suffix (repo name)
        hf_model_name = f"{HF_ORG}/{EXPERIMENT_NAME}"
        print(f"Uploading checkpoints to Hugging Face: {hf_model_name}")
        
        uploaded_urls = upload_all_checkpoints_to_hf(
            output_dir=str(OUTPUT_DIR),
            model_name=hf_model_name,
            experiment_name=EXPERIMENT_NAME,
        )
        
        if uploaded_urls:
            print(f"\nSuccessfully uploaded {len(uploaded_urls)} checkpoints:")
            for url in uploaded_urls:
                print(f"  - {url}")
        else:
            print("\nNo checkpoints generated or upload failed.")
            
    except Exception as e:
        print(f"\nWarning: Failed to upload to Hugging Face: {e}")
        print("ensure HF_TOKEN is set in .env and you have write access to the org.")

    # ---------------------------------------------------------
    # Inference Phase
    # ---------------------------------------------------------
    print("\n=== Starting Inference Phase ===")

    # Find the best checkpoint (Saved by trainer at the end or explicitly)
    # Since load_best_model_at_end=True, the 'checkpoint-X' in output_dir 
    # might be slightly confusing if save_total_limit is small, 
    # but normally the best model weights are loaded in trainer.model.
    # However, to use the 'inference.py' flow which uses `load_checkpoint`, 
    # we should find the checkpoint directory.
    
    checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
    if not checkpoints:
         print(f"Warning: No checkpoints found in {OUTPUT_DIR}. Using current model state logic might be needed but for safety stopping.")
         # In a real scenario, could try to use trainer.save_model() to save the final state explicitly.
         # But trainer usually saves checkpoints.
         return

    best_checkpoint = checkpoints[-1] # Usually the latest if simple sort, or need parsing.
    # Since load_best_model_at_end=True, the last checkpoint might not be the best per se 
    # if we continued training, but with save_total_limit=1 and epoch strategy, it is likely the relevant one.
    # Alternatively, use 'best_model' dir if it exists (trainer implementation detail).
    
    best_model_dir = OUTPUT_DIR / "best_model"
    if best_model_dir.exists():
        checkpoint_path = best_model_dir
        print(f"Using best model from: {checkpoint_path}")
    else:
        checkpoint_path = best_checkpoint
        print(f"Using latest checkpoint: {checkpoint_path}")

    # Load Model for Inference
    # Note: We need to reload to ensure clean state and correct inference mode, 
    # though technically we could use `trainer.model`. 
    # Sticking to `inference.py` pattern for robustness.
    print(f"Loading checkpoint for inference: {checkpoint_path}")
    model, tokenizer = load_checkpoint(str(checkpoint_path), device_map="cuda")

    # Load Test Data
    print(f"Loading test data: {TEST_DATA_PATH}")
    test_df = load_data(str(TEST_DATA_PATH))
    test_df = flatten_dataset(test_df)
    print(f"Loaded {len(test_df)} test samples")

    # Prepare Inference Data
    print("Preparing inference data...")
    test_dataset = prepare_inference_data(test_df)

    # Predict
    print("Running prediction...")
    predictions = predict_batch(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        device="cuda",
        show_progress=True
    )

    # Save Submission
    submission_path = OUTPUT_DIR / f"{EXPERIMENT_NAME}.csv"
    print(f"Saving submission to: {submission_path}")
    create_submission(predictions, str(submission_path))
    
    print("\n=== Experiment Completed Successfully ===")
    print(f"Submission available at: {submission_path}")


if __name__ == "__main__":
    main()
