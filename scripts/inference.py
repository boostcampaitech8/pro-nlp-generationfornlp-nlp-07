"""Inference script"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import TEST_DATA_PATH, DEFAULT_OUTPUT_DIR
from src.data.loader import load_data, flatten_dataset
from src.data.preprocessor import prepare_inference_data
from src.models.model_loader import load_checkpoint
from src.inference.predictor import predict_batch
from src.inference.submission import create_submission
import argparse


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Run inference on test data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (default: latest checkpoint in output_dir)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=str(TEST_DATA_PATH),
        help="Path to test data CSV file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint is None:
        # Find latest checkpoint
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {output_dir}")
        checkpoint_path = Path(checkpoints[-1])
        print(f"Using latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint)
    
    # Determine submission path based on checkpoint name
    checkpoint_name = checkpoint_path.name  # e.g., "checkpoint-1000" or "best_model"
    submission_dir = project_root / "submissions" / checkpoint_name
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / "output.csv"
    
    # Load model and tokenizer
    print(f"Loading checkpoint: {checkpoint_path}")
    model, tokenizer = load_checkpoint(str(checkpoint_path), device_map=args.device)
    
    # Load test data
    print(f"Loading test data: {args.test_data}")
    test_df = load_data(args.test_data)
    test_df = flatten_dataset(test_df)
    print(f"Loaded {len(test_df)} test samples")
    
    # Prepare inference data
    print("Preparing inference data...")
    test_dataset = prepare_inference_data(test_df)
    
    # Run inference
    print("Running inference...")
    predictions = predict_batch(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        device=args.device,
        show_progress=True
    )
    
    # Create submission file
    print(f"Creating submission file: {submission_path}")
    create_submission(predictions, str(submission_path))
    print(f"Inference completed! Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()

