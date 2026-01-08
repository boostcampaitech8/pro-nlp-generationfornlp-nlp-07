"""
Upload trained models to Hugging Face Hub
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
# Assuming this script is in scripts/experiments/phk-01/
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import HF_ORG
from src.utils.hf_utils import upload_all_checkpoints_to_hf

def parse_args():
    parser = argparse.ArgumentParser(description="Upload checkpoints to Hugging Face")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name (Required)")
    parser.add_argument("--hf_org", type=str, default=HF_ORG, help="Hugging Face Organization")
    return parser.parse_args()

def main():
    args = parse_args()
    EXPERIMENT_NAME = args.experiment_name
    HF_ORG_NAME = args.hf_org
    
    # Path setup
    OUTPUT_DIR = project_root / "outputs" / "T8091" / EXPERIMENT_NAME
    
    # Check if output directory exists
    if not OUTPUT_DIR.exists():
        print(f"Error: Output directory not found: {OUTPUT_DIR}")
        return

    print(f"Starting Upload for Experiment: {EXPERIMENT_NAME}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Target Org: {HF_ORG_NAME}")

    try:
        # Use EXPERIMENT_NAME as the HF model name suffix (repo name)
        hf_model_name = f"{HF_ORG_NAME}/{EXPERIMENT_NAME}"
        print(f"Uploading checkpoints to Hugging Face: {hf_model_name}")
        
        uploaded_urls = upload_all_checkpoints_to_hf(
            output_dir=str(OUTPUT_DIR),
            model_name=hf_model_name,
            experiment_name=EXPERIMENT_NAME,
            upload_best_to_main=True
        )
        
        if uploaded_urls:
            print(f"\nSuccessfully uploaded {len(uploaded_urls)} branch checkpoints:")
            for url in uploaded_urls:
                print(f"  - {url}")
        else:
            print("\nNote: Only main branch (best model) might have been uploaded, or no checkpoints found.")
            
    except Exception as e:
        print(f"\nError: Failed to upload to Hugging Face: {e}")
        print("Please ensure HF_TOKEN is set in .env and you have write access to the org.")

if __name__ == "__main__":
    main()
