"""Hugging Face utilities"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from huggingface_hub import HfApi, login
from src.config.config import HF_ORG, HF_TOKEN

# Hugging Face organization
HF_ORG = HF_ORG


def get_hf_token() -> Optional[str]:
    """Get Hugging Face token from environment variable"""
    token = os.getenv("HF_TOKEN", HF_TOKEN)
    if not token or token == "your_huggingface_token_here":
        return None
    return token


def login_to_hf(token: Optional[str] = None) -> bool:
    """
    Login to Hugging Face
    
    Args:
        token: Hugging Face token. If None, uses HF_TOKEN from environment
        
    Returns:
        True if login successful, False otherwise
    """
    token = token or get_hf_token()
    if not token:
        raise ValueError("Hugging Face token not found. Please set HF_TOKEN in .env file.")
    
    try:
        login(token=token)
        return True
    except Exception as e:
        print(f"Failed to login to Hugging Face: {e}")
        return False


def upload_model_to_hf(
    checkpoint_path: str,
    model_name: str,
    experiment_name: Optional[str] = None,
    tags: Optional[list] = None,
    **kwargs
) -> str:
    """
    Upload model to Hugging Face Hub
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Model name (e.g., "NLP-07-ODQA/gemma-ko-2b-lora-v1")
        experiment_name: Experiment name for README
        tags: List of tags for the model
        **kwargs: Additional arguments for HfApi.upload_folder
        
    Returns:
        Model repository URL
    """
    if not model_name.startswith(f"{HF_ORG}/"):
        model_name = f"{HF_ORG}/{model_name}"
    
    # Login if needed
    token = get_hf_token()
    if token:
        login_to_hf(token)
    
    api = HfApi()
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Create README if experiment_name is provided
    if experiment_name:
        readme_content = f"""---
tags:
  - korean
  - csat
  - nlp-competition
  - generation-for-nlp
"""
        if tags:
            for tag in tags:
                readme_content += f"  - {tag}\n"
        readme_content += f"""
---
# {model_name}

## Experiment: {experiment_name}

This model was trained for the Generation for NLP competition.

## Model Details

- **Organization**: {HF_ORG}
- **Experiment**: {experiment_name}
- **Checkpoint**: {checkpoint_path.name}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
```
"""
        readme_path = checkpoint_path / "README.md"
        readme_path.write_text(readme_content)
    
    # Upload model
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=model_name,
        repo_type="model",
        **kwargs
    )
    
    repo_url = f"https://huggingface.co/{model_name}"
    print(f"Model uploaded successfully: {repo_url}")
    return repo_url


def load_model_from_hf(model_name: str, **kwargs):
    """
    Load model from Hugging Face Hub
    
    Args:
        model_name: Model name (e.g., "NLP-07-ODQA/gemma-ko-2b-lora-v1")
        **kwargs: Additional arguments for from_pretrained
        
    Returns:
        Model and tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if not model_name.startswith(f"{HF_ORG}/"):
        model_name = f"{HF_ORG}/{model_name}"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    
    return model, tokenizer

