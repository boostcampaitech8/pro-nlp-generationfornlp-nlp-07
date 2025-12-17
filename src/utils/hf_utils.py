"""Hugging Face utilities"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from huggingface_hub import HfApi, login
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
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
    Upload model to Hugging Face Hub (main branch)
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Model name (e.g., "NLP-07-ODQA/gemma-ko-2b-lora-v1")
        experiment_name: Experiment name for README
        tags: List of tags for the model
        **kwargs: Additional arguments
        
    Returns:
        Model repository URL
    """
    if not model_name.startswith(f"{HF_ORG}/"):
        model_name = f"{HF_ORG}/{model_name}"
    
    # Login if needed
    token = get_hf_token()
    if token:
        login_to_hf(token)
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Create repository if it doesn't exist
    create_repo_if_not_exists(model_name)
    
    # Load model and tokenizer from checkpoint
    print(f"Loading model and tokenizer from {checkpoint_path}...")
    model = AutoPeftModelForCausalLM.from_pretrained(str(checkpoint_path))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    
    # Load best model info if exists (원본 checkpoint 정보)
    original_checkpoint = None
    best_model_info_path = checkpoint_path / "best_model_info.json"
    if best_model_info_path.exists():
        import json
        try:
            with open(best_model_info_path, 'r') as f:
                best_model_info = json.load(f)
                original_checkpoint = best_model_info.get("original_checkpoint")
        except Exception:
            pass
    
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
"""
        if original_checkpoint:
            readme_content += f"- **Original Checkpoint**: {original_checkpoint} (This is the best model selected during training)\n"
        
        readme_content += f"""
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
```
"""
        readme_path = checkpoint_path / "README.md"
        readme_path.write_text(readme_content)
    
    # Upload model and tokenizer to main branch using push_to_hub
    print(f"Uploading model and tokenizer to main branch...")
    model.push_to_hub(
        repo_id=model_name,
        private=False,
    )
    tokenizer.push_to_hub(
        repo_id=model_name,
    )
    
    # Upload README if exists
    if (checkpoint_path / "README.md").exists():
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(checkpoint_path / "README.md"),
            path_in_repo="README.md",
            repo_id=model_name,
            repo_type="model",
            revision="main"
        )
    
    repo_url = f"https://huggingface.co/{model_name}"
    print(f"Model uploaded successfully to main branch: {repo_url}")
    return repo_url


def create_repo_if_not_exists(
    model_name: str,
    private: bool = False
) -> bool:
    """
    Create Hugging Face repository if it doesn't exist
    
    Args:
        model_name: Model name (e.g., "NLP-07-ODQA/gemma-ko-2b-lora-v1")
        private: Whether the repository should be private
        
    Returns:
        True if repository was created or already exists, False otherwise
    """
    if not model_name.startswith(f"{HF_ORG}/"):
        model_name = f"{HF_ORG}/{model_name}"
    
    # Login if needed
    token = get_hf_token()
    if token:
        login_to_hf(token)
    
    api = HfApi()
    
    try:
        # Check if repo exists
        api.model_info(model_name)
        return True
    except Exception:
        # Repository doesn't exist, create it
        try:
            api.create_repo(
                repo_id=model_name,
                repo_type="model",
                private=private,
                exist_ok=True
            )
            print(f"Created repository: {model_name}")
            return True
        except Exception as e:
            print(f"Failed to create repository {model_name}: {e}")
            return False


def upload_model_to_hf_with_branch(
    checkpoint_path: str,
    model_name: str,
    branch_name: str,
    experiment_name: Optional[str] = None,
    tags: Optional[list] = None,
    **kwargs
) -> str:
    """
    Upload model to Hugging Face Hub with a specific branch
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Model name (e.g., "NLP-07-ODQA/gemma-ko-2b-lora-v1")
        branch_name: Branch name (e.g., "checkpoint-1000")
        experiment_name: Experiment name for README
        tags: List of tags for the model
        **kwargs: Additional arguments for HfApi.upload_folder
        
    Returns:
        Model repository URL with branch
    """
    if not model_name.startswith(f"{HF_ORG}/"):
        model_name = f"{HF_ORG}/{model_name}"
    
    # Login if needed
    token = get_hf_token()
    if token:
        login_to_hf(token)
    
    # Create repository if it doesn't exist
    create_repo_if_not_exists(model_name)
    
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
- **Branch**: {branch_name}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}", revision="{branch_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}", revision="{branch_name}")
```
"""
        readme_path = checkpoint_path / "README.md"
        readme_path.write_text(readme_content)
    
    # Create branch first if it doesn't exist (참고 코드 방식)
    print(f"Creating branch '{branch_name}' if it doesn't exist...")
    try:
        api.create_branch(
            repo_id=model_name,
            repo_type="model",
            branch=branch_name,
            exist_ok=True,  # 이미 존재하면 에러 발생하지 않음
        )
        print(f"Branch '{branch_name}' created or already exists")
    except Exception as branch_error:
        # 브랜치가 이미 존재하거나 다른 이유로 실패할 수 있음
        print(f"Branch {branch_name} may already exist or creation skipped: {branch_error}")
    
    # Upload checkpoint to branch
    print(f"Uploading checkpoint to branch '{branch_name}'...")
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=model_name,
        repo_type="model",
        revision=branch_name,  # 생성한 브랜치에 업로드
        commit_message=f"Upload {checkpoint_path.name} checkpoint",
        **kwargs
    )
    
    repo_url = f"https://huggingface.co/{model_name}/tree/{branch_name}"
    print(f"Model uploaded successfully to branch '{branch_name}': {repo_url}")
    return repo_url


def upload_all_checkpoints_to_hf(
    output_dir: str,
    model_name: str,
    experiment_name: Optional[str] = None,
    tags: Optional[list] = None,
    upload_best_to_main: bool = True,
) -> list:
    """
    Upload all checkpoints to Hugging Face Hub, each as a separate branch
    
    Args:
        output_dir: Output directory containing checkpoints
        model_name: Model name (e.g., "NLP-07-ODQA/gemma-ko-2b-lora-v1")
        experiment_name: Experiment name for README
        tags: List of tags for the model
        
    Returns:
        List of uploaded branch URLs
    """
    if not model_name.startswith(f"{HF_ORG}/"):
        model_name = f"{HF_ORG}/{model_name}"
    
    # Create repository if it doesn't exist
    if not create_repo_if_not_exists(model_name):
        print(f"Failed to create repository. Skipping upload.")
        return []
    
    output_path = Path(output_dir)
    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    # Find all checkpoints
    checkpoints = sorted(
        output_path.glob("checkpoint-*"),
        key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0
    )
    
    if not checkpoints:
        print(f"No checkpoints found in {output_dir}")
        return []
    
    # Login if needed
    token = get_hf_token()
    if token:
        login_to_hf(token)
    
    api = HfApi()
    
    # 먼저 main 브랜치에 초기 README 파일 업로드 (빈 리포지토리에 브랜치 생성 불가 문제 해결)
    try:
        # Check if main branch has any files
        try:
            api.model_info(model_name, revision="main")
            # If we get here, main branch exists, check if it has files
            repo_info = api.repo_info(model_name, repo_type="model")
            if repo_info.sha is None:
                # Empty repo, need to initialize
                print("Initializing repository with README on main branch...")
                initial_readme = f"""---
tags:
  - korean
  - csat
  - nlp-competition
  - generation-for-nlp
---
# {model_name}

## Experiment: {experiment_name or 'Training Experiment'}

This repository contains checkpoints from the Generation for NLP competition training.

Each checkpoint is stored in a separate branch.

## Available Checkpoints

"""
                # Upload initial README to main branch
                from io import BytesIO
                api.upload_file(
                    path_or_fileobj=BytesIO(initial_readme.encode()),
                    path_in_repo="README.md",
                    repo_id=model_name,
                    repo_type="model",
                    revision="main"
                )
                print("Repository initialized successfully.")
        except Exception:
            # Repository might be empty, initialize it
            print("Initializing repository with README on main branch...")
            initial_readme = f"""---
tags:
  - korean
  - csat
  - nlp-competition
  - generation-for-nlp
---
# {model_name}

## Experiment: {experiment_name or 'Training Experiment'}

This repository contains checkpoints from the Generation for NLP competition training.

Each checkpoint is stored in a separate branch.

## Available Checkpoints

"""
            from io import BytesIO
            api.upload_file(
                path_or_fileobj=BytesIO(initial_readme.encode()),
                path_in_repo="README.md",
                repo_id=model_name,
                repo_type="model",
                revision="main"
            )
            print("Repository initialized successfully.")
    except Exception as e:
        print(f"Warning: Failed to initialize repository: {e}")
        # Continue anyway, might work if repo already has content
    
    # Find best checkpoint and upload to main branch FIRST
    # (main 브랜치에 파일이 있어야 다른 브랜치를 생성할 수 있음)
    best_checkpoint = None
    main_uploaded = False
    if upload_best_to_main:
        # best_model 폴더 확인 (학습 중에 SaveBestModelCallback이 저장한 best model)
        best_model_dir = output_path / "best_model"
        if best_model_dir.exists() and best_model_dir.is_dir():
            best_checkpoint = best_model_dir
            print(f"\nFound best model directory: {best_model_dir}")
        else:
            print(f"\nWarning: best_model directory not found at {best_model_dir}")
            print("Best model should be saved automatically during training by SaveBestModelCallback.")
            print("Skipping best model upload to main branch.")
        
        if best_checkpoint:
            print(f"\nUploading best model ({best_checkpoint.name}) to main branch...")
            try:
                # Upload best model to main branch
                url = upload_model_to_hf(
                    checkpoint_path=str(best_checkpoint),
                    model_name=model_name,
                    experiment_name=experiment_name,
                    tags=tags,
                )
                print(f"Best model uploaded to main branch: {url}")
                main_uploaded = True
                # 잠시 대기 (업로드 완료 확인)
                import time
                time.sleep(2)
            except Exception as e:
                print(f"Failed to upload best model to main branch: {e}")
                print("Cannot create branches without main branch content. Skipping branch uploads.")
                return []
    
    # Main 브랜치에 파일이 있어야 브랜치를 생성할 수 있음
    if not main_uploaded:
        print("\nWarning: Main branch upload failed or skipped. Cannot create branches.")
        print("Branches can only be created after main branch has content.")
        return []
    
    uploaded_urls = []
    print(f"\nFound {len(checkpoints)} checkpoints. Uploading to Hugging Face branches...")
    
    for checkpoint in checkpoints:
        # Skip best checkpoint if it was already uploaded to main
        if upload_best_to_main and best_checkpoint:
            # best_model 폴더와 checkpoint 경로 비교
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.name == best_checkpoint.name or str(checkpoint_path) == str(best_checkpoint):
                print(f"Skipping {checkpoint.name} (already on main branch)")
                continue
            
        branch_name = checkpoint.name  # e.g., "checkpoint-1000"
        try:
            url = upload_model_to_hf_with_branch(
                checkpoint_path=str(checkpoint),
                model_name=model_name,
                branch_name=branch_name,
                experiment_name=experiment_name,
                tags=tags,
            )
            uploaded_urls.append(url)
        except Exception as e:
            print(f"Failed to upload {checkpoint.name}: {e}")
            continue
    
    print(f"\nUploaded {len(uploaded_urls)} checkpoints to Hugging Face branches")
    if best_checkpoint and upload_best_to_main:
        print(f"Best model is available on main branch")
    return uploaded_urls


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

