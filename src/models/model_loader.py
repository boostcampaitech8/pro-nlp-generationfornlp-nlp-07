"""Model loading utilities"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from typing import Optional, Union


def load_model(
    model_name: str,
    torch_dtype: Optional[torch.dtype] = torch.float16,
    trust_remote_code: bool = True,
    device_map: Optional[Union[str, dict]] = None,
    **kwargs
):
    """
    Load model from Hugging Face
    
    Args:
        model_name: Model name or path
        torch_dtype: Torch data type
        trust_remote_code: Whether to trust remote code
        device_map: Device mapping for model
        **kwargs: Additional arguments for from_pretrained
        
    Returns:
        Loaded model
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        **kwargs
    )
    return model


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    **kwargs
):
    """
    Load tokenizer from Hugging Face
    
    Args:
        model_name: Model name or path
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments for from_pretrained
        
    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    return tokenizer


def load_checkpoint(
    checkpoint_path: str,
    trust_remote_code: bool = True,
    device_map: Optional[Union[str, dict]] = "auto",
    **kwargs
):
    """
    Load model and tokenizer from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint directory
        trust_remote_code: Whether to trust remote code
        device_map: Device mapping for model
        **kwargs: Additional arguments for from_pretrained
        
    Returns:
        Tuple of (model, tokenizer)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        **kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_path),
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    return model, tokenizer

