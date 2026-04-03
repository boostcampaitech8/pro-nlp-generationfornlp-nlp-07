"""LoRA configuration utilities"""

from peft import LoraConfig
from typing import List, Optional


def get_lora_config(
    r: int = 6,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> LoraConfig:
    """
    Get LoRA configuration
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: List of target module names
        bias: Bias type
        task_type: Task type
        
    Returns:
        LoRA configuration
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj']
    
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )
    
    return peft_config

