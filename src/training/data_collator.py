"""Data collator utilities"""

from trl import DataCollatorForCompletionOnlyLM
from transformers import PreTrainedTokenizer
from src.config.config import RESPONSE_TEMPLATE


def get_data_collator(
    tokenizer: PreTrainedTokenizer,
    response_template: str = None
) -> DataCollatorForCompletionOnlyLM:
    """
    Get data collator for completion-only language modeling
    
    Args:
        tokenizer: Tokenizer to use
        response_template: Response template string
        
    Returns:
        Data collator
    """
    if response_template is None:
        response_template = RESPONSE_TEMPLATE
    
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    
    return data_collator

