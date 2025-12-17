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

    ** 해당 모듈은 trl>0.20.0 버전에서 deprecated되었습니다. **
    trl>0.20.0을 사용하는 경우 해당 모듈을 import 해제하고,
    SFTConfig에 completion_only_loss=True로 설정하세요.
    
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

