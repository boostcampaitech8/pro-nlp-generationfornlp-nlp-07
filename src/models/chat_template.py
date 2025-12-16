"""Chat template utilities"""

from transformers import PreTrainedTokenizer
from typing import Optional


# Gemma chat template
GEMMA_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"


def setup_chat_template(
    tokenizer: PreTrainedTokenizer,
    template: Optional[str] = None,
    model_name: Optional[str] = None
) -> PreTrainedTokenizer:
    """
    Setup chat template for tokenizer
    
    Args:
        tokenizer: Tokenizer to setup
        template: Custom template string. If None, uses default based on model
        model_name: Model name to determine default template
        
    Returns:
        Tokenizer with chat template set
    """
    if template is None:
        # Determine template based on model name
        if model_name and "gemma" in model_name.lower():
            template = GEMMA_CHAT_TEMPLATE
        else:
            # Use default template if available
            if tokenizer.chat_template is None:
                template = GEMMA_CHAT_TEMPLATE
            else:
                # Keep existing template
                return tokenizer
    
    tokenizer.chat_template = template
    return tokenizer

