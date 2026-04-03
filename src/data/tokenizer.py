"""Tokenization utilities"""

from typing import List, Dict, Any
from datasets import Dataset
from transformers import PreTrainedTokenizer


def formatting_prompts_func(example: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> List[str]:
    """
    Format prompts using chat template
    
    Args:
        example: Example with 'messages' key
        tokenizer: Tokenizer with chat template
        
    Returns:
        List of formatted prompt strings
    """
    output_texts = []
    for i in range(len(example["messages"])):
        output_texts.append(
            tokenizer.apply_chat_template(
                example["messages"][i],
                tokenize=False,
            )
        )
    return output_texts


def tokenize(
    element: Dict[str, Any],
    tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    """
    Tokenize data
    
    Args:
        element: Element with 'messages' key
        tokenizer: Tokenizer to use
        
    Returns:
        Dictionary with 'input_ids' and 'attention_mask'
    """
    def formatting_prompts_func_local(example):
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False,
                )
            )
        return output_texts
    
    outputs = tokenizer(
        formatting_prompts_func_local(element),
        truncation=False,
        padding=False,
        return_overflowing_tokens=False,
        return_length=False,
    )
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
    }


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    num_proc: int = 4,
    load_from_cache_file: bool = True
) -> Dataset:
    """
    Tokenize entire dataset
    
    Args:
        dataset: Dataset to tokenize
        tokenizer: Tokenizer to use
        num_proc: Number of processes for parallel processing
        load_from_cache_file: Whether to load from cache
        
    Returns:
        Tokenized dataset
    """
    def tokenize_func(element):
        def formatting_prompts_func_local(example):
            output_texts = []
            for i in range(len(example["messages"])):
                output_texts.append(
                    tokenizer.apply_chat_template(
                        example["messages"][i],
                        tokenize=False,
                    )
                )
            return output_texts
        
        outputs = tokenizer(
            formatting_prompts_func_local(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }
    
    tokenized_dataset = dataset.map(
        tokenize_func,
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc="Tokenizing",
    )
    
    return tokenized_dataset


def filter_by_length(dataset: Dataset, max_length: int = 1024) -> Dataset:
    """
    Filter dataset by token length
    
    Args:
        dataset: Dataset to filter
        max_length: Maximum token length
        
    Returns:
        Filtered dataset
    """
    filtered_dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)
    return filtered_dataset

