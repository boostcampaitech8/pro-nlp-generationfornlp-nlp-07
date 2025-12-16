"""Training metrics utilities"""

import torch
import numpy as np
from transformers import PreTrainedTokenizer
import evaluate
from src.config.config import INT_OUTPUT_MAP


def preprocess_logits_for_metrics(logits, labels, tokenizer: PreTrainedTokenizer):
    """
    Preprocess logits for metrics calculation
    
    Args:
        logits: Model logits
        labels: Ground truth labels
        tokenizer: Tokenizer to use
        
    Returns:
        Preprocessed logits for answer tokens only
    """
    logits = logits if not isinstance(logits, tuple) else logits[0]
    
    # Get indices for answer tokens (1, 2, 3, 4, 5)
    logit_idx = [
        tokenizer.vocab.get("1", tokenizer.vocab.get("1", 0)),
        tokenizer.vocab.get("2", tokenizer.vocab.get("2", 0)),
        tokenizer.vocab.get("3", tokenizer.vocab.get("3", 0)),
        tokenizer.vocab.get("4", tokenizer.vocab.get("4", 0)),
        tokenizer.vocab.get("5", tokenizer.vocab.get("5", 0)),
    ]
    
    # -2: answer token, -1: eos token
    logits = logits[:, -2, logit_idx]
    return logits


def compute_metrics(evaluation_result, tokenizer: PreTrainedTokenizer):
    """
    Compute metrics for evaluation
    
    Args:
        evaluation_result: Tuple of (logits, labels)
        tokenizer: Tokenizer to use
        
    Returns:
        Dictionary with computed metrics
    """
    logits, labels = evaluation_result
    
    # Load accuracy metric
    acc_metric = evaluate.load("accuracy")
    
    # Decode labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = [x.split("<end_of_turn>")[0].strip() for x in labels]
    labels = [INT_OUTPUT_MAP.get(x, 0) for x in labels]
    
    # Convert logits to probabilities and get predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs.numpy(), axis=-1)
    
    # Compute accuracy
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc

