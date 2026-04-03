"""Prediction utilities"""

import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.config.config import PRED_CHOICES_MAP


def predict(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    len_choices: int = 5,
    device: str = "cuda"
) -> str:
    """
    Predict answer for a single sample
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        messages: List of messages in chat format
        len_choices: Number of choices
        device: Device to run inference on
        
    Returns:
        Predicted answer (1-5)
    """
    model.eval()
    
    with torch.inference_mode():
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        
        # Get model outputs
        outputs = model(inputs)
        
        # Get logits for the last token
        logits = outputs.logits[:, -1].flatten().cpu()
        
        # Get logits for answer tokens (1, 2, 3, 4, 5)
        target_logit_list = [
            logits[tokenizer.vocab.get(str(i + 1), 0)]
            for i in range(len_choices)
        ]
        
        # Apply softmax
        probs = torch.nn.functional.softmax(
            torch.tensor(target_logit_list, dtype=torch.float32),
            dim=0
        ).detach().cpu().numpy()
        
        # Get prediction
        predict_idx = np.argmax(probs, axis=-1)
        predict_value = PRED_CHOICES_MAP[predict_idx]
        
        return predict_value


def predict_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: List[Dict[str, Any]],
    device: str = "cuda",
    show_progress: bool = True
) -> List[Dict[str, str]]:
    """
    Predict answers for a batch of samples
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_dataset: List of test samples with 'id', 'messages', 'len_choices'
        device: Device to run inference on
        show_progress: Whether to show progress bar
        
    Returns:
        List of predictions with 'id' and 'answer' keys
    """
    model.eval()
    infer_results = []
    
    iterator = tqdm(test_dataset) if show_progress else test_dataset
    
    with torch.inference_mode():
        for data in iterator:
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]
            
            # Apply chat template
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            
            # Get model outputs
            outputs = model(inputs)
            
            # Get logits for the last token
            logits = outputs.logits[:, -1].flatten().cpu()
            
            # Get logits for answer tokens (1, 2, 3, 4, 5)
            target_logit_list = [
                logits[tokenizer.vocab.get(str(i + 1), 0)]
                for i in range(len_choices)
            ]
            
            # Apply softmax
            probs = torch.nn.functional.softmax(
                torch.tensor(target_logit_list, dtype=torch.float32),
                dim=0
            ).detach().cpu().numpy()
            
            # Get prediction
            predict_idx = np.argmax(probs, axis=-1)
            predict_value = PRED_CHOICES_MAP[predict_idx]
            
            infer_results.append({"id": _id, "answer": predict_value})
    
    return infer_results

