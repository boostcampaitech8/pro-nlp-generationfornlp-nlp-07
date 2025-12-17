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
        for i, data in enumerate(iterator):
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
            
            # Debug: Print input shape to check for huge samples
            if i == 0:
                print(f"Sample {i} input shape: {inputs.shape}")
                # Debug: Check token IDs for 1-5
                for val in range(1, 6):
                    tid = tokenizer.encode(str(val), add_special_tokens=False)[-1]
                    print(f"Debug: Token ID for '{val}': {tid}")
            
            # Get model outputs
            outputs = model(inputs)
            
            # Get logits for the last token
            logits = outputs.logits[:, -1].flatten().cpu()
            
            # Get logits for answer tokens (1, 2, 3, 4, 5)
            # Use encode to get reliable token IDs. [-1] takes the last token in case of prefix space/start token behavior.
            token_ids = [tokenizer.encode(str(i + 1), add_special_tokens=False)[-1] for i in range(len_choices)]
            target_logit_list = [logits[tid] for tid in token_ids]
            
            # Apply softmax
            probs = torch.nn.functional.softmax(
                torch.tensor(target_logit_list, dtype=torch.float32),
                dim=0
            ).detach().cpu().numpy()
            
            if i == 0:
                print(f"Debug: Logits for 1-5: {target_logit_list}")
                print(f"Debug: Probs for 1-5: {probs}")
            
            # Get prediction
            predict_idx = np.argmax(probs, axis=-1)
            predict_value = PRED_CHOICES_MAP[predict_idx]
            
            infer_results.append({
                "id": _id, 
                "answer": predict_value,
                "logits": [float(x) for x in target_logit_list],
                "probs": [float(x) for x in probs]
            })
            
            # Clean up memory
            del inputs, outputs, logits, probs
            if i % 10 == 0:
                torch.cuda.empty_cache()
                
    return infer_results

