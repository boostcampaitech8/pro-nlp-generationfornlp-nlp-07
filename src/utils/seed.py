"""Seed setting utility"""

import random
import numpy as np
import torch


def set_seed(random_seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        random_seed: Random seed value (default: 42)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

