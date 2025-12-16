"""Configuration settings for the project"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Hugging Face settings
HF_ORG = "NLP-07-ODQA"
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_PATH = DATA_DIR / "train" / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test" / "test.csv"

# Model settings
DEFAULT_MODEL_NAME = "beomi/gemma-ko-2b"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "outputs_gemma"

# Training hyperparameters
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
MAX_SEQ_LENGTH = 1024
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"
LOGGING_STEPS = 1
SAVE_STRATEGY = "epoch"
EVALUATION_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 2
SAVE_ONLY_MODEL = True
REPORT_TO = "none"

# LoRA settings
LORA_R = 6
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ['q_proj', 'k_proj']

# Data processing
MAX_TOKEN_LENGTH = 1024
TEST_SIZE = 0.1
RANDOM_SEED = 42

# System message
SYSTEM_MESSAGE = "지문을 읽고 질문의 답을 구하세요."

# Response template for data collator
RESPONSE_TEMPLATE = "<start_of_turn>model"

# Output mapping
INT_OUTPUT_MAP = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
PRED_CHOICES_MAP = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}


def get_config():
    """Get configuration dictionary"""
    return {
        "hf_org": HF_ORG,
        "hf_token": HF_TOKEN,
        "data_dir": str(DATA_DIR),
        "train_data_path": str(TRAIN_DATA_PATH),
        "test_data_path": str(TEST_DATA_PATH),
        "model_name": DEFAULT_MODEL_NAME,
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "weight_decay": WEIGHT_DECAY,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "logging_steps": LOGGING_STEPS,
        "save_strategy": SAVE_STRATEGY,
        "evaluation_strategy": EVALUATION_STRATEGY,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "save_only_model": SAVE_ONLY_MODEL,
        "report_to": REPORT_TO,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "max_token_length": MAX_TOKEN_LENGTH,
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "system_message": SYSTEM_MESSAGE,
        "response_template": RESPONSE_TEMPLATE,
    }


def load_config():
    """Load and return configuration"""
    return get_config()

