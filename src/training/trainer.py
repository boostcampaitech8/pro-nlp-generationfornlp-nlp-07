"""Training utilities"""

from pathlib import Path
from trl import SFTTrainer, SFTConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from peft import LoraConfig
# from src.training.data_collator import get_data_collator  # trl>0.20.0 버전은 data_collator를 사용하지 않습니다.
from src.training.metrics import compute_metrics, preprocess_logits_for_metrics
from src.training.callbacks import SaveBestModelCallback
from src.config.config import (
    LEARNING_RATE,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    MAX_SEQ_LENGTH,
    WEIGHT_DECAY,
    LR_SCHEDULER_TYPE,
    LOGGING_STEPS,
    SAVE_STRATEGY,
    EVALUATION_STRATEGY,
    SAVE_TOTAL_LIMIT,
    SAVE_ONLY_MODEL,
    REPORT_TO,
    RESPONSE_TEMPLATE,
)


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    peft_config: LoraConfig,
    output_dir: str,
    learning_rate: float = None,
    num_train_epochs: int = None,
    per_device_train_batch_size: int = None,
    per_device_eval_batch_size: int = None,
    max_seq_length: int = None,
    weight_decay: float = None,
    lr_scheduler_type: str = None,
    logging_steps: int = None,
    save_strategy: str = None,
    evaluation_strategy: str = None,
    save_total_limit: int = None,
    save_only_model: bool = None,
    report_to: str = None,
    response_template: str = None,
    **kwargs
) -> SFTTrainer:
    """
    Create SFT trainer
    
    Args:
        model: Model to train
        tokenizer: Tokenizer to use
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        peft_config: LoRA configuration
        output_dir: Output directory for checkpoints
        learning_rate: Learning rate
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        max_seq_length: Maximum sequence length
        weight_decay: Weight decay
        lr_scheduler_type: Learning rate scheduler type
        logging_steps: Logging steps
        save_strategy: Save strategy
        evaluation_strategy: Evaluation strategy
        save_total_limit: Save total limit
        save_only_model: Save only model
        report_to: Report to (e.g., "wandb", "tensorboard", "none")
        response_template: Response template for data collator
        **kwargs: Additional arguments for SFTConfig
        
    Returns:
        SFT trainer
    """
    # Set defaults
    learning_rate = learning_rate or LEARNING_RATE
    num_train_epochs = num_train_epochs or NUM_TRAIN_EPOCHS
    per_device_train_batch_size = per_device_train_batch_size or PER_DEVICE_TRAIN_BATCH_SIZE
    per_device_eval_batch_size = per_device_eval_batch_size or PER_DEVICE_EVAL_BATCH_SIZE
    max_seq_length = max_seq_length or MAX_SEQ_LENGTH
    weight_decay = weight_decay or WEIGHT_DECAY
    lr_scheduler_type = lr_scheduler_type or LR_SCHEDULER_TYPE
    logging_steps = logging_steps or LOGGING_STEPS
    save_strategy = save_strategy or SAVE_STRATEGY
    evaluation_strategy = evaluation_strategy or EVALUATION_STRATEGY
    save_total_limit = save_total_limit or SAVE_TOTAL_LIMIT
    save_only_model = save_only_model if save_only_model is not None else SAVE_ONLY_MODEL
    report_to = report_to or REPORT_TO
    response_template = response_template or RESPONSE_TEMPLATE
    
    # Setup tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    
    # Create data collator
    # data_collator = get_data_collator(tokenizer, response_template)  # trl>0.20.0 버전은 data_collator를 사용하지 않습니다.
    
    # Create training config
    sft_config = SFTConfig(
        do_train=True,
        do_eval=True,
        lr_scheduler_type=lr_scheduler_type,
        max_seq_length=max_seq_length,
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy,
        save_total_limit=save_total_limit,
        save_only_model=save_only_model,
        report_to=report_to,
        gradient_checkpointing=True,  # 메모리 절약을 위해 활성화
        load_best_model_at_end=True,  # 학습 끝에 best model 로드
        metric_for_best_model="eval_loss",  # eval_loss를 기준으로 best model 선택
        greater_is_better=False,  # loss는 작을수록 좋음
        completion_only_loss=True, # trl>0.20.0 버전은 해당 구문이 data_collator 대신 사용됩니다.
        **kwargs
    )
    
    # Create compute_metrics function with tokenizer
    def compute_metrics_func(eval_pred):
        return compute_metrics(eval_pred, tokenizer)
    
    # Create preprocess_logits function with tokenizer
    def preprocess_logits_func(logits, labels):
        return preprocess_logits_for_metrics(logits, labels, tokenizer)
    
    # Create best model directory path
    best_model_dir = str(Path(output_dir) / "best_model")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # data_collator=data_collator, # trl>0.20.0 버전은 data_collator를 사용하지 않습니다.
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_func,
        preprocess_logits_for_metrics=preprocess_logits_func,
        peft_config=peft_config,
        args=sft_config,
        callbacks=[SaveBestModelCallback(best_model_dir=best_model_dir)],  # Best model 저장 callback 추가
    )
    
    return trainer


def train(trainer: SFTTrainer):
    """
    Train the model
    
    Args:
        trainer: SFT trainer
    """
    trainer.train()

