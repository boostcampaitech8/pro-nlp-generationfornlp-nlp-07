from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import evaluate
import torch
import numpy as np
import pandas as pd
from ast import literal_eval
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from pathlib import Path
from tqdm import tqdm
import sys
from dotenv import load_dotenv


# 해당 파일은 scripts/experiments/memberA/ 폴더에 위치한 것이므로,
# project root를 따로 추가해줍니다.
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.hf_utils import upload_all_checkpoints_to_hf

load_dotenv()

### 직접 수정 가능한 변수들은 실험하기 편하게 상단에 모아 두었습니다.
# 상수
RANDOM_STATE = 42
CAMPER_ID = "T8091"
EXP_NAME = "mistral-small-3.2-24b-it-2506-qlora-v1"
HF_ORG = "NLP-07-ODQA"

# 경로
OUTPUT_DIR = project_root / "outputs" / CAMPER_ID / EXP_NAME
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
SUBMISSION_DIR = project_root / "submissions" / CAMPER_ID

# 모델 로더 설정값
MODEL_LOADER_CONFIG = {
    "model_name": "unsloth/Mistral-Small-3.2-24B-Instruct-2506-bnb-4bit", # str(BEST_MODEL_DIR),#
    "max_seq_length": 4096, # 현재 데이터의 시퀀스 길이가 대부분 500~3000 사이이므로, 그 이상으로 설정합니다.
    "dtype": torch.float16, # V100 사용중이므로 Float16 기본 사용
    "load_in_4bit": True,  # Use 4bit quantization to reduce memory usage. Can be False.
    # token = "hf_...",     # 승인이 필요한 모델을 사용하는 경우 허깅페이스 토큰이 필요하다는 뜻인 것 같습니다. (원문: use one if using gated models like meta-llama/Llama-2-7b-hf)
}

# LoRA 어댑터 설정값
LORA_CONFIG = {
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ],                                         # 모듈 종류: "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    "r": 8,                                    # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    "lora_alpha": 16,                           # 보통 r 값과 동일하거나 2배로 설정
    "lora_dropout": 0,                      # Supports any, but = 0 is optimized
    "bias": "none",                            # Supports any, but = "none" is optimized
    "use_gradient_checkpointing": "unsloth",   # True or "unsloth" for very long context
    "random_state": RANDOM_STATE,
    "use_rslora": False,                       # We support rank stabilized LoRA
    "loftq_config": None,                      # And LoftQ
}

# SFT 설정값
SFT_CONFIG = {
    "output_dir": OUTPUT_DIR,
    "lr_scheduler_type": "cosine",
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    # "gradient_accumulation_steps": 4,
    "weight_decay": 0.01,
    "logging_steps": 1,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "seed": RANDOM_STATE,
    "report_to": "none",
}

# chat_template 설정값
CHAT_TEMPLATE_CONFIG = {
    "instruction_part": None,   # instruction_part, response_part는 None으로 사용해야 자동으로 찾아주는 로직이 동작합니다!
    "response_part": None,      # 만약! None으로 했을 때, 마스킹 검증 과정에서 오류가 발생한다면 둘 다 사용하는 모델에 맞게 입력해야 합니다.
    # Tokenizer에 chat_template이 없을 경우, 기본 사용되는 chat_template, instruction_part, response_part 입니다. (범용 ChatML 양식, 수정하지 않는 것을 권장)
    "default_chat_template": """{% for message in messages %}{% if message['role'] == 'user' %}{{'<|im_start|>user
' + message['content'] + '<|im_end|>
'}}{% elif message['role'] == 'assistant' %}{{'<|im_start|>assistant
' + message['content'] + '<|im_end|>
' }}{% else %}{{ '<|im_start|>system
' + message['content'] + '<|im_end|>
' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}""",
}

# 데이터 파일 설정
DATA_FILES = {
    "train": str(project_root / "data" / "train" / "train.csv"),
    "test": str(project_root / "data" / "test" / "test.csv"),
}

# chat_template마다 system role의 지원 여부가 다르므로, system_prompt를 user role의 맨 처음 부분에 통합하였습니다.
# 또한, 전처리 로직에서 현재 프롬프트는 question_plus 컬럼의 존재 여부, choices 컬럼의 갯수에 따라 최종 내용을 다르게 처리합니다.
PROCESSING_CONFIG = {
    "eval_split_ratio": 0.1,                            # Train 데이터셋에서 Evaluation 데이터셋으로 분할할 비율 (0으로 설정 시 분할하지 않음: 자동적으로 Eval도 수행 안함)
    "system_prompt": "지문을 읽고 질문의 답을 구하세요.",   # 시스템 프롬프트는 User role의 맨 앞에 추가됩니다 (chat_template마다 system role의 지원 여부가 다르므로)
    "prompt_template": """{system_prompt}

지문:
{paragraph}

질문:
{question}
{question_plus_section}

선택지:
{choices}

{choice_range} 중에 하나를 정답으로 고르세요.
정답:""",
}

INFERENCE_CONFIG = {
    "batch_size": 1,
}

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# 우리가 4비트 양자화를 사용하는 QLoRA를 사용하게 되면 아래 단계를 거치게 됩니다.
# 1. 16비트 모델을 다운로드
# 2. Lora에 사용되지 않는 부분을 4비트 양자화
# 3. Lora에 사용되는 부분은 16비트로 유지
# 4. Lora 어댑터 학습
# 여기서 16비트 모델을 다운로드하고 양자화하는 과정을 건너뛰기 위해,
# Unsloth에서 미리 4비트 양자화된 모델을 다운로드하여 사용합니다.
# 그러면 아래와 같은 장점이 있습니다.
# - 다운로드 시간 단축, 저장 공간 사용량 절약
# - 자체 양자화에 필요한 컴퓨팅 자원 절약
# 아래는 4비트 양자화된 모델의 예시가 들어있는 것이며, 실험과는 무관한 내용입니다.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth


### 모델 로드 및 준비 과정
print("=" * 50)
print("   모델 로드 및 준비 과정")
print("=" * 50)
# 모델, 토크나이저 로드 (Unsloth는 둘 다 한번에 불러옵니다)
model, tokenizer = FastLanguageModel.from_pretrained(**MODEL_LOADER_CONFIG)
print(f"✅ 모델 로드 완료: {MODEL_LOADER_CONFIG['model_name']}")

# Processor 구조인 모델의 경우 tokenizer만 로드
if hasattr(tokenizer, 'tokenizer'):
    tokenizer = tokenizer.tokenizer
    print("✅ 현재 모델은 Processor를 사용하고 있습니다! 자동으로 tokenizer를 추출합니다.")

# 모델에 LoRA 어댑터 추가
model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
print("✅ LoRA 어댑터 추가 완료")

# 구형 토크나이저의 경우 vocab을 불러오는 방식이 다름 (get_vocab 메서드로 통일)
if not callable(getattr(tokenizer, 'get_vocab', None)):
    # get_vocab이 없거나 호출 불가능하면 추가
    if hasattr(tokenizer, 'vocab'):
        tokenizer.get_vocab = lambda: tokenizer.vocab
        print("⚠️ 구형 토크나이저 감지: get_vocab() 메서드 추가됨")
    else:
        choice_tokens = [str(i) for i in range(1, 6)]
        choice_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in choice_tokens]
        partial_vocab= {t: i for t, i in zip(choice_tokens, choice_ids)}
        tokenizer.get_vocab = lambda: partial_vocab
        print("⚠️ get_vocab, vocab 둘 다 없음: 부분적인 vocab 추가됨")


### Tokenizer의 chat_template 확인 과정
print("=" * 50)
print("   Tokenizer의 chat_template 확인 과정")
print("=" * 50)
# 자동으로 instruction_part, response_part를 찾아주는 함수
def auto_parts(model_name=None):
    if model_name is None:
        return '<|im_start|>user\n', '<|im_start|>assistant\n'

    model_name = model_name.lower()

    # Llama-3 계열
    if 'llama-3' in model_name or 'llama3' in model_name:
        return (
            '<|start_header_id|>user<|end_header_id|>\n\n',
            '<|start_header_id|>assistant<|end_header_id|>\n\n'
        )
    
    # Llama-2 계열
    elif 'llama-2' in model_name or 'llama2' in model_name:
        return '[INST]', '[/INST]'
    
    # Qwen 계열 (ChatML)
    elif 'qwen' in model_name:
        return '<|im_start|>user\n', '<|im_start|>assistant\n'
    
    # Gemma 계열
    elif 'gemma' in model_name:
        return '<start_of_turn>user\n', '<start_of_turn>model\n'
    
    # Mistral/Mixtral
    elif 'mistral' in model_name or 'mixtral' in model_name:
        return '[INST]', '[/INST]'
    
    # Phi 계열
    elif 'phi' in model_name:
        return '<|user|>\n', '<|assistant|>\n'
    
    # 기본값 (ChatML)
    else:
        return '<|im_start|>user\n', '<|im_start|>assistant\n'

# Tokenizer에 chat_template이 없을 경우 기본값 적용
if tokenizer.chat_template is None:
    tokenizer.chat_template = CHAT_TEMPLATE_CONFIG['default_chat_template']
    CHAT_TEMPLATE_CONFIG['instruction_part'], CHAT_TEMPLATE_CONFIG['response_part'] = auto_parts()
    print("⚠️ Tokenizer에 chat_template이 없어서 기본 템플릿 사용")
else:
    print("✅ Tokenizer의 chat_template 확인됨")

# instruction_part, response_part가 None일 경우 자동 탐색
if CHAT_TEMPLATE_CONFIG['instruction_part'] is None or CHAT_TEMPLATE_CONFIG['response_part'] is None:
    CHAT_TEMPLATE_CONFIG['instruction_part'], CHAT_TEMPLATE_CONFIG['response_part'] = auto_parts(MODEL_LOADER_CONFIG['model_name'])
    print("✅ instruct_part, response_part 자동 탐색 사용됨")

# 설정된 instruction_part, response_part 확인
print(f"   instruction_part: {repr(CHAT_TEMPLATE_CONFIG['instruction_part'])}")
print(f"   response_part: {repr(CHAT_TEMPLATE_CONFIG['response_part'])}")


### 데이터셋 전처리 및 준비 과정
print("=" * 50)
print("   데이터셋 전처리 및 준비 과정")
print("=" * 50)
# 데이터 파싱 함수 (기본 데이터셋 형식에 맞춰져 있습니다.)
# 사용하려는 데이터셋의 컬럼 구성이 다르다면 이 함수를 반드시 수정해야 합니다!
def parse_data(example):
    problems = literal_eval(example['problems']) # 하위 컬럼이 존재하므로 별도의 작업 필요
    return {
        'id': example['id'],
        'paragraph': example['paragraph'],
        'question': problems['question'],
        'question_plus': problems.get('question_plus', None),
        'choices': problems['choices'],
        'answer': problems.get('answer', None),
    }

# 프롬프트 빌더 함수 (기본 데이터셋 형식에 맞춰져 있습니다.)
# 사용하려는 데이터셋의 컬럼 구성이 다르다면 이 함수를 반드시 수정해야 합니다!
def build_prompt(data):
    question_plus_section = f"\n<보기>:\n{data['question_plus']}" if data['question_plus'] else ""
    choices_string = '\n'.join([f"{i + 1}. {choice}" for i, choice in enumerate(data['choices'])])
    choice_range = ', '.join(map(str, range(1, len(data['choices']) + 1)))

    return PROCESSING_CONFIG['prompt_template'].format(
        system_prompt=PROCESSING_CONFIG['system_prompt'],
        paragraph=data['paragraph'],
        question=data['question'],
        question_plus_section=question_plus_section,
        choices=choices_string,
        choice_range=choice_range,
    )

# 메시지 빌더 함수
def build_messages(data, include_answer):
    user_content = build_prompt(data)

    messages = [{"role": "user", "content": user_content}]

    if include_answer:
        messages.append({"role": "assistant", "content": str(data['answer'])})
    
    return messages

# 별도 토큰(<think>) 자동 생성 방지를 위한 안전 래퍼
def apply_chat_template_safe(tokenizer, messages, include_answer, tokenize=False):
    text = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=tokenize,
        add_generation_prompt=False,
    )

    # [/INST]를 사용하는 경우 User 템플릿에 이미 포함되어 있으므로 추가하지 않음 (Error case: Mistral 계열)
    if CHAT_TEMPLATE_CONFIG['response_part'] != "[/INST]":
        text += CHAT_TEMPLATE_CONFIG['response_part']
    
    if len(messages) == 2:
        text += messages[1]['content'] + tokenizer.eos_token

    return text

# 포맷팅 함수 (기본 데이터셋 형식에 맞춰져 있습니다.)
# 사용하려는 데이터셋의 컬럼 구성이 다르다면 이 함수를 반드시 수정해야 합니다!
def formatting_func(example, tokenizer, include_answer):
    texts = []

    for idx in range(len(example['paragraph'])):
        messages = build_messages({
            'paragraph': example['paragraph'][idx],
            'question': example['question'][idx],
            'question_plus': example['question_plus'][idx],
            'choices': example['choices'][idx],
            'answer': example['answer'][idx],
        }, include_answer)
        
        text = apply_chat_template_safe(tokenizer, messages, include_answer)
        texts.append(text)
    
    return {"text": texts}

# 데이터셋 단위로 포맷팅 적용하는 함수
def format_dataset(dataset, tokenizer, include_answer, desc):
    remove_cols = [col for col in dataset.column_names if col != 'id']

    return dataset.map(
        lambda examples: formatting_func(examples, tokenizer, include_answer),
        batched=True,
        remove_columns=remove_cols,
        desc=desc,
    )

# 데이터셋 로드 및 기본적인 파싱 진행
train_dataset = load_dataset('csv', data_files={'train': DATA_FILES['train']})['train']
test_dataset = load_dataset('csv', data_files={'test': DATA_FILES['test']})['test']

# 테스트 데이터셋의 Unnamed: 0 컬럼 제거
test_dataset = test_dataset.remove_columns(['Unnamed: 0'])

print("✅ 데이터셋 로드 완료")
print(f"   Train: {len(train_dataset)}")
print(f"   Test: {len(test_dataset)}")

train_dataset = train_dataset.map(parse_data, remove_columns=['problems'], desc="Parsing dataset")
test_dataset = test_dataset.map(parse_data, remove_columns=['problems'], desc="Parsing dataset")
print("✅ 데이터셋 파싱 완료")

# 데이터셋 분할
if PROCESSING_CONFIG['eval_split_ratio'] > 0:
    split = train_dataset.train_test_split(test_size=PROCESSING_CONFIG['eval_split_ratio'], seed=RANDOM_STATE)
    train_dataset, eval_dataset = split['train'], split['test']
    print(f"✅ 데이터셋 분할 완료 (Eval 비율: {PROCESSING_CONFIG['eval_split_ratio'] * 100:.2f}%)")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Eval: {len(eval_dataset)}")
else:
    eval_dataset = None
    print("⚠️ Eval 데이터셋 분할 안 함 (eval_split_ratio = 0)")

# 데이터셋 포맷팅 적용
train_dataset = format_dataset(train_dataset, tokenizer, True, "Formatting Train dataset")
if eval_dataset is not None:
    eval_dataset = format_dataset(eval_dataset, tokenizer, True, "Formatting Eval dataset")
test_dataset = format_dataset(test_dataset, tokenizer, False, "Formatting Test dataset")
print("✅ 데이터셋 포맷팅 완료")
print(f"   포맷팅 이후 컬럼: {train_dataset.column_names}")


### 모델 학습 준비 과정
print("=" * 50)
print("   모델 학습 준비 과정")
print("=" * 50)
# 모델 학습을 위한 Tokenizer 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right' # Training용 padding side, Inference용 padding side는 'left'

# Evaluation을 위한 Mertic 설정
if eval_dataset is None:
    preprocess_logits_for_metrics = None
    compute_metrics = None
    print("⚠️ Evaluation을 위한 Metric 설정 안 함 (eval_split_ratio = 0)")
else:
    metric = evaluate.load("accuracy")
    vocab = tokenizer.get_vocab()
    _answer_ids = [vocab[str(i)] for i in range(1, 6)]
    _answer_map = {str(i): i-1 for i in range(1, 6)}

    def preprocess_logits_for_metrics(logits, labels):
        logits = logits[0] if isinstance(logits, tuple) else logits
        return logits[:, -2, _answer_ids]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        
        # 디코딩
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # skip_special_tokens가 제대로 동작하지 않을 경우를 대비한 안전장치
        if tokenizer.eos_token:
            labels = [l.split(tokenizer.eos_token)[0].strip() for l in labels]
        else:
            labels = [l.strip() for l in labels]
        
        # 정답 변환
        labels = [_answer_map.get(l, 0) for l in labels]
        preds = np.argmax(torch.softmax(torch.tensor(logits), dim=-1), axis=-1)

        return metric.compute(predictions=preds, references=labels)

# 모델 학습을 위한 트레이너 설정
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = MODEL_LOADER_CONFIG['max_seq_length'],
    packing = False, # Can make training 5x faster for short sequences.(여러 샘플을 연속으로 붙여서 한번에 넣는 방식: 마스킹 오류 발생 확률 높아서 사용 안함)
    compute_metrics = compute_metrics,
    preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    args = SFTConfig(**SFT_CONFIG),
)

# Unsloth 마스킹 적용
trainer = train_on_responses_only(
    trainer,
    instruction_part = CHAT_TEMPLATE_CONFIG['instruction_part'],
    response_part = CHAT_TEMPLATE_CONFIG['response_part'],
)
print("✅ 모델 학습 준비 완료")


### 학습 내용 검증 과정
print("=" * 50)
print("   학습 내용 검증 과정")
print("=" * 50)

batch = next(iter(trainer.get_train_dataloader()))
labels = batch["labels"][0]

learning_ratio = (labels != -100).sum().item() / len(labels)
learning_text = tokenizer.decode(labels[labels != -100])
print(f"학습 토큰 비율: {learning_ratio*100:.1f}%")
print(f"학습 내용: {learning_text}")

# 마스킹 비율 검사
if learning_ratio > 0.5:
    print("⚠️ 마스킹 검증 실패 (학습 비율 50% 초과)")
    print("→ CHAT_TEMPLATE_CONFIG에서 instruction_part와 response_part를 사용 모델에 맞게 수정하세요.")
    sys.exit(1)
else:
    print("✅ 마스킹 검증 성공")

# 학습 내용 검사
stripped = learning_text.strip()
if not stripped or stripped[0] not in ["1", "2", "3", "4", "5"]:
    print("⚠️ 학습 내용 검증 실패 (최초 학습 토큰이 1~5 사이의 값이 아님)")
    print("→ chat_template, apply_chat_template_safe(), response_part 중 어딘가 잘못된 부분이 있습니다!")
    print("마스킹 전 전체 텍스트 (디버그 용):")
    print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False))
    sys.exit(1)
else:
    print("✅ 학습 내용 검증 성공")



### 학습 시작
print("=" * 50)
print("   학습 시작")
print("=" * 50)
# 모델 저장 폴더 확인
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 현재 메모리 상태 출력
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# 학습 시작
trainer_stats = trainer.train()
print("✅ 학습 완료")

# 최종 메모리 상태 및 소요 시간 출력
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# 학습 완료된 모델 저장
BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(BEST_MODEL_DIR))
tokenizer.save_pretrained(str(BEST_MODEL_DIR))
print("✅ 학습 완료 모델 저장됨")


### 허깅페이스 모델 업로드
print("=" * 50)
print("   허깅페이스 모델 업로드")
print("=" * 50)
# 체크포인트 업로드
try:
    hf_model_name = f"{HF_ORG}/{EXP_NAME}"

    print(f"모델 업로드 시작: {hf_model_name}")
    uploaded_urls = upload_all_checkpoints_to_hf(
        output_dir=str(OUTPUT_DIR),
        model_name=hf_model_name,
        experiment_name=EXP_NAME,
    )

    if uploaded_urls:
        print(f"✅ {len(uploaded_urls)}개의 체크포인트 업로드 완료")
        for url in uploaded_urls:
            print(f"   - {url}")
    else:
        print("⚠️ 업로드할 체크포인트가 없습니다.")

except Exception as e:
    print(f"⚠️ 모델 업로드 과정에서 오류가 발생하였습니다: {e}")
    raise


### 추론 과정
print("=" * 50)
print("   추론 준비 과정")
print("=" * 50)
# 모델 불러오는 부분은 생략됨. 모듈화 시 해당 부분 별도 정의 필요
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=str(BEST_MODEL_DIR),
#     max_seq_length=MODEL_LOADER_CONFIG['max_seq_length'],
#     dtype=MODEL_LOADER_CONFIG['dtype'],
#     load_in_4bit=MODEL_LOADER_CONFIG['load_in_4bit'],
# )
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
tokenizer.padding_side = "left"         # Inference용 padding side
print("✅ 추론 모드 활성화")

# 테스트 데이터 정보
test_ids = test_dataset['id']
test_texts = test_dataset['text']
print(f"✅ 테스트 데이터 {len(test_ids)}개 준비됨")

# Choices 개수 로드 (logits 추출용)
# 모듈화 시 해당 부분 개선 필요
original_test = load_dataset('csv', data_files={'test': DATA_FILES['test']})['test']
original_test_parsed = original_test.map(parse_data, remove_columns=['problems'])
test_len_choices = [len(row['choices']) for row in original_test_parsed]
print(f"✅ Choices 로드 완료")


### 배치 추론 시작
print("=" * 50)
print(f"   배치 추론 시작 (batch_size = {INFERENCE_CONFIG['batch_size']})")
print("=" * 50)
# 추론 결과 저장 폴더 확인
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

predictions = []
pred_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
vocab = tokenizer.get_vocab()

model.eval()
with torch.no_grad():
    for i in tqdm(range(0, len(test_dataset), INFERENCE_CONFIG['batch_size']), desc="Batch Inference"):
        # 배치 데이터 준비
        batch_texts = test_texts[i:i+INFERENCE_CONFIG['batch_size']]
        batch_len_choices = test_len_choices[i:i+INFERENCE_CONFIG['batch_size']]
        
        # 토큰화
        inputs = tokenizer(
            batch_texts,
            return_tensors = 'pt',
            padding = True,
            truncation = True,
            max_length = MODEL_LOADER_CONFIG['max_seq_length'],
        ).to(model.device)
        
        # 첫 샘플로 디버깅
        if i == 0:
            print("첫 샘플로 디버깅 수행")
            debug_output = model.generate(
                **{k: v[:1] for k, v in inputs.items()},
                do_sample=False,
            )
            debug_decode = tokenizer.decode(debug_output[0], skip_special_tokens=False)
            print("   Prompt:")
            print(batch_texts[0])
            print("   Model Answer:")
            print(debug_decode)

        # 추론
        outputs = model(**inputs)

        # 각 샘플 처리
        for j in range(len(batch_texts)):
            logits = outputs.logits[j, -1].cpu()
            len_choices = batch_len_choices[j]
            
            # 정답 토큰 logits 추출
            target_logits = [logits[vocab[str(k + 1)]] for k in range(len_choices)]
            
            # Softmax 및 예측
            probs = torch.nn.functional.softmax(
                torch.tensor(target_logits, dtype=torch.float32),
                dim=-1
            )
            pred = pred_map[torch.argmax(probs).item()]
            predictions.append(pred)
print("✅ 추론 완료")

output_df = pd.DataFrame({
    'id': test_ids,
    'answer': predictions
})

# 샘플 결과 확인
print("\n첫 5개 예측 결과:")
print(output_df.head())

# 예측 분포
print("\n예측 분포:")
print(output_df['answer'].value_counts().sort_index())

submission_file = SUBMISSION_DIR / f"{EXP_NAME}.csv"
output_df.to_csv(submission_file, index=False)
print(f"✅ 추론 결과 저장됨: {submission_file}")