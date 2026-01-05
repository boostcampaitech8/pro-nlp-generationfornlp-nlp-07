#from unsloth.chat_templates import train_on_responses_only
import evaluate
import torch
import numpy as np
import pandas as pd
from ast import literal_eval
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import sys
import os
from dotenv import load_dotenv

# =========================
# Unsloth: force non-xformers attention (V100 safe)
# =========================
os.environ["UNSLOTH_FORCE_SDPA"] = "1"
os.environ["UNSLOTH_FORCE_MATH_ATTENTION"] = "1"
os.environ["UNSLOTH_DISABLE_XFORMERS"] = "1"
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "1"
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("XFORMERS_FORCE_DISABLE", "1")
os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ⚠️ 반드시 이 다음에 unsloth import
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


# 해당 파일은 scripts/experiments/memberA/ 폴더에 위치한 것이므로,
# project root를 따로 추가해줍니다.
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.hf_utils import upload_all_checkpoints_to_hf

load_dotenv()

# === Env overrides (pipeline-friendly) ===
def _env_int(name, default):
    try: return int(os.environ.get(name, str(default)))
    except: return default

def _env_float(name, default):
    try: return float(os.environ.get(name, str(default)))
    except: return default

def _env_str(name, default):
    v = os.environ.get(name, "").strip()
    return v if v else default

# Experiment identity / paths
CAMPER_ID = _env_str("CAMPER_ID", "T8164")
EXP_NAME  = _env_str("EXP_NAME", "Qwen3-32B-bnb-4bit-CoT")
HF_ORG    = _env_str("HF_ORG", "NLP-07-ODQA")

# Model + training knobs
BASE_MODEL_NAME = _env_str("BASE_MODEL", "unsloth/Qwen3-32B-bnb-4bit")
MAX_SEQ_LENGTH  = _env_int("MAX_SEQ_LENGTH", 4096)

LORA_R     = _env_int("LORA_R", 128)
LORA_ALPHA = _env_int("LORA_ALPHA", 128)

LEARNING_RATE = _env_float("LEARNING_RATE", 1e-4)
EPOCHS        = _env_float("EPOCHS", 2)
WEIGHT_DECAY  = _env_float("WEIGHT_DECAY", 0.01)
GRAD_ACCUM    = _env_int("GRAD_ACCUM", 8)

# Upload control (default: 기존 동작 유지 = 업로드 함)
UPLOAD_HF = _env_int("UPLOAD_HF", 1) == 1

# Optional override for OUTPUT_DIR root
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "").strip()
# =========================================


### 직접 수정 가능한 변수들은 실험하기 편하게 상단에 모아 두었습니다.
# 상수
RANDOM_STATE = 42

# 경로
OUTPUT_DIR = (Path(OUTPUT_ROOT) / CAMPER_ID / EXP_NAME) if OUTPUT_ROOT else (project_root / "outputs" / CAMPER_ID / EXP_NAME)
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
SUBMISSION_DIR = project_root / "submissions" / CAMPER_ID

# 모델 로더 설정값
MODEL_LOADER_CONFIG = {
    "model_name": BASE_MODEL_NAME,
    "max_seq_length": MAX_SEQ_LENGTH,
    "dtype": torch.float16, # V100 사용중이므로 Float16 기본 사용
    "load_in_4bit": True,  # Use 4bit quantization to reduce memory usage. Can be False.
    # token = "hf_...",     # 승인이 필요한 모델을 사용하는 경우 허깅페이스 토큰이 필요하다는 뜻인 것 같습니다. (원문: use one if using gated models like meta-llama/Llama-2-7b-hf)
}

# LoRA 어댑터 설정값
LORA_CONFIG = {
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ],                                         # 모듈 종류: "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    "r": LORA_R,
    "lora_alpha": LORA_ALPHA,
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
    "learning_rate": LEARNING_RATE,
    "num_train_epochs": EPOCHS,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "weight_decay": WEIGHT_DECAY,
    "logging_steps": 1,
    "save_strategy": "no",
    "save_total_limit": 1,
    "max_seq_length": 4096,
    "eval_strategy": "no",
    "seed": RANDOM_STATE,
    "report_to": "none",
}
SFT_CONFIG["optim"] = "paged_adamw_8bit"
SFT_CONFIG["bf16"] = False
SFT_CONFIG["fp16"] = True

if "bf16_full_eval" in SFT_CONFIG:
    SFT_CONFIG["bf16_full_eval"] = False

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
# --- CoT train 옵션 (기본값: 비활성; 기존 동작 그대로) ---
USE_COT_TRAIN = int(os.environ.get("USE_COT_TRAIN", "0")) == 1
COT_TRAIN_PATH = os.environ.get("COT_TRAIN_PATH", "").strip()

DATA_FILES = {
    "train": str(project_root / "data" / "train" / "train_with_cot.csv"),
    "test": str(project_root / "data" / "test" / "test.csv"),
    # CoT 학습 파일은 env로만 켜기 (없으면 미사용)
    "train_cot": COT_TRAIN_PATH,
}

# chat_template마다 system role의 지원 여부가 다르므로, system_prompt를 user role의 맨 처음 부분에 통합하였습니다.
# 또한, 전처리 로직에서 현재 프롬프트는 question_plus 컬럼의 존재 여부, choices 컬럼의 갯수에 따라 최종 내용을 다르게 처리합니다.
PROCESSING_CONFIG = {
    "eval_split_ratio": 0,                            # Train 데이터셋에서 Evaluation 데이터셋으로 분할할 비율 (0으로 설정 시 분할하지 않음: 자동적으로 Eval도 수행 안함)
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
    # --- CoT 학습용: 프롬프트 명령/형식 고정 ---
    # user 메시지에 [생각]을 포함시키고, assistant는 숫자 1개만 출력하도록 학습
    "cot_prompt_template": """[문제]
{paragraph}

[질문]
{question}
{question_plus_section}

[선지]
{choices}

[생각]
{cot_text}

[정답]
""",

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
# =========================
# TRL SFTTrainer EOS 정렬 (Qwen ChatML)
# =========================
# Qwen2/2.5 Instruct 계열은 대개 EOS를 "<|im_end|>"로 맞춰야 TRL이 정상 동작합니다.
SFT_CONFIG["eos_token"] = "<|im_end|>"

# pad_token이 비어있으면 eos를 pad로 사용 (TRL도 pad 필요)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Processor 구조인 모델의 경우 tokenizer만 로드
if hasattr(tokenizer, "tokenizer"):
    tokenizer = tokenizer.tokenizer
    print("✅ 현재 모델은 Processor를 사용하고 있습니다! 자동으로 tokenizer를 추출합니다.")

from peft import PeftModel

RESUME_ADAPTER_DIR = os.environ.get("RESUME_ADAPTER_DIR", "").strip()

if RESUME_ADAPTER_DIR and os.path.isdir(RESUME_ADAPTER_DIR):
    # ✅ TAPT에서 저장한 어댑터를 로드하여 이어학습
    print(f"✅ TAPT LoRA 어댑터 로드: {RESUME_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(
        model,
        RESUME_ADAPTER_DIR,
        is_trainable=True,
    )
else:
    # ✅ TAPT가 없으면 새 LoRA 생성(기존 동작)
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
    print("✅ 새 LoRA 어댑터 생성")

# ✅ k-bit + gradient checkpointing에서 loss가 그래프에 안 붙는 이슈 방지
try:
    model.enable_input_require_grads()
except Exception as e:
    print(f"⚠️ enable_input_require_grads() skipped: {e}")

# (선택) LoRA 파라미터가 실제로 grad 대상인지 한 번 더 확인
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ trainable params = {n_trainable:,}")
if n_trainable == 0:
    raise RuntimeError("No trainable parameters. Check LoRA attachment / RESUME_ADAPTER_DIR.")



# ✅ 안전장치: trainable params가 0이면 즉시 중단(지금 오류를 사전에 막음)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ trainable params = {n_trainable:,}")
if n_trainable == 0:
    raise RuntimeError(
        "No trainable parameters. "
        "Check RESUME_ADAPTER_DIR path or LoRA attachment."
    )
print("CUDA capability:", torch.cuda.get_device_capability())
print("xFormers enabled:", not bool(os.environ.get("XFORMERS_DISABLED")))

model.gradient_checkpointing_enable()
model.config.use_cache = False

model.train()
FastLanguageModel.for_training(model)
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
        # Mistral 계열 모델은 현재 Eval 사용 시 오류가 발생합니다!
        PROCESSING_CONFIG['eval_split_ratio'] = 0
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
    problems = literal_eval(example['problems'])  # 하위 컬럼이 존재하므로 별도의 작업 필요

    # question_plus: 컬럼 우선(기본/CoT 데이터 모두 커버), 없으면 problems에서 fallback
    qp = example.get('question_plus', None)
    if qp is None or (isinstance(qp, float) and pd.isna(qp)):
        qp = problems.get('question_plus', None)

    # cot_text: CoT 학습용 CSV에만 존재 (없으면 None)
    cot = example.get('cot_text', None)
    if cot is not None and isinstance(cot, float) and pd.isna(cot):
        cot = None

    return {
        'id': example['id'],
        'paragraph': example['paragraph'],
        'question': problems['question'],
        'question_plus': qp,
        'choices': problems['choices'],
        'answer': problems.get('answer', None),
        'cot_text': cot,  # CoT 학습용(기본 데이터에서는 None)
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

def build_prompt_cot(data):
    # CoT 학습 프롬프트 (형식/명령 고정)
    question_plus_section = f"\n<보기>:\n{data['question_plus']}" if data['question_plus'] else ""
    choices_string = '\n'.join([f"{i + 1}. {choice}" for i, choice in enumerate(data['choices'])])
    cot_text = data.get("cot_text") or ""

    return PROCESSING_CONFIG['cot_prompt_template'].format(
        paragraph=data['paragraph'],
        question=data['question'],
        question_plus_section=question_plus_section,
        choices=choices_string,
        cot_text=cot_text,
    )


# 메시지 빌더 함수
def build_messages(data, include_answer):
    # 기본은 기존 프롬프트 (기존 동작 보존)
    user_content = build_prompt(data)

    # CoT 학습 옵션: 학습(include_answer=True)일 때만 CoT 프롬프트 사용
    if USE_COT_TRAIN and include_answer and (data.get("cot_text") is not None):
        user_content = build_prompt_cot(data)

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
if USE_COT_TRAIN:
    assert DATA_FILES["train_cot"], "USE_COT_TRAIN=1 인데 COT_TRAIN_PATH가 비어있습니다."
    train_dataset = load_dataset('csv', data_files={'train': DATA_FILES['train_cot']})['train']
else:
    train_dataset = load_dataset('csv', data_files={'train': DATA_FILES['train']})['train']
test_dataset = load_dataset('csv', data_files={'test': DATA_FILES['test']})['test']

# 테스트 데이터셋의 Unnamed: 0 컬럼 제거 (존재할 때만)
if 'Unnamed: 0' in test_dataset.column_names:
    test_dataset = test_dataset.remove_columns(['Unnamed: 0'])

# CoT 학습 데이터셋에도 Unnamed: 0이 있을 수 있으므로 동일 처리
if 'Unnamed: 0' in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns(['Unnamed: 0'])
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
    SFT_CONFIG['eval_strategy'] = 'no'
    print("⚠️ Eval 데이터셋 분할 안 함 (eval_split_ratio = 0)")

# 데이터셋 포맷팅 적용
train_dataset = format_dataset(train_dataset, tokenizer, True, "Formatting Train dataset")
if eval_dataset is not None:
    eval_dataset = format_dataset(eval_dataset, tokenizer, True, "Formatting Eval dataset")
test_dataset = format_dataset(test_dataset, tokenizer, False, "Formatting Test dataset")
print("✅ 데이터셋 포맷팅 완료")
print(f"   포맷팅 이후 컬럼: {train_dataset.column_names}")
vocab = tokenizer.get_vocab()
_answer_ids = [vocab.get(str(i), None) for i in range(1, 6)]
_answer_ids = [i for i in _answer_ids if i is not None]
_answer_ids = set(_answer_ids)

def _has_answer_near_end(example):
    enc = tokenizer(
        example["text"],
        truncation=True,
        max_length=MODEL_LOADER_CONFIG["max_seq_length"],
        add_special_tokens=False,
    )
    ids = enc["input_ids"]
    if not ids:
        return False
    # 끝부분(정답+eos가 있어야 하는 위치) 근처에 1~5 토큰이 존재하는지 확인
    tail = ids[-32:] if len(ids) >= 32 else ids
    return any(t in _answer_ids for t in tail)

_before = len(train_dataset)
train_dataset = train_dataset.filter(_has_answer_near_end, desc="Filtering broken samples (no answer token near end)")
_after = len(train_dataset)
print(f"✅ broken sample filter: {_before} -> {_after} (removed {_before - _after})")
# class SafeSFTTrainer(SFTTrainer):
#     """
#     Unsloth + train_on_responses_only 조합에서
#     어떤 배치가 labels 전부 -100(=학습 토큰 0개)이면 loss가 grad_fn 없이 상수로 떨어져 backward가 터짐.
#     그 배치를 '실질적으로 skip'하도록 0*trainable_param 형태의 더미 loss를 반환해서 안전하게 통과.
#     """
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.get("labels", None)
#         if labels is not None and torch.is_tensor(labels):
#             if (labels != -100).sum().item() == 0:
#                 # ✅ grad_fn을 가진 0-loss 만들기 (배치 스킵 효과)
#                 p = next((p for p in model.parameters() if p.requires_grad), None)
#                 loss = (p.sum() * 0.0) if p is not None else torch.tensor(0.0, device=model.device)
#                 # 디버그 로그(원하면 유지)
#                 if int(os.environ.get("DEBUG_ZERO_LABEL", "1")) == 1:
#                     print("[WARN] zero-label batch detected -> skip via dummy grad loss")
#                 return (loss, None) if return_outputs else loss

#         return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)



### 모델 학습 준비 과정
print("=" * 50)
print("   모델 학습 준비 과정")
print("=" * 50)
# 모델 학습을 위한 Tokenizer 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right' # Training용 padding side, Inference용 padding side는 'left'
tokenizer.truncation_side = "left"


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
    #dataset_text_field = "text",
    #max_seq_length = MODEL_LOADER_CONFIG["max_seq_length"],
    # packing = False,
    args = SFTConfig(**SFT_CONFIG),
)


# Unsloth 마스킹 적용
# trainer = train_on_responses_only(
#     trainer,
#     instruction_part = CHAT_TEMPLATE_CONFIG['instruction_part'],
#     response_part = CHAT_TEMPLATE_CONFIG['response_part'],
# )
#print("✅ 모델 학습 준비 완료")
print("✅ 모델 학습 준비 완료 (response-only masking 비활성화)")

# 간단한 sanity check: 마지막에 정답이 들어가 있는지 확인 (앞부분 포함 학습이라도 OK)
batch = next(iter(trainer.get_train_dataloader()))
txt = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
print("샘플(끝 300자):")
print(txt[-300:])

# ### 학습 내용 검증 과정
# print("=" * 50)
# print("   학습 내용 검증 과정")
# print("=" * 50)

# batch = next(iter(trainer.get_train_dataloader()))
# dl = trainer.get_train_dataloader()
# bad = 0
# for i, b in enumerate(dl):
#     labels = b["labels"]
#     # 배치 단위(여기서는 per_device_train_batch_size=1이라 사실상 1개)
#     n_valid = (labels != -100).sum().item()
#     if n_valid == 0:
#         bad += 1
#         print(f"[WARN] zero-label sample at batch {i}")
#         print(tokenizer.decode(b["input_ids"][0], skip_special_tokens=False)[:2000])
#         if bad >= 3:
#             break
#     if i >= 200:  # 200개 정도만 봐도 충분히 잡힘
#         break

# print(f"✅ sanity scan done. zero-label batches found: {bad}")
# labels = batch["labels"][0]

# learning_ratio = (labels != -100).sum().item() / len(labels)
# learning_text = tokenizer.decode(labels[labels != -100])
# print(f"학습 토큰 비율: {learning_ratio*100:.1f}%")
# print(f"학습 내용: {learning_text}")

# # 마스킹 비율 검사
# if learning_ratio > 0.5:
#     print("⚠️ 마스킹 검증 실패 (학습 비율 50% 초과)")
#     print("→ CHAT_TEMPLATE_CONFIG에서 instruction_part와 response_part를 사용 모델에 맞게 수정하세요.")
#     sys.exit(1)
# else:
#     print("✅ 마스킹 검증 성공")

# # 학습 내용 검사
# stripped = learning_text.strip()
# if not stripped or stripped[0] not in ["1", "2", "3", "4", "5"]:
#     print("⚠️ 학습 내용 검증 실패 (최초 학습 토큰이 1~5 사이의 값이 아님)")
#     print("→ chat_template, apply_chat_template_safe(), response_part 중 어딘가 잘못된 부분이 있습니다!")
#     print("마스킹 전 전체 텍스트 (디버그 용):")
#     print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False))
#     sys.exit(1)
# else:
#     print("✅ 학습 내용 검증 성공")



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
if UPLOAD_HF:
    try:
        hf_model_name = f"{HF_ORG}/{EXP_NAME}"
        print(f"모델 업로드 시작: {hf_model_name}")
        uploaded_urls = upload_all_checkpoints_to_hf(
            output_dir=str(OUTPUT_DIR),
            model_name=hf_model_name,
            experiment_name=EXP_NAME,
        )
        ...
    except Exception as e:
        print(f"⚠️ 모델 업로드 과정에서 오류가 발생하였습니다: {e}")
        raise
else:
    print("⚠️ UPLOAD_HF=0 설정으로 업로드를 건너뜁니다.")