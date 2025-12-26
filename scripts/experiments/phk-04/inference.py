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
import argparse
import json

# 해당 파일은 scripts/experiments/memberA/ 폴더에 위치한 것이므로,
# project root를 따로 추가해줍니다.
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.hf_utils import upload_all_checkpoints_to_hf

load_dotenv()

### 명령줄 인자 파싱
# Boolean 타입 인자 처리를 위한 함수
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean 값이 아닙니다!')

# 인자 등록
parser = argparse.ArgumentParser(description="LLM PEFT Training and Inference Script w.Unsloth")
# 필수 인자
parser.add_argument('--exp_name', type=str, required=True, help='실험 이름')
parser.add_argument('--model_name', type=str, required=True, help='모델 이름 또는 경로')
# 모델 로더 설정
parser.add_argument('--max_seq_length', type=int, default=4096, help='최대 시퀀스 길이')
parser.add_argument('--load_in_4bit', type=str2bool, default=True, help='4비트 양자화 사용 여부')
# LoRA 설정
parser.add_argument('--target_modules', type=str, nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                    help='LoRA 타겟 모듈 (기본값: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)')
parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank (기본값: 8)')
parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha (기본값: 16, 보통 r의 1~2배)')
parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout (기본값: 0.0)')
# SFT 설정
parser.add_argument('--learning_rate', type=float, default=2e-5, help='학습률 (기본값: 2e-5)')
parser.add_argument('--num_train_epochs', type=int, default=3, help='학습 에포크 수 (기본값: 3)')
parser.add_argument('--per_device_train_batch_size', type=int, default=1, help='디바이스당 학습 배치 크기 (기본값: 1)')
parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help='디바이스당 평가 배치 크기 (기본값: 1)')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='그래디언트 누적 스텝 (기본값: 1)')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (기본값: 0.01)')
# 데이터 설정
parser.add_argument('--train_data', type=str, default='train.csv', help='학습 데이터 파일 이름 (기본값: train.csv)')
parser.add_argument('--eval_split_ratio', type=float, default=0.0, help='Eval 데이터 분할 비율 (기본값: 0.0)')
parser.add_argument('--is_cot_data', type=str2bool, default=False, help='CoT 데이터 여부 (기본값: False)')

args = parser.parse_args()


### 직접 수정 가능한 변수들은 실험하기 편하게 상단에 모아 두었습니다.
# 상수
RANDOM_STATE = 42
CAMPER_ID = "T8091"
EXP_NAME = args.exp_name
HF_ORG = "NLP-07-ODQA"

# 경로
OUTPUT_DIR = project_root / "outputs" / CAMPER_ID / EXP_NAME
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
SUBMISSION_DIR = project_root / "submissions" / CAMPER_ID

# 모델 로더 설정값
MODEL_LOADER_CONFIG = {
    "model_name": args.model_name,
    "max_seq_length": args.max_seq_length, # 현재 데이터의 시퀀스 길이가 대부분 500~3000 사이이므로, 그 이상으로 설정합니다.
    "dtype": torch.float16, # V100 사용중이므로 Float16 기본 사용
    "load_in_4bit": args.load_in_4bit,  # Use 4bit quantization to reduce memory usage. Can be False.
    # token = "hf_...",     # 승인이 필요한 모델을 사용하는 경우 허깅페이스 토큰이 필요하다는 뜻인 것 같습니다. (원문: use one if using gated models like meta-llama/Llama-2-7b-hf)
}

# LoRA 어댑터 설정값
LORA_CONFIG = {
    "target_modules": args.target_modules,      # 모듈 종류: "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    "r": args.lora_r,                           # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    "lora_alpha": args.lora_alpha,              # 보통 r 값과 동일하거나 2배로 설정
    "lora_dropout": args.lora_dropout,          # Supports any, but = 0 is optimized
    "bias": "none",                             # Supports any, but = "none" is optimized
    "use_gradient_checkpointing": "unsloth",    # True or "unsloth" for very long context
    "random_state": RANDOM_STATE,
    "use_rslora": False,                        # We support rank stabilized LoRA
    "loftq_config": None,                       # And LoftQ
}

# SFT 설정값
SFT_CONFIG = {
    "output_dir": OUTPUT_DIR,
    "lr_scheduler_type": "cosine",
    "learning_rate": args.learning_rate,
    "num_train_epochs": args.num_train_epochs,
    "per_device_train_batch_size": args.per_device_train_batch_size,
    "per_device_eval_batch_size": args.per_device_eval_batch_size,
    # "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "weight_decay": args.weight_decay,
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
    "train": str(project_root / "data" / "train" / args.train_data),
    "test": str(project_root / "data" / "test" / "test.csv"),
}

# chat_template마다 system role의 지원 여부가 다르므로, system_prompt를 user role의 맨 처음 부분에 통합하였습니다.
# 또한, 전처리 로직에서 현재 프롬프트는 question_plus 컬럼의 존재 여부, choices 컬럼의 갯수에 따라 최종 내용을 다르게 처리합니다.
PROCESSING_CONFIG = {
    "eval_split_ratio": args.eval_split_ratio,          # Train 데이터셋에서 Evaluation 데이터셋으로 분할할 비율 (0으로 설정 시 분할하지 않음: 자동적으로 Eval도 수행 안함)
    "is_cot_data": args.is_cot_data,
    "system_prompt": "지문을 읽고 질문의 답을 구하세요.",   # 시스템 프롬프트는 User role의 맨 앞에 추가됩니다 (chat_template마다 system role의 지원 여부가 다르므로)
    "prompt_template": """{system_prompt}


지문:
{paragraph}


질문:
{question}
{question_plus_section}


선택지:
{choices}
{cot_section}


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

# 토크나이저 토큰 점검
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("⚠️ 토크나이저 pad_token이 없어서 eos_token으로 설정됨")

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
    
    # Solar 계열
    elif 'solar' in model_name:
        return '### User:\n', '### Assistant:\n'

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
        'cot_text': problems.get('cot_text', None),
    }

# 프롬프트 빌더 함수 (기본 데이터셋 형식에 맞춰져 있습니다.)
# 사용하려는 데이터셋의 컬럼 구성이 다르다면 이 함수를 반드시 수정해야 합니다!
def build_prompt(data):
    question_plus_section = f"\n\n<보기>:\n{data['question_plus']}" if data['question_plus'] else ""
    choices_string = '\n'.join([f"{i + 1}. {choice}" for i, choice in enumerate(data['choices'])])
    choice_range = ', '.join(map(str, range(1, len(data['choices']) + 1)))
    cot_section = f"\n\n문제 풀이 과정:\n{data['cot_text'] or '단계적으로 생각해봅시다.'}" if PROCESSING_CONFIG['is_cot_data'] else ""

    return PROCESSING_CONFIG['prompt_template'].format(
        system_prompt=PROCESSING_CONFIG['system_prompt'],
        paragraph=data['paragraph'],
        question=data['question'],
        question_plus_section=question_plus_section,
        choices=choices_string,
        choice_range=choice_range,
        cot_section=cot_section,
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
            'cot_text': example['cot_text'][idx],
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
test_dataset = load_dataset('csv', data_files={'test': DATA_FILES['test']})['test']

# 테스트 데이터셋의 Unnamed: 0 컬럼 제거
test_dataset = test_dataset.remove_columns(['Unnamed: 0'])

print("✅ 데이터셋 로드 완료")
print(f"   Test: {len(test_dataset)}")

test_dataset = test_dataset.map(parse_data, remove_columns=['problems'], desc="Parsing dataset")
print("✅ 데이터셋 파싱 완료")

# 데이터셋 포맷팅 적용
test_dataset = format_dataset(test_dataset, tokenizer, False, "Formatting Test dataset")
print("✅ 데이터셋 포맷팅 완료")
print(f"   포맷팅 이후 컬럼: {test_dataset.column_names}")


### 추론 과정
print("=" * 50)
print("   추론 준비 과정")
print("=" * 50)
# 추론 모드 활성화
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
detailed_results = []
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
            pred_idx = torch.argmax(probs).item()
            pred = pred_map[pred_idx]
            predictions.append(pred)

            sample_id = test_ids[i + j]
            prob_dict = {str(k + 1): probs[k].item() for k in range(len_choices)}

            detailed_results.append({
                'id': sample_id,
                'prediction': pred,
                'probabilities': prob_dict,
                'confidence': probs[pred_idx].item(),
                'num_choices': len_choices,
            })

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

# 신뢰도 통계 출력
confidences = [r['confidence'] for r in detailed_results]
print(f"\n신뢰도 통계:")
print(f"  평균: {sum(confidences)/len(confidences):.4f}")
print(f"  최소: {min(confidences):.4f}")
print(f"  최대: {max(confidences):.4f}")

submission_file = SUBMISSION_DIR / f"{EXP_NAME}.csv"
output_df.to_csv(submission_file, index=False)
print(f"✅ 추론 결과 저장됨: {submission_file}")

json_output = {
    'experiment_name': EXP_NAME,
    'total_samples': len(test_ids),
    'predictions': detailed_results,
    'summary': {
        'answer_distribution': output_df['answer'].value_counts().sort_index().to_dict(),
        'average_confidence': sum(r['confidence'] for r in detailed_results) / len(detailed_results)
    }
}

json_file = SUBMISSION_DIR / f"{EXP_NAME}_detailed.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)
print(f"✅ JSON 저장됨: {json_file}")