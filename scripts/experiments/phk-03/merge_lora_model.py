from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import torch
import os
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

BASE_MODEL = "upstage/SOLAR-10.7B-Instruct-v1.0"   # 원래 베이스
ADAPTOR_DIR   = "adaptor"
MERGED_DIR = "best_model"

os.makedirs(MERGED_DIR, exist_ok=True)

# 1) 학습 끝난 토크나이저 로드 (여기에 스페셜 토큰 추가가 반영돼 있음)
tokenizer = AutoTokenizer.from_pretrained(ADAPTOR_DIR)

# 2) 베이스 모델 로드 + vocab 크기를 토크나이저에 맞게 리사이즈
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype = torch.bfloat16,   # 필요에 따라 "float16" / "auto" 등으로 조정
    device_map  = "auto",      # V100 메모리 아끼고 싶으면 CPU에서 머지
)
base_model.resize_token_embeddings(len(tokenizer))

# 3) 베이스 위에 LoRA 어댑터 로드
peft_model = PeftModel.from_pretrained(
    base_model,
    ADAPTOR_DIR,
    device_map  = "auto",
)

# 4) LoRA를 weight에 머지하고 어댑터 제거
merged_model = peft_model.merge_and_unload()  # PeftModel → 일반 CausalLM

# 5) 머지된 모델 + 토크나이저 저장
merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("✅ Merged model saved to:", MERGED_DIR)
