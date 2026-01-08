"""LLM Wrapper (Thinking 모드 비활성화)"""
import torch
from typing import Dict, Any
from utils.memory import cleanup_memory


class MultipleChoiceLLM:
    """객관식 문제를 위한 LLM 래퍼 (Thinking 모드 비활성화)"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.last_probs = None
        self.last_answer = None
        self.last_confidence = None
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0
    ) -> str:
        """생성 수행 (Thinking 모드 비활성화)"""
        # Qwen3 모델의 경우 enable_thinking=False 설정
        messages = [{"role": "user", "content": prompt}]
        
        # chat_template 적용 (enable_thinking=False)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Thinking 모드 비활성화
        )
        
        # 입력 길이 확인 (로깅)
        tokenized_check = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_length = tokenized_check.input_ids.shape[1]
        max_length = 8192
        
        if input_length > max_length:
            print(f"⚠️ 경고: 입력 길이 {input_length} 토큰이 {max_length}를 초과합니다. 잘림 가능성 있음.")
        else:
            print(f"📏 입력 길이: {input_length} 토큰 (최대: {max_length})")
        
        del tokenized_check  # 메모리 정리
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 디코딩
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        # 메모리 정리
        del inputs, outputs
        cleanup_memory()
        
        return response.strip()
    
    def predict_choice(
        self,
        prompt: str,
        num_choices: int = 5
    ) -> Dict[str, Any]:
        """선택지 예측 (logits 기반)"""
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # 입력 길이 확인 (로깅)
        tokenized_check = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_length = tokenized_check.input_ids.shape[1]
        max_length = 8192
        
        if input_length > max_length:
            print(f"⚠️ 경고: 입력 길이 {input_length} 토큰이 {max_length}를 초과합니다. 잘림 가능성 있음.")
        else:
            print(f"📏 입력 길이: {input_length} 토큰 (최대: {max_length})")
        
        del tokenized_check  # 메모리 정리
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            
            # 선택지 토큰 ID 추출
            choice_tokens = [str(i) for i in range(1, num_choices + 1)]
            choice_token_ids = []
            for choice in choice_tokens:
                token_id = self.tokenizer.encode(choice, add_special_tokens=False)
                if token_id:
                    choice_token_ids.append(token_id[0])
                else:
                    # 폴백: vocab에서 직접 찾기
                    vocab = self.tokenizer.get_vocab()
                    if choice in vocab:
                        choice_token_ids.append(vocab[choice])
            
            if not choice_token_ids:
                # 최종 폴백: 1~5를 직접 인코딩
                choice_token_ids = [
                    self.tokenizer.encode(str(i), add_special_tokens=False)[0]
                    for i in range(1, num_choices + 1)
                ]
            
            # Logits 추출 및 확률 계산
            choice_logits = logits[0, choice_token_ids]
            choice_probs = torch.nn.functional.softmax(choice_logits, dim=-1)
            
            self.last_probs = {
                choice: prob.item() 
                for choice, prob in zip(choice_tokens, choice_probs)
            }
            
            predicted_idx = torch.argmax(choice_probs).item()
            self.last_answer = choice_tokens[predicted_idx]
            self.last_confidence = choice_probs[predicted_idx].item()
        
        # 메모리 정리
        del inputs, outputs, logits
        cleanup_memory()
        
        return {
            'probs': self.last_probs,
            'answer': self.last_answer,
            'confidence': self.last_confidence
        }

