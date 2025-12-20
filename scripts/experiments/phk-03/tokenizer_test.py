from transformers import AutoTokenizer

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("beomi/gemma-ko-2b")

print("=" * 60)
print("beomi/gemma-ko-2b Tokenizer 분석")
print("=" * 60)

# 1. Special tokens 확인
print("\n[1] Special Tokens Map:")
print(tokenizer.special_tokens_map)

# 2. Gemma chat tokens 존재 여부
print("\n[2] Gemma Chat Tokens 확인:")
gemma_tokens = ["<start_of_turn>", "<end_of_turn>"]

for token in gemma_tokens:
    exists = token in tokenizer.get_vocab()
    print(f"  {token}: {'✅ 있음' if exists else '❌ 없음'}")
    
    if exists:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"    → Token ID: {token_id}")

# 3. Llama3 chat tokens 존재 여부
print("\n[3] Llama3 Chat Tokens 확인:")
llama3_tokens = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]

for token in llama3_tokens:
    exists = token in tokenizer.get_vocab()
    print(f"  {token}: {'✅ 있음' if exists else '❌ 없음'}")
    
    if exists:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"    → Token ID: {token_id}")

# 4. 실제 인코딩 테스트
print("\n[4] 인코딩 테스트:")
test_text = "<start_of_turn>user\nHello<end_of_turn>"
tokens = tokenizer.encode(test_text, add_special_tokens=False)
decoded_tokens = [tokenizer.decode([t]) for t in tokens]

print(f"  입력 텍스트: {test_text}")
print(f"  Token 개수: {len(tokens)}개")
print(f"  Token IDs: {tokens}")
print(f"  Decoded tokens: {decoded_tokens}")

# 5. Vocabulary 크기
print(f"\n[5] Vocabulary 크기: {tokenizer.vocab_size:,}")

print("\n" + "=" * 60)
