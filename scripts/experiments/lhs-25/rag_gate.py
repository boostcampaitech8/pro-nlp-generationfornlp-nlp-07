import torch
@torch.inference_mode()
def decide_use_rag(
    model,
    tokenizer,
    paragraph: str,
    question: str,
    question_plus: str | None,
    choices: list[str],
    max_len: int = 2048,
) -> str:
    """
    Return exactly one of:
    - USE_RAG
    - NO_RAG
    """

    qps = f"\n[보기]\n{question_plus}" if question_plus else ""

    prompt = f"""
너는 수능 문제를 푸는 AI가 아니라,
"이 문제를 풀기 위해 외부 지식 검색(RAG)이 필요한지"만 판단하는 분류기다.

아래 문제를 보고 판단하라.

판단 기준:
- 지문과 보기, 선택지에 **정답에 필요한 정보가 모두 있으면** → NO_RAG
- 특정 개념, 사실, 배경지식, 정의, 연도, 인물, 사건 등
  **외부 지식이 없으면 정답을 알 수 없으면** → USE_RAG

중요 규칙:
- 문제를 풀지 마라
- 정답 번호를 고르지 마라
- 설명하지 마라
- 반드시 아래 둘 중 하나만 출력하라

출력 형식 (이 줄만 출력):
USE
또는
NO

[지문]
{paragraph}

[질문]
{question}
{qps}

[선택지]
{chr(10).join(choices)}

출력:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=2,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    print(text)
    if "USE" in text:
        return "USE"
    return "NO"
