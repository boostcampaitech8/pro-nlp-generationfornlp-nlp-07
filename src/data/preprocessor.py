"""Data preprocessing utilities"""

from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset


# Prompt templates
PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

SYSTEM_MESSAGE = "지문을 읽고 질문의 답을 구하세요."


def create_prompt(
    paragraph: str,
    question: str,
    choices: List[str],
    question_plus: str = None
) -> str:
    """
    Create prompt from data
    
    Args:
        paragraph: Paragraph text
        question: Question text
        choices: List of choices
        question_plus: Optional question_plus text
        
    Returns:
        Formatted prompt string
    """
    choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(choices)])
    
    if question_plus:
        user_message = PROMPT_QUESTION_PLUS.format(
            paragraph=paragraph,
            question=question,
            question_plus=question_plus,
            choices=choices_string,
        )
    else:
        user_message = PROMPT_NO_QUESTION_PLUS.format(
            paragraph=paragraph,
            question=question,
            choices=choices_string,
        )
    
    return user_message


def prepare_training_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Prepare training data in chat message format
    
    Args:
        df: DataFrame with columns: id, paragraph, question, choices, answer, question_plus
        
    Returns:
        List of dictionaries with 'id', 'messages', and 'label' keys
    """
    processed_dataset = []
    
    for _, row in df.iterrows():
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        
        # <보기>가 있을 때
        if row["question_plus"] and pd.notna(row["question_plus"]):
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )
        
        # chat message 형식으로 변환
        processed_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": f"{row['answer']}"}
                ],
                "label": row["answer"],
            }
        )
    
    return processed_dataset


def prepare_inference_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Prepare inference data in chat message format
    
    Args:
        df: DataFrame with columns: id, paragraph, question, choices, question_plus
        
    Returns:
        List of dictionaries with 'id', 'messages', 'len_choices' keys
    """
    test_dataset = []
    
    for _, row in df.iterrows():
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        len_choices = len(row["choices"])
        
        # <보기>가 있을 때
        if row.get("question_plus") and pd.notna(row["question_plus"]):
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )
        
        test_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_message},
                ],
                "len_choices": len_choices,
            }
        )
    
    return test_dataset

