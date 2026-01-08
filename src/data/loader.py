"""Data loading utilities"""

import pandas as pd
from ast import literal_eval
from pathlib import Path
from typing import List, Dict, Any


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data file
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def flatten_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten JSON dataset by parsing 'problems' column
    
    Args:
        dataset: DataFrame with 'problems' column containing JSON strings
        
    Returns:
        Flattened DataFrame
    """
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
            'question_plus': problems.get('question_plus', None),
        }
        # Include 'question_plus' if it exists
        if 'question_plus' in problems:
            record['question_plus'] = problems['question_plus']
        records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df

