"""Submission file creation utilities"""

import pandas as pd
from pathlib import Path
from typing import List, Dict


def create_submission(
    predictions: List[Dict[str, str]],
    output_path: str = "submissions/output.csv"
) -> pd.DataFrame:
    """
    Create submission CSV file
    
    Args:
        predictions: List of predictions with 'id' and 'answer' keys
        output_path: Path to save submission file
        
    Returns:
        DataFrame with submission data
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    
    print(f"Submission file saved to {output_path}")
    return df

