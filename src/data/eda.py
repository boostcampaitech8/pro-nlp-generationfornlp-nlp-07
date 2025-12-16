"""Exploratory Data Analysis utilities"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Optional


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in dataset
    
    Args:
        df: DataFrame to check
        
    Returns:
        Series with missing value counts
    """
    missing_values = df.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values)
    return missing_values


def analyze_question_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze question length distribution
    
    Args:
        df: DataFrame with 'question' and optional 'question_plus' columns
        
    Returns:
        DataFrame with added 'question_length' column
    """
    # Fill NaN values
    df = df.copy()
    df['question_plus'] = df['question_plus'].fillna('')
    
    # Combine 'question' and 'question_plus' if available
    df['full_question'] = df.apply(
        lambda x: x['question'] + ' ' + x['question_plus'] if x['question_plus'] else x['question'],
        axis=1
    )
    
    # Calculate the length of each question
    df['question_length'] = df['full_question'].apply(len)
    
    return df


def plot_question_length_distribution(df: pd.DataFrame, bins: int = 30, figsize: Tuple[int, int] = (5, 3)):
    """
    Plot question length distribution
    
    Args:
        df: DataFrame with 'question_length' column
        bins: Number of bins for histogram
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.hist(df['question_length'], bins=bins, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Question Lengths')
    plt.xlabel('Question Length')
    plt.ylabel('Frequency')
    plt.show()


def compute_tfidf(
    df: pd.DataFrame,
    text_column: str = 'full_question',
    max_features: int = 1000
) -> Tuple[TfidfVectorizer, pd.DataFrame]:
    """
    Compute TF-IDF features
    
    Args:
        df: DataFrame with text column
        text_column: Name of text column
        max_features: Maximum number of features
        
    Returns:
        Tuple of (TfidfVectorizer, DataFrame with TF-IDF features)
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out()
    )
    
    print("\nTF-IDF Features:")
    print(tfidf_df.head(20))
    
    return tfidf_vectorizer, tfidf_df

