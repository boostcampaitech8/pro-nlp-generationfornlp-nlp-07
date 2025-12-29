"""Data processing module"""

from .loader import load_data, flatten_dataset
from .preprocessor import prepare_training_data, prepare_inference_data, PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS
from .tokenizer import tokenize_dataset, filter_by_length
from .eda import check_missing_values, analyze_question_length, compute_tfidf
from .wikipedia_parser import (
    parse_wikipedia_xml_stream,
    collect_statistics,
    analyze_wikitext_tags,
    analyze_language_content,
    get_namespace_name,
)
from .wikipedia_cleaner import clean_wikitext, extract_sections
from .rag_dataset_creator import (
    create_rag_dataset,
    create_cleaned_dataset,
    create_chunks_from_cleaned,
    upload_cleaned_dataset_to_hf,
)

__all__ = [
    "load_data",
    "flatten_dataset",
    "prepare_training_data",
    "prepare_inference_data",
    "PROMPT_NO_QUESTION_PLUS",
    "PROMPT_QUESTION_PLUS",
    "tokenize_dataset",
    "filter_by_length",
    "check_missing_values",
    "analyze_question_length",
    "compute_tfidf",
    "parse_wikipedia_xml_stream",
    "collect_statistics",
    "analyze_wikitext_tags",
    "analyze_language_content",
    "get_namespace_name",
    "clean_wikitext",
    "extract_sections",
    "create_rag_dataset",
    "create_cleaned_dataset",
    "create_chunks_from_cleaned",
    "upload_cleaned_dataset_to_hf",
]

