"""
Utility functions for VERL dataset deduplication.

This module provides core functionality for text normalization, hashing,
and VERL format validation for any VERL-formatted dataset.
"""

import hashlib
import json
import os
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import Counter
import numpy as np


def normalize_text(text: str) -> str:
    """
    Normalize problem text for fair comparison.
    Conservative approach to minimize false positives.

    Args:
        text: Raw problem text from prompt[0]['content']

    Returns:
        Normalized text string
    """
    # 1. Strip leading/trailing whitespace
    text = text.strip()

    # 2. Normalize internal whitespace (multiple → single space)
    text = ' '.join(text.split())

    # 3. Remove LaTeX formatting variations
    text = text.replace('\\\\', '\\')      # Double backslash
    text = text.replace('\\,', '')          # Thin space
    text = text.replace('\\!', '')          # Negative space
    text = text.replace('\\;', '')          # Medium space
    text = text.replace('\\quad', ' ')      # Quad space
    text = text.replace('\\qquad', '  ')    # Double quad

    # 4. Normalize quotes
    text = text.replace('"', '"').replace('"', '"')  # Smart quotes
    text = text.replace(''', "'").replace(''', "'")  # Smart apostrophes

    # 5. Normalize dashes
    text = text.replace('–', '-').replace('—', '-')  # En/em dash

    # 6. Remove zero-width characters
    text = text.replace('\u200b', '')       # Zero-width space
    text = text.replace('\ufeff', '')       # BOM

    # NOT converting to lowercase (preserves LaTeX case sensitivity)

    return text


def compute_hash(text: str) -> str:
    """
    Compute SHA-256 hash of normalized text.

    Args:
        text: Problem text to hash

    Returns:
        Hexadecimal hash string (64 characters)
    """
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def extract_problem_text(row: Dict[str, Any]) -> str:
    """
    Extract problem text from VERL format row.

    Handles both flat and nested (DataTrove Document) formats:
    - Flat: {'prompt': [...], 'data_source': ..., ...}
    - Nested: {'id': ..., 'metadata': {'prompt': [...], 'data_source': ..., ...}}

    Args:
        row: Dictionary containing VERL formatted data

    Returns:
        Problem text string

    Raises:
        KeyError: If expected fields are missing
        IndexError: If prompt array is empty
    """
    try:
        # Check if this is a nested DataTrove Document format
        if 'metadata' in row and isinstance(row.get('metadata'), dict):
            return row['metadata']['prompt'][0]['content']
        # Otherwise assume flat format
        return row['prompt'][0]['content']
    except (KeyError, IndexError) as e:
        raise ValueError(f"Invalid VERL format: {e}")


def validate_verl_row(
    row: Dict[str, Any],
    allowed_abilities: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Validate that a row conforms to VERL schema.

    Handles both flat and nested (DataTrove Document) formats:
    - Flat: {'prompt': [...], 'data_source': ..., ...}
    - Nested: {'id': ..., 'metadata': {'prompt': [...], 'data_source': ..., ...}}

    Args:
        row: Dictionary to validate
        allowed_abilities: List of allowed ability values (None = allow any)

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Determine if this is nested format and extract data
        if 'metadata' in row and isinstance(row.get('metadata'), dict):
            # Nested DataTrove Document format
            data = row['metadata']
        else:
            # Flat format
            data = row

        # Check required top-level fields
        required_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

        # Check prompt structure (can be list or numpy array from pandas)
        if not isinstance(data['prompt'], (list, np.ndarray)):
            return False, "prompt must be a list or array"

        if len(data['prompt']) == 0:
            return False, "prompt array is empty"

        if 'role' not in data['prompt'][0]:
            return False, "prompt[0] missing 'role' field"

        # Allow both 'user' and 'system' roles (toolrl datasets use system prompts)
        if data['prompt'][0]['role'] not in ['user', 'system']:
            return False, f"prompt[0]['role'] must be 'user' or 'system', got '{data['prompt'][0]['role']}'"

        if 'content' not in data['prompt'][0]:
            return False, "prompt[0] missing 'content' field"

        # Check ability (if allowed_abilities specified)
        if allowed_abilities is not None:
            if data['ability'] not in allowed_abilities:
                return False, f"ability must be one of {allowed_abilities}, got '{data['ability']}'"

        # Check reward_model
        if 'ground_truth' not in data['reward_model']:
            return False, "reward_model missing 'ground_truth' field"

        if 'style' not in data['reward_model']:
            return False, "reward_model missing 'style' field"

        return True, ""

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def format_bytes(bytes_size: int) -> str:
    """
    Format bytes to human-readable string.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def format_number(num: int) -> str:
    """
    Format large numbers with comma separators.

    Args:
        num: Integer to format

    Returns:
        Formatted string (e.g., "1,234,567")
    """
    return f"{num:,}"


def format_percentage(numerator: int, denominator: int) -> str:
    """
    Calculate and format percentage.

    Args:
        numerator: Numerator value
        denominator: Denominator value

    Returns:
        Formatted percentage string (e.g., "45.67%")
    """
    if denominator == 0:
        return "N/A"
    percentage = (numerator / denominator) * 100
    return f"{percentage:.2f}%"


def save_stats(stats: Dict[str, Any], output_path: str) -> None:
    """
    Save statistics to JSON file.

    Args:
        stats: Statistics dictionary
        output_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Statistics saved to: {output_path}")


def load_stats(stats_path: str) -> Dict[str, Any]:
    """
    Load statistics from JSON file.

    Args:
        stats_path: Path to JSON file

    Returns:
        Statistics dictionary
    """
    with open(stats_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_hash_collision(hash_to_text: Dict[str, str], new_hash: str, new_text: str) -> bool:
    """
    Check if a hash collision has occurred.

    Args:
        hash_to_text: Mapping of hashes to original text
        new_hash: Hash to check
        new_text: Text corresponding to new_hash

    Returns:
        True if collision detected, False otherwise
    """
    if new_hash in hash_to_text:
        existing_text = hash_to_text[new_hash]
        if existing_text != new_text:
            print(f"⚠️  HASH COLLISION DETECTED!")
            print(f"Hash: {new_hash}")
            print(f"Text 1: {existing_text[:100]}...")
            print(f"Text 2: {new_text[:100]}...")
            return True
    return False


def estimate_memory_usage(num_hashes: int) -> str:
    """
    Estimate memory usage for hash set.

    Args:
        num_hashes: Number of hashes to store

    Returns:
        Formatted memory estimate string
    """
    # SHA-256 hash = 64 bytes (hex string)
    # Python overhead ~2x
    bytes_per_hash = 64 * 2
    total_bytes = num_hashes * bytes_per_hash
    return format_bytes(total_bytes)


def print_progress_bar(current: int, total: int, prefix: str = '', suffix: str = '',
                       length: int = 50, fill: str = '█') -> None:
    """
    Print a progress bar to console.

    Args:
        current: Current progress value
        total: Total/max value
        prefix: Prefix string
        suffix: Suffix string
        length: Bar length in characters
        fill: Fill character
    """
    if total == 0:
        return

    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)

    if current == total:
        print()  # New line on completion


class DuplicationStats:
    """Track deduplication statistics."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.total_rows = 0
        self.unique_rows = 0
        self.duplicate_rows = 0
        self.hash_set: Set[str] = set()
        self.duplicate_samples: List[Tuple[str, str, int]] = []  # (hash, text, count)
        self.duplicate_counts = Counter()

    def add_row(self, problem_hash: str, problem_text: str, is_duplicate: bool) -> None:
        """Record a processed row."""
        self.total_rows += 1

        if is_duplicate:
            self.duplicate_rows += 1
            self.duplicate_counts[problem_hash] += 1
        else:
            self.unique_rows += 1
            self.hash_set.add(problem_hash)

    def get_duplicate_rate(self) -> float:
        """Calculate duplicate rate."""
        if self.total_rows == 0:
            return 0.0
        return self.duplicate_rows / self.total_rows

    def get_top_duplicates(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N most frequent duplicates."""
        top = self.duplicate_counts.most_common(n)
        return [
            {
                'hash': hash_val,
                'occurrences': count
            }
            for hash_val, count in top
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'dataset': self.dataset_name,
            'total_rows': self.total_rows,
            'unique_rows': self.unique_rows,
            'duplicate_rows': self.duplicate_rows,
            'duplicate_rate': self.get_duplicate_rate(),
            'top_duplicates': self.get_top_duplicates()
        }

    def print_summary(self) -> None:
        """Print statistics summary."""
        print(f"\n{'=' * 60}")
        print(f"Dataset: {self.dataset_name}")
        print(f"{'=' * 60}")
        print(f"Total rows:      {format_number(self.total_rows)}")
        print(f"Unique rows:     {format_number(self.unique_rows)}")
        print(f"Duplicate rows:  {format_number(self.duplicate_rows)}")
        print(f"Duplicate rate:  {format_percentage(self.duplicate_rows, self.total_rows)}")
        print(f"{'=' * 60}\n")
