#!/usr/bin/env python3
"""
Analyze sample quality of maximum cleaned OpenR1-Math dataset.
Provides detailed quality assessment with specific examples.
"""

import random
import re
from datasets import load_dataset

def detect_quality_issues(text: str) -> dict:
    """Detect potential quality issues in a sample."""
    issues = {
        'has_url': bool(re.search(r'https?://|www\.', text)),
        'has_image_ref': bool(re.search(r'!\[.*?\]\(|see Figure|shown in Diagram|\[asy\]', text, re.IGNORECASE)),
        'has_multipart': bool(re.search(r'(?:^|\n)\s*(?:[a-z]\)|\([IVXivx]+\))\s+[A-Z]', text, re.MULTILINE)),
        'has_problem_number': bool(re.search(r'^(?:Problem|Question|Exercise|Task)\s+\d+', text, re.MULTILINE)),
        'has_trailing_header': bool(re.search(r'\n+#{1,6}\s+[^\n]+$', text)),
        'has_author': bool(re.search(r'\$\\underline\{\\text\s*\{[^\}]+\}\}\$', text)),
        'very_short': len(text.strip()) < 50,
        'very_long': len(text.strip()) > 2000,
    }
    return issues

def categorize_sample(text: str, issues: dict) -> str:
    """Categorize sample quality."""
    if issues['has_url'] or issues['has_multipart']:
        return "SHOULD_BE_FILTERED"  # These should have been removed
    elif issues['has_image_ref']:
        return "IMAGE_DEPENDENT"
    elif any([issues['has_problem_number'], issues['has_trailing_header'], issues['has_author']]):
        return "HAS_ARTIFACTS"
    elif issues['very_short']:
        return "TOO_SHORT"
    elif issues['very_long']:
        return "VERY_LONG"
    else:
        return "CLEAN"

def analyze_dataset(parquet_path: str, num_random: int = 15, num_per_category: int = 5):
    """Analyze dataset quality with categorized examples."""

    print("Loading dataset...")
    ds = load_dataset('parquet', data_files=parquet_path, split='train')
    total_samples = len(ds)

    print(f"Total samples: {total_samples:,}\n")

    # Categorize all samples
    categories = {
        'CLEAN': [],
        'IMAGE_DEPENDENT': [],
        'HAS_ARTIFACTS': [],
        'TOO_SHORT': [],
        'VERY_LONG': [],
        'SHOULD_BE_FILTERED': [],
    }

    print("Analyzing all samples...")
    for idx in range(total_samples):
        sample = ds[idx]
        prompt_content = sample['prompt'][0]['content']
        issues = detect_quality_issues(prompt_content)
        category = categorize_sample(prompt_content, issues)
        categories[category].append((idx, sample, issues))

    # Print statistics
    print("\n" + "="*70)
    print("Quality Distribution")
    print("="*70)
    for category, samples in categories.items():
        count = len(samples)
        pct = (count / total_samples) * 100
        print(f"{category:20s}: {count:6,} samples ({pct:5.2f}%)")

    print("\n" + "="*70)
    print("Sample Examples by Category")
    print("="*70)

    # Show examples from each category
    for category in ['CLEAN', 'IMAGE_DEPENDENT', 'HAS_ARTIFACTS', 'TOO_SHORT', 'VERY_LONG', 'SHOULD_BE_FILTERED']:
        samples_in_category = categories[category]
        if not samples_in_category:
            continue

        print(f"\n{'─'*70}")
        print(f"Category: {category} ({len(samples_in_category):,} samples)")
        print('─'*70)

        # Sample up to num_per_category examples
        num_to_show = min(num_per_category, len(samples_in_category))
        sampled = random.sample(samples_in_category, num_to_show) if len(samples_in_category) > num_per_category else samples_in_category

        for i, (idx, sample, issues) in enumerate(sampled, 1):
            prompt_content = sample['prompt'][0]['content']
            ground_truth = sample.get('reward_model', {}).get('ground_truth', 'N/A')

            # Truncate very long samples
            display_text = prompt_content if len(prompt_content) <= 500 else prompt_content[:500] + "..."

            print(f"\nExample {i} (Index: {idx})")
            print(f"Issues: {[k for k, v in issues.items() if v]}")
            print(f"\nProblem:")
            print(display_text)
            print(f"\nGround Truth: {ground_truth}")
            print()

    # Overall quality score
    print("\n" + "="*70)
    print("Overall Quality Assessment")
    print("="*70)

    clean_rate = (len(categories['CLEAN']) / total_samples) * 100
    problematic_rate = (len(categories['IMAGE_DEPENDENT']) + len(categories['HAS_ARTIFACTS']) +
                       len(categories['SHOULD_BE_FILTERED'])) / total_samples * 100

    print(f"Clean samples:       {len(categories['CLEAN']):6,} ({clean_rate:.2f}%)")
    print(f"Image-dependent:     {len(categories['IMAGE_DEPENDENT']):6,} ({len(categories['IMAGE_DEPENDENT'])/total_samples*100:.2f}%)")
    print(f"Has artifacts:       {len(categories['HAS_ARTIFACTS']):6,} ({len(categories['HAS_ARTIFACTS'])/total_samples*100:.2f}%)")
    print(f"Should be filtered:  {len(categories['SHOULD_BE_FILTERED']):6,} ({len(categories['SHOULD_BE_FILTERED'])/total_samples*100:.2f}%)")
    print(f"\nOverall quality score: {clean_rate:.1f}% clean")

    if len(categories['SHOULD_BE_FILTERED']) > 0:
        print(f"\n⚠️  WARNING: {len(categories['SHOULD_BE_FILTERED'])} samples should have been filtered but weren't!")

if __name__ == "__main__":
    parquet_path = "output/openr1-cleaned-maxclean/train.parquet"
    analyze_dataset(parquet_path, num_random=15, num_per_category=3)
