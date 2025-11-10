#!/usr/bin/env python3
"""
Update Standalone Dataset README

This script updates the README.md for sungyub/big-math-rl-verl with:
- Updated sample counts (242,092)
- Updated file sizes and metadata
- New cleaning statistics section
- Processing date

Usage:
    python scripts/hub/update_standalone_readme.py
"""

import os
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi


def download_current_readme(repo_id: str = "sungyub/big-math-rl-verl") -> str:
    """Download current README from Hub.

    Args:
        repo_id: HuggingFace repo ID

    Returns:
        README content as string
    """
    api = HfApi()

    try:
        readme_content = api.hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="dataset",
        )

        with open(readme_content, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not download existing README: {e}")
        print("Will create new README from scratch.")
        return None


def calculate_metadata(parquet_file: str) -> dict:
    """Calculate metadata from parquet file.

    Args:
        parquet_file: Path to parquet file

    Returns:
        Dictionary with num_examples, num_bytes, etc.
    """
    table = pq.read_table(parquet_file)
    file_size = Path(parquet_file).stat().st_size

    return {
        "num_examples": len(table),
        "num_bytes": table.nbytes,
        "download_size": file_size,
        "dataset_size": table.nbytes,
    }


def create_readme_content(metadata: dict, old_readme: str = None) -> str:
    """Create updated README content.

    Args:
        metadata: Metadata dictionary
        old_readme: Old README content (if available)

    Returns:
        Updated README content
    """
    # YAML frontmatter
    yaml_dict = {
        "license": "apache-2.0",
        "task_categories": [
            "question-answering",
            "reinforcement-learning",
            "text-generation",
        ],
        "tags": [
            "math",
            "reasoning",
            "rlhf",
            "verl",
            "mathematical-reasoning",
            "dataset-quality",
        ],
        "size_categories": [
            "100K<n<1M",
        ],
        "language": "en",
        "pretty_name": "Big Math RL VERL Dataset (Cleaned v2.0)",
        "dataset_info": {
            "features": [
                {"name": "data_source", "dtype": "string"},
                {
                    "name": "prompt",
                    "list": [
                        {"name": "role", "dtype": "string"},
                        {"name": "content", "dtype": "string"},
                    ],
                },
                {"name": "ability", "dtype": "string"},
                {
                    "name": "reward_model",
                    "struct": [
                        {"name": "style", "dtype": "string"},
                        {"name": "ground_truth", "dtype": "string"},
                    ],
                },
                {
                    "name": "extra_info",
                    "struct": [
                        {"name": "split", "dtype": "string"},
                        {"name": "index", "dtype": "int64"},
                        {"name": "source", "dtype": "string"},
                        {"name": "domain", "dtype": "string"},
                        {"name": "solve_rate", "dtype": "float64"},
                    ],
                },
            ],
            "splits": [
                {
                    "name": "train",
                    "num_bytes": metadata["num_bytes"],
                    "num_examples": metadata["num_examples"],
                }
            ],
            "download_size": metadata["download_size"],
            "dataset_size": metadata["dataset_size"],
        },
        "configs": [
            {
                "config_name": "default",
                "data_files": [{"split": "train", "path": "data/train-*"}],
            }
        ],
    }

    # Convert to YAML string
    yaml_str = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True)

    # Markdown content
    markdown_content = f"""
# Big Math RL VERL Dataset (Cleaned v2.0)

A high-quality mathematical reasoning dataset in VERL format, containing {metadata["num_examples"]:,} problems across various mathematical domains including Algebra, Geometry, Calculus, Probability, and Number Theory.

## Dataset Summary

This dataset is derived from SynthLabsAI/Big-Math-RL-Verified and has been extensively cleaned and deduplicated to ensure maximum quality for reinforcement learning training.

**Version 2.0 Updates (2025-11-09)**:
- Applied comprehensive cleaning using `orz-math` preset
- Removed 2 duplicate problems
- Filtered 30 multi-part problems
- Modified 11,432 samples (4.7%) to remove artifacts
- Total samples: {metadata["num_examples"]:,}

## Cleaning Methodology

The dataset underwent aggressive cleaning to remove exam-specific artifacts while preserving mathematical content:

### Artifacts Removed:
- **Problem numbers**: Removed 474 samples with numbering artifacts (e.g., "Problem 6.", "Question 230,")
- **Point allocations**: Removed 128 samples with point markers (e.g., "(8 points)", "[15 points]")
- **Special artifacts**: Removed 19 samples with horizontal rules, translation instructions
- **Trailing artifacts**: Removed 9 samples with category labels, difficulty markers
- **Contest metadata**: Removed 8 samples with competition information
- **Markdown headers**: Removed 3 samples with standalone header keywords

### Samples Filtered:
- **Multi-part problems**: 30 samples with structure like "a) ... b) ..." (0.01%)
- **URL references**: 0 samples (none found)

### Quality Preservation:
- ✅ All mathematical notation preserved (LaTeX)
- ✅ Ground truth answers unchanged
- ✅ Problem semantics maintained
- ✅ VERL schema integrity verified

## Schema

The dataset follows the VERL (Verifiable Explanation Reasoning and Learning) format:

```python
{{
    "data_source": str,              # "deepscaler"
    "prompt": [                      # List of messages
        {{"role": str, "content": str}}
    ],
    "ability": str,                  # "math"
    "reward_model": {{
        "style": str,                # "rule"
        "ground_truth": str          # Expected answer
    }},
    "extra_info": {{
        "split": str,                # "train"
        "index": int,                # Sample index
        "source": str,               # Origin dataset (e.g., "cn_k12", "olympiads")
        "domain": str,               # Math category (e.g., "Algebra", "Geometry")
        "solve_rate": float          # Llama-3.1-8B success rate
    }}
}}
```

## Statistics

- **Total samples**: {metadata["num_examples"]:,}
- **Dataset size**: {metadata["dataset_size"] / (1024**2):.2f} MB
- **Download size**: {metadata["download_size"] / (1024**2):.2f} MB
- **Duplication rate**: 0.00% (2 duplicates removed)
- **Modification rate**: 4.7% (11,432 samples cleaned)

### Domain Distribution

The dataset covers diverse mathematical domains:
- Algebra
- Geometry
- Calculus
- Probability & Statistics
- Number Theory
- Combinatorics

### Difficulty Distribution

Problems are annotated with `solve_rate` (Llama-3.1-8B success rate):
- Easy: solve_rate > 0.7
- Medium: 0.3 < solve_rate ≤ 0.7
- Hard: solve_rate ≤ 0.3

## Usage

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("sungyub/big-math-rl-verl", split="train")

# Access a sample
sample = dataset[0]
problem = sample["prompt"][0]["content"]
answer = sample["reward_model"]["ground_truth"]
domain = sample["extra_info"]["domain"]
difficulty = "Easy" if sample["extra_info"]["solve_rate"] > 0.7 else "Hard"

print(f"Problem: {{problem}}")
print(f"Answer: {{answer}}")
print(f"Domain: {{domain}}")
print(f"Difficulty: {{difficulty}}")
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{big_math_rl_verl_cleaned,
  title={{Big Math RL VERL Dataset (Cleaned v2.0)}},
  author={{sungyub}},
  year={{2025}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/datasets/sungyub/big-math-rl-verl}}}},
  note={{Cleaned version of SynthLabsAI/Big-Math-RL-Verified}}
}}
```

## License

Apache 2.0

## Acknowledgments

- Original dataset: [SynthLabsAI/Big-Math-RL-Verified](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified)
- Cleaning pipeline: DataTrove MathDatasetCleaner
- Processing date: 2025-11-09
"""

    # Combine YAML frontmatter and markdown content
    full_content = f"---\n{yaml_str}---{markdown_content}"

    return full_content


def update_readme(
    parquet_file: str = "./output/hub-upload/big-math-rl-verl/data/train-00000.parquet",
    output_file: str = "./output/hub-upload/big-math-rl-verl/README.md",
    repo_id: str = "sungyub/big-math-rl-verl",
):
    """Update README with new statistics.

    Args:
        parquet_file: Path to parquet file
        output_file: Output README path
        repo_id: HuggingFace repo ID
    """
    print(f"\n{'='*70}")
    print(f"Updating Standalone Dataset README")
    print(f"{'='*70}")
    print(f"Parquet: {parquet_file}")
    print(f"Output:  {output_file}")
    print(f"Repo:    {repo_id}")
    print(f"{'='*70}\n")

    # Download current README (optional)
    print("Step 1: Downloading current README...")
    old_readme = download_current_readme(repo_id)
    if old_readme:
        print(f"✓ Downloaded current README ({len(old_readme)} bytes)")
    else:
        print("⚠ Creating new README")

    # Calculate metadata
    print("\nStep 2: Calculating metadata...")
    metadata = calculate_metadata(parquet_file)
    print(f"✓ Metadata calculated")
    print(f"  Samples: {metadata['num_examples']:,}")
    print(f"  Size: {metadata['dataset_size'] / (1024**2):.2f} MB")

    # Create new README
    print("\nStep 3: Creating updated README...")
    new_readme = create_readme_content(metadata, old_readme)
    print(f"✓ README created ({len(new_readme)} bytes)")

    # Write to file
    print("\nStep 4: Writing to file...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_readme)

    print(f"✓ Saved to {output_file}")

    print(f"\n{'='*70}")
    print(f"✅ README update completed successfully!")
    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    update_readme()

    print("\n📝 Next steps:")
    print("  1. Review README: cat output/hub-upload/big-math-rl-verl/README.md")
    print("  2. Validate: python scripts/hub/validate_before_upload.py --dataset-dir output/hub-upload/big-math-rl-verl")


if __name__ == "__main__":
    main()
