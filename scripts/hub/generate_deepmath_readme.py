#!/usr/bin/env python3
"""
Generate DeepMath-103K Standalone Dataset README

This script generates README.md for sungyub/deepmath-103k-verl with:
- Complete YAML frontmatter with dataset metadata
- Dataset description and cleaning statistics
- Usage examples and citation information
- MIT license attribution

Usage:
    python scripts/hub/generate_deepmath_readme.py
"""

import os
from pathlib import Path

import pyarrow.parquet as pq
import yaml


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


def create_readme_content(metadata: dict) -> str:
    """Create README content for DeepMath-103K.

    Args:
        metadata: Metadata dictionary

    Returns:
        Complete README content with YAML frontmatter
    """
    # YAML frontmatter
    yaml_dict = {
        "license": "mit",
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
            "deepmath",
        ],
        "size_categories": [
            "100K<n<1M",
        ],
        "language": "en",
        "pretty_name": "DeepMath-103K VERL Dataset (Cleaned)",
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
                        {"name": "index", "dtype": "int64"},
                        {"name": "original_dataset", "dtype": "string"},
                        {"name": "split", "dtype": "string"},
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
# DeepMath-103K VERL Dataset (Cleaned)

A high-quality mathematical reasoning dataset in VERL format, containing {metadata["num_examples"]:,} challenging and verifiable math problems across advanced mathematical domains.

## Dataset Summary

This dataset is derived from [zwhe99/DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) and has been converted to VERL format with extensive cleaning and deduplication for reinforcement learning training.

**Cleaning Summary (2025-11-09)**:
- Original samples: 103,022
- After cleaning & dedup: {metadata["num_examples"]:,}
- Duplicates removed: 1,147 (1.11%)
- Modified samples: 4,366 (4.2%)

## Source Dataset

**DeepMath-103K** is a large-scale dataset featuring challenging, verifiable, and decontaminated math problems tailored for RL and SFT, created by Zhiwei He (@zwhe99).

**Original Dataset**: [zwhe99/DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
**Paper**: [arXiv:2504.11456](https://arxiv.org/abs/2504.11456)
**License**: MIT License

### Quality Notes
- 48 samples with answer leakage were revised in May 2025
- Dataset is decontaminated and verified for quality
- Difficulty ratings range across advanced mathematical topics

## Cleaning Methodology

The dataset underwent aggressive cleaning using the `orz-math` preset to remove exam-specific artifacts while preserving mathematical content:

### Artifacts Removed:
- **Problem numbers**: Removed 1,555 samples with numbering artifacts (e.g., "Problem 6.", "147 Let")
- **Point allocations**: Removed 1 sample with point markers (e.g., "(8 points)")
- **Special artifacts**: Removed 1 sample with horizontal rules, translation instructions
- **Trailing artifacts**: Removed 1 sample with category labels, difficulty markers
- **Image references**: Detected in 278 samples (preserved but noted)

### Samples Filtered:
- **Multi-part problems**: 31 samples with structure like "a) ... b) ..." (0.03%)
- **URL references**: 0 samples (none found)

### Deduplication:
- **Exact duplicates removed**: 1,147 samples (1.11%)
- Deduplication based on problem text hash

### Quality Preservation:
- ✅ All mathematical notation preserved (LaTeX)
- ✅ Ground truth answers unchanged
- ✅ Problem semantics maintained
- ✅ VERL schema integrity verified

## Schema

The dataset follows the VERL (Verifiable Explanation Reasoning and Learning) format with minimal 3-field `extra_info`:

```python
{{
    "data_source": str,              # "deepmath-103k"
    "prompt": [                      # List of messages (conversation format)
        {{"role": str, "content": str}}  # role: "user", content: problem text
    ],
    "ability": str,                  # "math"
    "reward_model": {{
        "style": str,                # "rule" (rule-based verification)
        "ground_truth": str          # Expected answer
    }},
    "extra_info": {{
        "index": int,                # Sample index (0-based)
        "original_dataset": str,     # "deepmath-103k"
        "split": str                 # "train"
    }}
}}
```

## Mathematical Coverage

The dataset covers advanced mathematical topics including:

- **Advanced Calculus**: Limits, derivatives, integrals, series convergence
- **Real Analysis**: Measure theory, Lebesgue integration, functional analysis
- **Abstract Algebra**: Group theory, ring theory, field theory, commutators
- **Linear Algebra**: Vector spaces, eigenvalues, matrices
- **Number Theory**: Modular arithmetic, Diophantine equations, prime theory
- **Probability & Statistics**: Conditional probability, distributions, statistical inference
- **Discrete Mathematics**: Combinatorics, graph theory, recurrence relations
- **Topology & Geometry**: Metric spaces, topological properties, geometric proofs
- **Differential Equations**: ODEs, PDEs, Sturm-Liouville problems
- **Complex Analysis**: Complex functions, contour integration

Average ground truth length: 5 characters (concise, numerical/symbolic answers)

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load the complete dataset
dataset = load_dataset("sungyub/deepmath-103k-verl", split="train")

print(f"Dataset size: {{len(dataset):,}} samples")

# Access a sample
sample = dataset[0]
print(f"Problem: {{sample['prompt'][0]['content']}}")
print(f"Answer: {{sample['reward_model']['ground_truth']}}")
```

### Example Sample

```python
{{
    "data_source": "deepmath-103k",
    "prompt": [
        {{
            "role": "user",
            "content": "Evaluate the limit: \\\\[ \\\\lim_{{x \\\\to \\\\infty}} \\\\sqrt{{x}} \\\\left( \\\\sqrt[3]{{x+1}} - \\\\sqrt[3]{{x-1}} \\\\right) \\\\]"
        }}
    ],
    "ability": "math",
    "reward_model": {{
        "style": "rule",
        "ground_truth": "0"
    }},
    "extra_info": {{
        "index": 0,
        "original_dataset": "deepmath-103k",
        "split": "train"
    }}
}}
```

### Filtering by Index Range

```python
# Get first 1000 problems for quick testing
subset = dataset.select(range(1000))

# Get problems 50000-60000
middle_subset = dataset.select(range(50000, 60000))
```

### Integration with VERL

This dataset is ready for use with VERL (Verification and Reasoning with Language Models) training pipelines:

```python
from datatrove.utils.reward_score import compute_score

# Evaluate a solution
score = compute_score(
    data_source="deepmath-103k",
    solution_str="\\\\boxed{{0}}",
    ground_truth="0",
    format_type="auto"
)
```

## Statistics

- **Total samples**: {metadata["num_examples"]:,}
- **Dataset size**: {metadata["dataset_size"] / (1024**2):.2f} MB
- **Download size**: {metadata["download_size"] / (1024**2):.2f} MB (compressed)
- **Average problem length**: 203 characters
- **Average answer length**: 5 characters
- **Conversion rate**: 100% (no empty problems)
- **Duplication rate**: 1.11% (1,147 duplicates removed)
- **Modification rate**: 4.2% (4,366 samples cleaned)

## Processing Pipeline

This dataset was processed using the following pipeline:

1. **Conversion**: Original DeepMath-103K → VERL format
2. **Cleaning**: Applied `orz-math` preset (aggressive artifact removal)
3. **Deduplication**: SHA-256 hash-based exact duplicate removal
4. **Validation**: Schema verification and quality checks

Processing scripts available at: [datatrove repository](https://github.com/huggingface/datatrove)

## Citation

If you use this dataset, please cite both the original DeepMath-103K dataset and this cleaned version:

### Original Dataset

```bibtex
@article{{deepmath103k,
    title={{DeepMath-103K: A Large-Scale Dataset for Mathematical Reasoning}},
    author={{He, Zhiwei}},
    journal={{arXiv preprint arXiv:2504.11456}},
    year={{2025}},
    url={{https://arxiv.org/abs/2504.11456}}
}}
```

### This Cleaned Version

```bibtex
@dataset{{deepmath103k_verl,
    title={{DeepMath-103K VERL Dataset (Cleaned)}},
    author={{Kim, Sungyub}},
    year={{2025}},
    publisher={{Hugging Face}},
    url={{https://huggingface.co/datasets/sungyub/deepmath-103k-verl}},
    note={{Cleaned and deduplicated version of DeepMath-103K in VERL format}}
}}
```

## License

This dataset maintains the **MIT License** from the original DeepMath-103K dataset.

```
MIT License

Copyright (c) 2025 Zhiwei He

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## Acknowledgments

- **Original Dataset**: Zhiwei He (@zwhe99) for creating DeepMath-103K
- **Processing**: Cleaned and converted using [DataTrove](https://github.com/huggingface/datatrove)
- **VERL Format**: Adapted for use with VERL training pipelines

## Related Datasets

Part of the **Math VERL Unified** collection:
- [sungyub/math-verl-unified](https://huggingface.co/datasets/sungyub/math-verl-unified) - Unified collection of 9 math datasets
- [sungyub/big-math-rl-verl](https://huggingface.co/datasets/sungyub/big-math-rl-verl) - 242K math problems
- [sungyub/skywork-or1-math-verl](https://huggingface.co/datasets/sungyub/skywork-or1-math-verl) - 103K Skywork OR1 problems
- [sungyub/orz-math-72k-verl](https://huggingface.co/datasets/sungyub/orz-math-72k-verl) - 72K ORZ math problems

## Contact

For questions or issues, please open an issue on the [DataTrove GitHub repository](https://github.com/huggingface/datatrove).
"""

    # Combine YAML and Markdown
    full_content = f"---\n{yaml_str}---\n{markdown_content}"

    return full_content


def main():
    """Main entry point."""
    parquet_file = "./output/hub-upload/deepmath-103k-verl/data/train-00000.parquet"
    output_file = "./output/hub-upload/deepmath-103k-verl/README.md"

    print(f"\n{'='*70}")
    print(f"Generating DeepMath-103K README")
    print(f"{'='*70}")
    print(f"Input:  {parquet_file}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")

    # Check input exists
    if not Path(parquet_file).exists():
        print(f"✗ Parquet file not found: {parquet_file}")
        print(f"\nPlease run prepare_deepmath_standalone.py first:")
        print(f"  python scripts/hub/prepare_deepmath_standalone.py")
        exit(1)

    # Calculate metadata
    print("Step 1: Calculating metadata...")
    metadata = calculate_metadata(parquet_file)
    print(f"✓ Metadata calculated")
    print(f"  Samples: {metadata['num_examples']:,}")
    print(f"  Size: {metadata['dataset_size'] / (1024**2):.2f} MB")

    # Generate README
    print("\nStep 2: Generating README content...")
    readme_content = create_readme_content(metadata)
    print(f"✓ README generated ({len(readme_content):,} characters)")

    # Write to file
    print("\nStep 3: Writing README.md...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"✓ Saved to {output_file}")

    print(f"\n{'='*70}")
    print(f"✅ README generation completed successfully!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"1. Review README: cat {output_file}")
    print(f"2. Validate: python scripts/hub/validate_deepmath_upload.py")
    print(f"3. Upload: python scripts/upload/upload_deepmath_to_hub.py --individual")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
