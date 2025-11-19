#!/usr/bin/env python3
"""
Upload unified VERL dataset to HuggingFace Hub with named splits.

This script uploads the Phase 2 deduplicated dataset as a single repository
with named splits (one for each original dataset).
"""

import argparse
import glob
import os
import sys
from typing import Dict, List

from huggingface_hub import HfApi, create_repo


def create_readme_qa(stats: Dict[str, int], dedup_info: Dict) -> str:
    """Create README for QA unified dataset."""
    total_rows = sum(stats.values())

    # Generate split table
    split_rows = []
    for split_name in sorted(stats.keys()):
        count = stats[split_name]
        percentage = (count / total_rows) * 100
        # Convert underscores back to hyphens for display
        display_name = split_name.replace('_', '-')
        split_rows.append(f"| {display_name} | {count:,} | {percentage:.2f}% |")

    split_table = "\n".join(split_rows)

    # Generate config section
    config_entries = []
    for split_name in sorted(stats.keys()):
        display_name = split_name.replace('_', '-')
        config_entries.append(f"""  - split: {split_name}
    path: data/{display_name}.parquet""")

    config_section = "\n".join(config_entries)

    readme = f"""---
language:
- en
license: apache-2.0
size_categories:
- 10K<n<100K
task_categories:
- question-answering
- text-generation
- reinforcement-learning
pretty_name: Unified QA VERL Dataset
tags:
- qa
- reasoning
- reinforcement-learning
- verl
- deduplication
- table-reasoning
- logic-reasoning
- document-qa
- tool-use
configs:
- config_name: default
  data_files:
{config_section}
---

# Unified QA VERL Dataset

A unified collection of 5 high-quality question-answering and reasoning datasets in VERL format, deduplicated and optimized for reinforcement learning training.

## Dataset Summary

This dataset combines 5 diverse QA and reasoning datasets into a single unified collection:
- **Total Problems**: {total_rows:,} unique problems (after {dedup_info.get('reduction_rate', 0)*100:.2f}% deduplication)
- **Original Size**: {dedup_info.get('original_size', 0):,} problems (before deduplication)
- **Format**: VERL (Volcano Engine Reinforcement Learning)
- **Language**: English (with some Chinese in docqa-rl)
- **License**: Apache 2.0 (see attribution requirements below)

## Dataset Structure

### Splits

The dataset is organized into 5 named splits, one for each source dataset:

| Split Name | Problems | Percentage |
|------------|----------|------------|
{split_table}

### Usage

**Load specific split:**
```python
from datasets import load_dataset

# Load only one dataset
dataset = load_dataset("sungyub/qa-verl-unified", split="docqa-rl-verl")

# Load multiple datasets
dataset = load_dataset("sungyub/qa-verl-unified", split="guru-logic-verl+toolrl-4k-verl")

# Load all datasets
dataset = load_dataset("sungyub/qa-verl-unified")
```

**Streaming mode (recommended for large splits):**
```python
dataset = load_dataset("sungyub/qa-verl-unified", split="table-r1-zero-verl", streaming=True)
```

### Data Format

All splits follow the VERL (Volcano Engine Reinforcement Learning) format:

```python
{{
    "data_source": str,        # Dataset identifier
    "prompt": [                # Chat template format
        {{
            "role": "user",
            "content": "problem text"
        }}
    ],
    "ability": str,            # Task category (qa, logic, etc.)
    "reward_model": {{          # Verification info
        "style": str,
        "ground_truth": str
    }},
    "extra_info": {{            # Metadata
        "index": int,
        "split": str,
        "original_dataset": str  # Source dataset name
    }}
}}
```

## Deduplication Process

The dataset underwent a rigorous 2-phase deduplication process:

**Phase 1: Intra-dataset deduplication**
- Removed duplicates within each dataset
- Reduced {dedup_info.get('original_size', 0):,} → {dedup_info.get('phase1_output', 0):,} problems ({dedup_info.get('phase1_rate', 0)*100:.2f}% reduction)

**Phase 2: Inter-dataset deduplication**
- Removed duplicates across datasets using size-based priority
- Priority: smallest datasets first (preserves rare problems)
- Reduced {dedup_info.get('phase1_output', 0):,} → {total_rows:,} problems ({dedup_info.get('phase2_rate', 0)*100:.2f}% reduction)

**Overall**: {dedup_info.get('reduction_rate', 0)*100:.2f}% duplicate removal

### Deduplication Method
- SHA-256 hash-based exact matching
- Conservative text normalization (preserves formatting)
- VERL format validation for all outputs
- Size-based priority (smallest datasets preserved first)

## Source Datasets

### Logic Reasoning

**guru-logic-verl** ({stats.get('guru_logic_verl', 0):,} problems) - Apache 2.0
- Source: microsoft/MAmmoTH2-Plus
- Logic puzzles: ordering, zebra puzzles, graph problems, visual patterns
- 4 reasoning types with diverse difficulty levels

**toolrl-4k-verl** ({stats.get('toolrl_4k_verl', 0):,} problems) - CC-BY-4.0
- Tool-use samples in GPT OSS 120B format
- 10-15% improved token efficiency
- Train/test splits for evaluation

### Document & Table QA

**docqa-rl-verl** ({stats.get('docqa_rl_verl', 0):,} problems) - Apache 2.0
- Long-context document QA with multi-hop reasoning
- Complex information extraction tasks

**guru-table-verl** ({stats.get('guru_table_verl', 0):,} problems) - MIT
- Table reasoning from HiTab, MultiHierTT, FinQA
- Hierarchical tables and financial data analysis

**table-r1-zero-verl** ({stats.get('table_r1_zero_verl', 0):,} problems) - Apache 2.0
- Table reasoning problems from Table-R1-Zero-Dataset
- Diverse table structures and question types

## Dataset Characteristics

### Quality Metrics

**Estimated intra-dataset duplication rates** (before Phase 1):
- Expected: 5-15% (QA datasets typically more diverse than math)

**Inter-dataset preservation** (Phase 2, size-based priority):
- Smallest datasets fully preserved (highest priority)
- Larger datasets may have overlap with smaller curated sets

### Task Coverage

The dataset covers diverse QA and reasoning tasks:
- Logic puzzles and constraint satisfaction
- Table reasoning and numerical analysis
- Long-context document understanding
- Multi-hop reasoning
- Tool-use and function calling

## Use Cases

**Reinforcement Learning Training:**
- Post-training for QA and reasoning capabilities
- Multi-task RL with diverse problem types
- Reward modeling with ground truth

**Fine-tuning:**
- Improving multi-domain reasoning
- Table understanding and analysis
- Tool-use capabilities

**Evaluation:**
- Diverse task difficulty levels
- Multiple reasoning domains
- Verified ground truth answers

## Dataset Creation

### Deduplication Pipeline

1. **Phase 1 (Intra-dataset)**:
   - Process each dataset independently
   - Remove exact duplicates based on normalized problem text
   - Validate VERL format

2. **Phase 2 (Inter-dataset)**:
   - Process datasets in size-based priority order (smallest first)
   - Remove duplicates across datasets
   - Add `original_dataset` field for tracking

### Priority Rationale

Size-based priority (smallest datasets first) was chosen to:
- Preserve rare problems from small, curated datasets
- Maximize diversity of the final collection
- Retain unique contributions from each dataset

## Limitations

- Some problems from larger datasets may be removed as duplicates
- Original metadata standardized to `extra_info` schema
- Primarily English language (some Chinese in docqa-rl)
- Dataset balance varies by source

## License and Attribution

**Primary License**: Apache 2.0

**Attribution Requirements**:
- **toolrl-4k-verl**: CC-BY-4.0 - Requires attribution
- **guru-table-verl**: MIT - Requires copyright notice

When using this dataset, please:
1. Include attribution to original dataset creators
2. Comply with Apache 2.0, CC-BY-4.0, and MIT license terms
3. See individual dataset licenses for specific requirements

**Source Licenses**:
- docqa-rl-verl: Apache 2.0
- guru-logic-verl: Apache 2.0
- toolrl-4k-verl: CC-BY-4.0 (⚠️ Requires attribution)
- guru-table-verl: MIT (⚠️ Requires copyright notice)
- table-r1-zero-verl: Apache 2.0

## Citation

If you use this dataset, please cite the original source datasets and this unified collection:

```bibtex
@dataset{{qa-verl-unified,
  title={{Unified QA VERL Dataset}},
  author={{Sungyub Kim}},
  year={{2025}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/datasets/sungyub/qa-verl-unified}}
}}
```

## Dataset Card Authors

- Sungyub Kim (sungyub)
"""
    return readme


def create_readme_if(stats: Dict[str, int], dedup_info: Dict) -> str:
    """Create README for IF unified dataset."""
    total_rows = sum(stats.values())

    # Generate split table
    split_rows = []
    for split_name in sorted(stats.keys()):
        count = stats[split_name]
        percentage = (count / total_rows) * 100
        display_name = split_name.replace('_', '-')
        split_rows.append(f"| {display_name} | {count:,} | {percentage:.2f}% |")

    split_table = "\n".join(split_rows)

    # Generate config section
    config_entries = []
    for split_name in sorted(stats.keys()):
        display_name = split_name.replace('_', '-')
        config_entries.append(f"""  - split: {split_name}
    path: data/{display_name}.parquet""")

    config_section = "\n".join(config_entries)

    readme = f"""---
language:
- en
license: odc-by
size_categories:
- 100K<n<1M
task_categories:
- text-generation
- reinforcement-learning
pretty_name: Unified IF VERL Dataset
tags:
- instruction-following
- evaluation
- ifeval
- reasoning
- reinforcement-learning
- verl
- deduplication
configs:
- config_name: default
  data_files:
{config_section}
---

# Unified IF VERL Dataset

A unified collection of 2 high-quality instruction-following (IF) evaluation datasets in VERL format, deduplicated and optimized for reinforcement learning training.

## Dataset Summary

This dataset combines 2 instruction-following evaluation datasets into a single unified collection:
- **Total Problems**: {total_rows:,} unique problems (after {dedup_info.get('reduction_rate', 0)*100:.2f}% deduplication)
- **Original Size**: {dedup_info.get('original_size', 0):,} problems (before deduplication)
- **Constraint Types**: 79 unique constraint types (25 + 54 from sources)
- **Format**: VERL (Volcano Engine Reinforcement Learning)
- **Language**: English
- **License**: ODC-BY (Open Data Commons Attribution License)

## Dataset Structure

### Splits

The dataset is organized into 2 named splits, one for each source dataset:

| Split Name | Problems | Percentage |
|------------|----------|------------|
{split_table}

### Usage

**Load specific split:**
```python
from datasets import load_dataset

# Load only one dataset
dataset = load_dataset("sungyub/if-verl-unified", split="ifeval-rlvr-verl")

# Load all datasets
dataset = load_dataset("sungyub/if-verl-unified")
```

**Using with IFEval Scorer:**
```python
from datatrove.utils.reward_score import compute_score

# Get an example
example = dataset[0]

# Generate a response
response = "<think>Analysis here</think>\\nFinal answer"

# Compute score
score = compute_score(
    data_source="sungyub/if-verl-unified",
    solution_str=response,
    ground_truth=example["reward_model"]["ground_truth"],
    format_type="auto"  # Supports both XML and GPT OSS formats
)
```

### Data Format

All splits follow the VERL (Volcano Engine Reinforcement Learning) format:

```python
{{
    "data_source": str,        # Dataset identifier
    "prompt": [                # Chat template format
        {{
            "role": "user",
            "content": "instruction with constraints"
        }}
    ],
    "ability": "instruction_following",
    "reward_model": {{          # Verification info
        "style": "ifeval",
        "ground_truth": str     # Python literal string with constraint specs
    }},
    "extra_info": {{            # Metadata
        "index": int,
        "split": str,
        "original_dataset": str  # Source dataset name
    }},
    "dataset": "ifeval"
}}
```

## Deduplication Process

The dataset underwent a rigorous 2-phase deduplication process:

**Phase 1: Intra-dataset deduplication**
- Removed duplicates within each dataset
- Reduced {dedup_info.get('original_size', 0):,} → {dedup_info.get('phase1_output', 0):,} problems ({dedup_info.get('phase1_rate', 0)*100:.2f}% reduction)

**Phase 2: Inter-dataset deduplication**
- Removed duplicates across datasets using size-based priority
- Priority: smallest datasets first (preserves rare problems)
- Reduced {dedup_info.get('phase1_output', 0):,} → {total_rows:,} problems ({dedup_info.get('phase2_rate', 0)*100:.2f}% reduction)

**Overall**: {dedup_info.get('reduction_rate', 0)*100:.2f}% duplicate removal

### Deduplication Method
- SHA-256 hash-based exact matching
- Conservative text normalization (preserves formatting)
- VERL format validation for all outputs
- Size-based priority (smallest datasets preserved first)

## Source Datasets

**ifeval-rlvr-verl** ({stats.get('ifeval_rlvr_verl', 0):,} problems) - ODC-BY
- Source: allenai/RLVR-IFeval
- 25 distinct constraint types
- Instruction-following evaluation from Allen Institute

**ifbench-verl** ({stats.get('ifbench_verl', 0):,} problems) - ODC-BY
- 54 distinct constraint types
- Comprehensive instruction-following benchmark
- Multi-source aggregation (95K examples)

## Constraint Types Coverage

The unified dataset includes 79 total constraint types across 9 categories:
- **Keywords** (4 types): existence, frequency, forbidden words, letter frequency
- **Language** (1 type): response language requirements
- **Length Constraints** (4 types): paragraphs, words, sentences, nth paragraph
- **Detectable Content** (2 types): postscript, placeholders
- **Detectable Format** (6 types): bullet lists, title, constrained response, highlighted sections, sections, JSON
- **Combination** (2 types): repeat prompt, two responses
- **Case Changes** (3 types): uppercase, lowercase, capital word frequency
- **Start/End** (2 types): end checker, quotation
- **Punctuation** (1 type): no comma

## Use Cases

**Reinforcement Learning Training:**
- Post-training for instruction-following capabilities
- Constraint satisfaction learning
- Reward modeling with verifiable constraints

**Evaluation:**
- Systematic instruction-following evaluation
- Constraint compliance testing
- Multi-constraint scenarios

**Fine-tuning:**
- Improving instruction adherence
- Constraint-aware generation
- Format compliance training

## Dataset Creation

### Deduplication Pipeline

1. **Phase 1 (Intra-dataset)**:
   - Process each dataset independently
   - Remove exact duplicates based on normalized instruction text
   - Validate VERL format

2. **Phase 2 (Inter-dataset)**:
   - Process datasets in size-based priority order (smallest first)
   - Remove duplicates across datasets
   - Add `original_dataset` field for tracking

### Priority Rationale

Size-based priority (smallest datasets first) was chosen to:
- Preserve rare constraint types from smaller datasets
- Maximize constraint type diversity
- Retain unique instruction patterns

## Limitations

- Some high-frequency constraints may be overrepresented
- English language only
- Constraint specifications in Python literal format
- Some overlap expected between IF evaluation datasets

## License and Attribution

**License**: ODC-BY (Open Data Commons Attribution License)

Under ODC-BY, you are free to:
- **Share**: Copy and redistribute the data
- **Adapt**: Transform and build upon the data
- **Use commercially**: Use the data for commercial purposes

**Requirements**:
- **Attribution**: You must give appropriate credit to original data sources
- Indicate if changes were made

**Source Attributions**:
- ifeval-rlvr-verl: allenai/RLVR-IFeval (ODC-BY)
- ifbench-verl: Multiple sources, aggregated (ODC-BY)

## Citation

If you use this dataset, please cite the original source datasets and this unified collection:

```bibtex
@dataset{{if-verl-unified,
  title={{Unified IF VERL Dataset}},
  author={{Sungyub Kim}},
  year={{2025}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/datasets/sungyub/if-verl-unified}}
}}

@misc{{rlvr-ifeval,
  title={{RLVR-IFeval: Instruction Following Evaluation Dataset}},
  author={{Allen Institute for AI}},
  year={{2024}},
  url={{https://huggingface.co/datasets/allenai/RLVR-IFeval}}
}}

@misc{{zhou2023instructionfollowing,
  title={{Instruction-Following Evaluation for Large Language Models}},
  author={{Jeffrey Zhou and Tianjian Lu and Swaroop Mishra and Siddhartha Brahma and Sujoy Basu and Yi Luan and Denny Zhou and Le Hou}},
  year={{2023}},
  eprint={{2311.07911}},
  archivePrefix={{arXiv}}
}}
```

## Dataset Card Authors

- Sungyub Kim (sungyub)
"""
    return readme


def upload_unified_dataset(
    splits_dir: str,
    repo_name: str,
    collection_type: str,
    dedup_stats_file: str = None,
    private: bool = False,
    verbose: bool = False
) -> None:
    """
    Upload unified dataset to HuggingFace Hub with named splits.

    Args:
        splits_dir: Directory containing split parquet files
        repo_name: Repository name (e.g., "sungyub/qa-verl-unified")
        collection_type: Type of collection ("qa" or "if")
        dedup_stats_file: Path to deduplication statistics JSON (optional)
        private: Whether to make the repository private
        verbose: Print detailed progress
    """
    print(f"\n{'=' * 70}")
    print(f"Uploading Unified {collection_type.upper()} VERL Dataset")
    print(f"{'=' * 70}\n")

    # Find all split files
    split_files = sorted(glob.glob(os.path.join(splits_dir, '*.parquet')))
    if not split_files:
        raise ValueError(f"No parquet files found in {splits_dir}")

    print(f"Repository: {repo_name}")
    print(f"Split files: {len(split_files)}")
    print()

    # Calculate statistics for README
    import pyarrow.parquet as pq
    stats = {}
    for file_path in split_files:
        split_name = os.path.basename(file_path).replace('.parquet', '').replace('-', '_')
        # Read only metadata to avoid chunked array issues
        parquet_file = pq.ParquetFile(file_path)
        num_rows = parquet_file.metadata.num_rows
        stats[split_name] = num_rows
        if verbose:
            print(f"  {split_name}: {num_rows:,} rows")

    print()

    # Load dedup info if available
    dedup_info = {}
    if dedup_stats_file and os.path.exists(dedup_stats_file):
        import json
        with open(dedup_stats_file, 'r') as f:
            dedup_data = json.load(f)
            # Calculate dedup info from phase stats
            original_size = dedup_data.get('total_input_rows', 0)
            phase1_output = dedup_data.get('total_output_rows', 0)
            final_size = sum(stats.values())

            dedup_info = {
                'original_size': original_size,
                'phase1_output': phase1_output,
                'phase1_rate': (original_size - phase1_output) / original_size if original_size > 0 else 0,
                'phase2_rate': (phase1_output - final_size) / phase1_output if phase1_output > 0 else 0,
                'reduction_rate': (original_size - final_size) / original_size if original_size > 0 else 0
            }

    # Create repository
    print(f"Creating repository: {repo_name}")
    api = HfApi()
    try:
        create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print("✅ Repository created/exists")
    except Exception as e:
        print(f"⚠️  Warning: {e}")

    print()

    # Create README
    print("Generating README.md...")
    if collection_type == "qa":
        readme_content = create_readme_qa(stats, dedup_info)
    elif collection_type == "if":
        readme_content = create_readme_if(stats, dedup_info)
    else:
        raise ValueError(f"Unknown collection type: {collection_type}")

    readme_path = "/tmp/README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("✅ README.md generated")
    print()

    # Upload README
    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
    )
    print("✅ README.md uploaded")
    print()

    # Upload split files
    print("Uploading split files...")
    for file_path in split_files:
        split_name = os.path.basename(file_path)
        print(f"  Uploading {split_name}...")

        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"data/{split_name}",
            repo_id=repo_name,
            repo_type="dataset",
        )

        print(f"  ✅ {split_name} uploaded")

    print()
    print(f"{'=' * 70}")
    print("Upload Complete!")
    print(f"{'=' * 70}")
    print(f"Repository: https://huggingface.co/datasets/{repo_name}")
    print(f"Total splits: {len(split_files)}")
    print(f"Total problems: {sum(stats.values()):,}")
    print(f"{'=' * 70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Upload unified VERL dataset to HuggingFace Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--splits-dir',
        type=str,
        required=True,
        help='Directory containing split parquet files'
    )

    parser.add_argument(
        '--repo-name',
        type=str,
        required=True,
        help='HuggingFace repository name (e.g., sungyub/qa-verl-unified)'
    )

    parser.add_argument(
        '--collection-type',
        type=str,
        required=True,
        choices=['qa', 'if'],
        help='Type of collection (qa or if)'
    )

    parser.add_argument(
        '--dedup-stats',
        type=str,
        help='Path to Phase 1 deduplication statistics JSON file'
    )

    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the repository private'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )

    args = parser.parse_args()

    try:
        upload_unified_dataset(
            splits_dir=args.splits_dir,
            repo_name=args.repo_name,
            collection_type=args.collection_type,
            dedup_stats_file=args.dedup_stats,
            private=args.private,
            verbose=args.verbose
        )

        print("✅ Unified dataset uploaded successfully!\n")

    except Exception as e:
        print(f"\n❌ Error uploading dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
