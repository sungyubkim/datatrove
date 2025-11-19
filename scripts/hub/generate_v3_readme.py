#!/usr/bin/env python3
"""
Generate README.md for v3.0 (Inter-Dataset Deduplication).
"""

import json
from pathlib import Path


def generate_readme():
    # Load dedup stats
    stats_file = Path('output/deduplicated-inter/stats/phase2_inter_splits_stats.json')
    with open(stats_file, 'r') as f:
        dedup_stats = json.load(f)

    # Extract stats
    by_split = dedup_stats['by_split']
    total_samples = dedup_stats['total_output_rows']
    removed_dupes = dedup_stats['total_duplicates_removed']

    # Calculate percentages
    def pct(rows):
        return (rows / total_samples) * 100

    readme = f"""---
language: en
license: apache-2.0
size_categories:
  - 1M<n<10M
task_categories:
  - question-answering
  - reinforcement-learning
  - text-generation
pretty_name: Unified Math VERL Dataset
tags:
  - deduplication
  - math
  - reasoning
  - reinforcement-learning
  - rlhf
  - verl
  - mathematical-reasoning
  - dataset-quality
configs:
  - config_name: default
    data_files:
      - split: mathx_5m_verl
        path: data/mathx-5m-verl.parquet
      - split: eurus_2_math_verl
        path: data/eurus-2-math-verl.parquet
      - split: big_math_rl_verl
        path: data/big-math-rl-verl.parquet
      - split: openr1_math_verl
        path: data/openr1-math-verl.parquet
      - split: orz_math_72k_verl
        path: data/orz-math-72k-verl.parquet
      - split: deepscaler_preview_verl
        path: data/deepscaler-preview-verl.parquet
      - split: skywork_or1_math_verl
        path: data/skywork_or1_math_verl.parquet
      - split: dapo_math_17k_verl
        path: data/dapo-math-17k-verl.parquet
      - split: deepmath_103k_verl
        path: data/deepmath_103k_verl.parquet
---

# Unified Math VERL Dataset

## Dataset Summary

**Unified Math VERL** is a comprehensive mathematical reasoning dataset that consolidates **{total_samples:,} high-quality problems** from 9 major math datasets, all formatted for the Volcano Engine Reinforcement Learning (VERL) framework.

This unified collection represents a **massive deduplication effort** with both intra-dataset and inter-dataset deduplication, ensuring maximum quality and minimal redundancy.

**Key Features:**
- 🎯 **2.27M unique problems**: Largest deduplicated math VERL collection
- ✅ **9 curated datasets**: Best math datasets in unified format
- 🔧 **Advanced cleaning**: Problem-specific cleaning methodologies
- 🔥 **Inter-dataset dedup**: Cross-dataset duplicate removal (v3.0)
- 📊 **Rich metadata**: Problem types, sources, difficulty levels
- 🚀 **VERL-ready**: Optimized for reinforcement learning training

**Perfect for:**
- Large-scale mathematical reasoning model training
- Diverse multi-domain math capability development
- Reinforcement learning from verification/reasoning
- Comprehensive math evaluation benchmarks

---

## Recent Updates

### 2025-11-09: v3.0 Inter-Dataset Deduplication 🔥

**Major quality improvement through cross-dataset deduplication:**

- **Total Samples**: 2,464,591 → **{total_samples:,}** (-{removed_dupes:,}, -12.7%)
- **Deduplication**: Added inter-dataset duplicate removal
- **Priority-Based**: Smaller datasets preserved over larger ones
- **Processing**: 2.2 minutes for 2.6M comparisons (~19.5K rows/sec)

**Impact by Dataset:**
- **skywork_or1_math**: 102,669 → 39,202 (-61.8% duplicates)
  - 34,718 duplicates with deepscaler-preview
  - 25,702 duplicates with orz-math-72k
- **openr1-math**: 184,439 → 120,387 (-34.7% duplicates)
  - 41,909 duplicates with orz-math-72k
  - 18,770 duplicates with skywork
- **eurus-2-math**: 411,863 → 283,612 (-31.1% duplicates)
  - 59,148 duplicates with big-math-rl
  - 40,381 duplicates with openr1-math
- **big-math-rl**: 242,092 → 196,329 (-18.9% duplicates)
- **deepmath_103k**: 101,844 → 95,496 (-6.2% duplicates)
- **deepscaler-preview**: 38,005 → 35,789 (-5.8% duplicates)
- **orz-math-72k**: 46,426 → 44,812 (-3.5% duplicates)
- **mathx-5m**: 1,453,400 → 1,436,392 (-1.2% duplicates)
- **dapo-math-17k**: 17,186 → 17,147 (-0.2% duplicates) ✅ highest preservation

**Priority Order** (highest to lowest preservation):
1. dapo-math-17k-verl (smallest, 99.8% kept)
2. deepscaler-preview-verl
3. orz-math-72k-verl
4. deepmath_103k_verl
5. skywork_or1_math_verl
6. openr1-math-verl
7. big-math-rl-verl
8. eurus-2-math-verl
9. mathx-5m-verl (largest, 98.8% kept)

**Key Findings:**
- Cross-dataset overlap was significant: 12.7% of total samples were duplicates
- Smaller curated datasets (dapo, deepscaler, orz) had high uniqueness
- Larger synthetic datasets had more overlap with others
- Priority-based approach preserved high-quality curated samples

**Breaking Change Notice:**
- ⚠️ Users will receive ~13% fewer samples
- Significantly affects skywork (-62%), openr1 (-35%), eurus (-31%)
- Quality improvement: eliminates cross-dataset redundancy
- For v2.2 (without inter-dedup), use commit `cbab00d`

---

### 2025-11-09: Eurus-2-Math v2.0 Advanced Cleaning ✨

**Major update to `eurus_2_math_verl` split:**

- **Sample Count**: 317,399 → **411,863** (+94,464, +29.8% increase)
- **Quality**: v2.0 with maximum cleaning (orz-math preset)
- **Deduplication**: Additional 17 duplicates removed
- **Cleaning Applied** (9.95% of samples enhanced):
  - Problem number removal: 23,806 samples (5.78%)
  - Point allocation removal: 399 samples (0.10%)
  - Contest metadata removal: 26 samples (0.01%)
  - Markdown header removal: 5 samples (<0.01%)
  - Trailing artifact removal: 109 samples (0.03%)
  - Special artifact removal: 26 samples (0.01%)
- **Quality Filtering** (4.53% filtered out):
  - URL samples filtered: 452 samples (0.11%)
  - Multi-part samples filtered: 18,211 samples (4.42%)
  - Total reduction: 18,663 samples for quality improvement
- **Image Reference Detection**: 449 samples (0.11%) flagged but retained
- **Processing Method**: DataTrove MathDatasetCleaner with orz-math preset

For detailed information about the Eurus-2-Math enhancement, see [sungyub/eurus-2-math-verl](https://huggingface.co/datasets/sungyub/eurus-2-math-verl).

---

### 2025-11-08: DeepScaleR v2.0 Enhanced Cleaning 🚀

**Major update to `deepscaler_preview_verl` split:**

- **Sample Count**: 36,877 → **38,005** (+3.1% increase)
- **Quality**: v2.0 with enhanced multiline-aware multi-part filtering
- **Deduplication**: Additional quality-based filtering beyond hash matching
- **Cleaning Applied** (14.6% of samples enhanced):
  - Problem number removal: 2,036 samples
  - Artifact removal: LaTeX formatting artifacts, contest metadata
  - Quality filtering: 1,090 problematic samples removed
- **False Positives Eliminated**: 160 samples (13.1% improvement in multi-part filtering)
  - Multiline-aware pattern detection preserves legitimate references
  - "equations $(1)$ and $(2)$" no longer incorrectly filtered
  - Enhanced precision while maintaining recall

---

## Dataset Statistics

### Overview

| Metric | Value |
|--------|-------|
| **Total Problems** | {total_samples:,} |
| **Number of Datasets** | 9 |
| **Intra-Dedup Rate** | 91.17% (from 27.9M original) |
| **Inter-Dedup Rate** | 12.7% (from 2.46M → 2.27M) |
| **Total Size** | ~1.78 GB |
| **Format** | VERL (Volcano Engine RL) |
| **License** | Apache-2.0 |

### Split Statistics

| Split | Rows | Percentage | Size | Status |
|-------|------|------------|------|--------|
| **mathx_5m_verl** | {by_split['mathx-5m-verl']['kept_rows']:,} | {pct(by_split['mathx-5m-verl']['kept_rows']):.1f}% | ~1.6 GB | ✓ |
| **eurus_2_math_verl** | {by_split['eurus-2-math-verl']['kept_rows']:,} | {pct(by_split['eurus-2-math-verl']['kept_rows']):.1f}% | ~37 MB | ✨ v2.0 |
| **big_math_rl_verl** | {by_split['big-math-rl-verl']['kept_rows']:,} | {pct(by_split['big-math-rl-verl']['kept_rows']):.1f}% | ~24 MB | ✓ |
| **openr1_math_verl** | {by_split['openr1-math-verl']['kept_rows']:,} | {pct(by_split['openr1-math-verl']['kept_rows']):.1f}% | ~18 MB | ✓ |
| **deepmath_103k_verl** | {by_split['deepmath_103k_verl']['kept_rows']:,} | {pct(by_split['deepmath_103k_verl']['kept_rows']):.1f}% | ~11 MB | ✓ |
| **orz_math_72k_verl** | {by_split['orz-math-72k-verl']['kept_rows']:,} | {pct(by_split['orz-math-72k-verl']['kept_rows']):.1f}% | ~7 MB | ✓ |
| **skywork_or1_math_verl** | {by_split['skywork_or1_math_verl']['kept_rows']:,} | {pct(by_split['skywork_or1_math_verl']['kept_rows']):.1f}% | ~7 MB | ✓ |
| **deepscaler_preview_verl** | {by_split['deepscaler-preview-verl']['kept_rows']:,} | {pct(by_split['deepscaler-preview-verl']['kept_rows']):.1f}% | ~5 MB | ✨ v2.0 |
| **dapo_math_17k_verl** | {by_split['dapo-math-17k-verl']['kept_rows']:,} | {pct(by_split['dapo-math-17k-verl']['kept_rows']):.1f}% | ~2.5 MB | ✓ |
| **Total** | **{total_samples:,}** | **100%** | **~1.78 GB** | |

---

## Quality and Deduplication

### Deduplication Methodology

This dataset employs a **two-stage deduplication** process:

#### Stage 1: Intra-Dataset Deduplication
- **Hash-Based Dedup**: SHA-256 with text normalization
- **Within-Dataset**: Duplicates removed within each source
- **First Occurrence Preserved**: Original ordering maintained
- **Metadata Preserved**: All enriched fields retained
- **Result**: ~27.9M → 2.46M samples (91.2% reduction)

#### Stage 2: Inter-Dataset Deduplication (v3.0)
- **Priority-Based**: Smaller datasets processed first
- **Cross-Dataset**: Duplicates removed across different sources
- **Hash Matching**: Same normalization as Stage 1
- **Preservation Strategy**: Higher-priority datasets keep their samples
- **Result**: 2.46M → 2.27M samples (12.7% reduction)
- **Processing**: 2.2 minutes for 2.6M row comparisons

**Combined Deduplication:**
- **Original**: ~27.9M samples across all sources
- **After Stage 1**: 2.46M samples (91.2% reduction)
- **After Stage 2**: 2.27M samples (12.7% additional reduction)
- **Total Reduction**: 91.9% from original

### Per-Dataset Quality

| Dataset | Quality Score | Notes |
|---------|---------------|-------|
| skywork_or1_math_verl | **99.0%** | Enhanced with orz-math preset |
| openr1_math_verl | **98.7%** | v3.0 maximum cleaning |
| orz_math_72k_verl | ~98% | Manual + automated |
| dapo_math_17k_verl | ~97% | High baseline quality |
| eurus_2_math_verl | ~95% | Competition-level |
| mathx_5m_verl | ~94% | Large-scale automated |
| deepscaler_preview_verl | ~93% | Preview quality |
| big_math_rl_verl | ~92% | Highly deduplicated |

---

## Usage

### Loading the Full Dataset

```python
from datasets import load_dataset

# Load all splits
dataset = load_dataset("sungyub/math-verl-unified")

# Check available splits
print(dataset.keys())

# Get total size
total = sum(len(dataset[split]) for split in dataset.keys())
print(f"Total samples: {{total:,}}")
# Output: Total samples: {total_samples:,}
```

### Loading Specific Splits

```python
# Load single split
openr1_data = load_dataset(
    "sungyub/math-verl-unified",
    split="openr1_math_verl"
)

print(f"OpenR1 samples: {{len(openr1_data):,}}")
# Output: OpenR1 samples: {by_split['openr1-math-verl']['kept_rows']:,}
```

---

## Schema

All splits share a unified VERL schema:

```python
{{
    "data_source": "dataset-name",  # Source dataset identifier
    "prompt": [  # Chat-format problem
        {{
            "role": "user",
            "content": "Problem text with LaTeX..."
        }}
    ],
    "ability": "math",  # Task type
    "reward_model": {{
        "style": "rule",  # Verification method
        "ground_truth": "answer"  # Expected solution
    }},
    "extra_info": {{
        "index": 0  # Original index
    }}
}}
```

---

## Limitations and Considerations

### Dataset-Specific Limitations

1. **Cross-Dataset Overlap Removed (v3.0)**
   - Inter-dataset deduplication removes 12.7% of samples
   - Smaller datasets prioritized and preserved
   - Breaking change for existing users

2. **Size Imbalance**
   - mathx_5m still dominates (63.3% of total)
   - Smaller splits may be underrepresented in random sampling
   - Use stratified sampling for balanced training

3. **Schema Consistency**
   - All splits use minimal single-field extra_info structure (index only)
   - Ensures Hub preview compatibility across all splits

---

## Citation

If you use this unified dataset, please cite:

```bibtex
@misc{{math-verl-unified,
  title={{Unified Math VERL Dataset: A Comprehensive Collection of Mathematical Reasoning Problems}},
  author={{Sung Yub Kim}},
  year={{2025}},
  url={{https://huggingface.co/datasets/sungyub/math-verl-unified}},
  note={{Unified collection of 9 major math datasets with intra- and inter-dataset deduplication}}
}}
```

---

## License

This unified dataset is released under the **Apache-2.0 License**.

**Individual Dataset Licenses:**
- All constituent datasets are Apache-2.0 or compatible licenses
- Check individual dataset pages for specific licensing details
- Combined dataset maintains Apache-2.0 compatibility

---

**Last Updated**: 2025-11-09
**Version**: 3.0 (Inter-Dataset Deduplication)
**Total Samples**: {total_samples:,}
**Format**: VERL v1.0
"""

    return readme


def main():
    readme = generate_readme()

    # Save README
    output_file = Path('output/README_v3.md')
    with open(output_file, 'w') as f:
        f.write(readme)

    print(f"✅ README generated: {output_file}")
    print(f"\nPreview:")
    print("=" * 70)
    print(readme[:1000])
    print("...")


if __name__ == '__main__':
    main()
