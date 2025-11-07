# VERL Dataset Deduplication Scripts

This directory contains scripts for downloading, deduplicating, and uploading unified VERL datasets to HuggingFace Hub.

## Overview

The deduplication pipeline consists of 3 main phases:

1. **Download**: Download individual datasets from HuggingFace Hub
2. **Phase 1 (Intra-dataset)**: Remove duplicates within each dataset
3. **Phase 2 (Inter-dataset)**: Remove duplicates across datasets using priority order
4. **Splits & Upload**: Create named splits and upload to HuggingFace Hub

## Prerequisites

```bash
pip install datasets huggingface_hub pandas pyarrow tqdm
```

## Quick Start

### QA Collection

```bash
# 1. Download QA datasets
python download_datasets.py --collection qa --output-dir qa_data

# 2. Phase 1: Intra-dataset deduplication
cd qa_data
python ../deduplicate_intra.py --all --output-dir ../_deduplicated/qa/phase1

# 3. Phase 2: Inter-dataset deduplication
python ../deduplicate_inter.py \
    --priority-file ../qa_priority.txt \
    --input-dir ../_deduplicated/qa/phase1 \
    --output-dir ../_deduplicated/qa/phase2

# 4. Create splits for HuggingFace
python ../create_splits.py \
    --combined-dir ../_deduplicated/qa/phase2/combined/data \
    --output-dir ../_deduplicated/qa/phase2/splits

# 5. Upload to HuggingFace Hub
python ../upload_unified.py \
    --splits-dir ../_deduplicated/qa/phase2/splits \
    --repo-name sungyub/qa-verl-unified \
    --collection-type qa \
    --dedup-stats ../_deduplicated/qa/phase1/phase1_summary.json
```

### IF Collection

```bash
# 1. Download IF datasets
python download_datasets.py --collection if --output-dir if_data

# 2. Phase 1: Intra-dataset deduplication
cd if_data
python ../deduplicate_intra.py --all --output-dir ../_deduplicated/if/phase1

# 3. Phase 2: Inter-dataset deduplication
python ../deduplicate_inter.py \
    --priority-file ../if_priority.txt \
    --input-dir ../_deduplicated/if/phase1 \
    --output-dir ../_deduplicated/if/phase2

# 4. Create splits for HuggingFace
python ../create_splits.py \
    --combined-dir ../_deduplicated/if/phase2/combined/data \
    --output-dir ../_deduplicated/if/phase2/splits

# 5. Upload to HuggingFace Hub
python ../upload_unified.py \
    --splits-dir ../_deduplicated/if/phase2/splits \
    --repo-name sungyub/if-verl-unified \
    --collection-type if \
    --dedup-stats ../_deduplicated/if/phase1/phase1_summary.json
```

## Scripts Reference

### `download_datasets.py`

Download datasets from HuggingFace Hub.

```bash
# Download QA collection
python download_datasets.py --collection qa --output-dir qa_data

# Download IF collection
python download_datasets.py --collection if --output-dir if_data

# Download specific dataset
python download_datasets.py --dataset sungyub/docqa-rl-verl --output-dir qa_data
```

**Output Structure:**
```
qa_data/
├── docqa-rl-verl/
│   └── data/
│       └── train-00000.parquet
├── guru-logic-verl/
│   └── data/
│       └── train-00000.parquet
...
```

### `deduplicate_intra.py`

Phase 1: Remove duplicates within each dataset.

```bash
# Process all datasets in directory
python deduplicate_intra.py --all --output-dir _deduplicated/phase1

# Process single dataset
python deduplicate_intra.py \
    --dataset docqa-rl-verl \
    --output-dir _deduplicated/phase1

# Dry run (count only, no output)
python deduplicate_intra.py --all --dry-run

# Verbose mode
python deduplicate_intra.py --all --verbose
```

**Features:**
- SHA-256 hash-based deduplication
- Streaming processing (memory efficient)
- Batch processing (10K rows per batch)
- Automatic file splitting (500K rows per file)
- Statistics tracking and reporting

### `deduplicate_inter.py`

Phase 2: Remove duplicates across datasets using priority order.

```bash
# Run with priority file
python deduplicate_inter.py \
    --priority-file qa_priority.txt \
    --input-dir _deduplicated/phase1 \
    --output-dir _deduplicated/phase2

# Use current directory datasets (skip Phase 1)
python deduplicate_inter.py \
    --priority-file qa_priority.txt \
    --use-current-datasets \
    --output-dir _deduplicated/phase2
```

**Priority Files:**
- `qa_priority.txt`: QA datasets (smallest to largest)
- `if_priority.txt`: IF datasets (smallest to largest)

**Features:**
- Priority-based deduplication (preserves higher-priority datasets)
- Tracks cross-dataset duplicate sources
- Adds `original_dataset` field to `extra_info`
- Creates combined dataset with all unique examples

### `create_splits.py`

Create named splits from combined dataset for HuggingFace Hub.

```bash
python create_splits.py \
    --combined-dir _deduplicated/phase2/combined/data \
    --output-dir _deduplicated/phase2/splits \
    --verbose
```

**Output:**
```
splits/
├── docqa-rl-verl.parquet
├── guru-logic-verl.parquet
├── toolrl-4k-verl.parquet
├── guru-table-verl.parquet
└── table-r1-zero-verl.parquet
```

### `generate_report.py`

Generate deduplication reports.

```bash
python generate_report.py \
    --phase1-stats _deduplicated/phase1/phase1_summary.json \
    --phase2-stats _deduplicated/phase2/stats/phase2_stats.json \
    --output-dir _deduplicated/reports \
    --format both
```

**Outputs:**
- `deduplication_report.md`: Markdown report with tables and statistics
- `deduplication_summary.json`: JSON summary with all metrics

### `upload_unified.py`

Upload unified dataset to HuggingFace Hub with comprehensive README.

```bash
# Upload QA unified dataset
python upload_unified.py \
    --splits-dir _deduplicated/qa/phase2/splits \
    --repo-name sungyub/qa-verl-unified \
    --collection-type qa \
    --dedup-stats _deduplicated/qa/phase1/phase1_summary.json

# Upload IF unified dataset
python upload_unified.py \
    --splits-dir _deduplicated/if/phase2/splits \
    --repo-name sungyub/if-verl-unified \
    --collection-type if \
    --dedup-stats _deduplicated/if/phase1/phase1_summary.json

# Upload as private repository
python upload_unified.py \
    --splits-dir _deduplicated/qa/phase2/splits \
    --repo-name sungyub/qa-verl-unified \
    --collection-type qa \
    --private
```

**Features:**
- Automatic README generation (collection-specific)
- YAML frontmatter with dataset card metadata
- Named splits support (load individual datasets)
- Deduplication statistics integration
- License information and attribution

### `utils.py`

Core utilities (imported by other scripts):
- `compute_hash()`: SHA-256 hashing of normalized text
- `normalize_text()`: Conservative text normalization
- `extract_problem_text()`: Extract prompt content from VERL format
- `validate_verl_row()`: VERL schema validation
- `DuplicationStats`: Statistics tracking class

## Dataset Priority Orders

### QA Collection (Size-based: Smallest First)

```
1. docqa-rl-verl      (1,591 examples)
2. guru-logic-verl    (1,742 examples)
3. toolrl-4k-verl     (4,000 examples)
4. guru-table-verl    (8,230 examples)
5. table-r1-zero-verl (69,265 examples)
```

### IF Collection (Size-based: Smallest First)

```
1. ifeval-rlvr-verl   (14,973 examples)
2. ifbench-verl       (95,373 examples)
```

## Deduplication Method

### Normalization

Conservative text normalization to minimize false positives:
- Strip whitespace
- Normalize internal whitespace
- Remove LaTeX formatting variations (`\\,`, `\\quad`, etc.)
- Normalize quotes and dashes
- Remove zero-width characters
- **Preserves**: Case sensitivity, LaTeX content

### Hashing

- Algorithm: SHA-256
- Input: Normalized `prompt[0]['content']` field
- Output: 64-character hexadecimal hash
- Collision detection: Tracks hash-to-text mappings

### Priority Strategy

Size-based priority (smallest datasets first) rationale:
- Preserves rare problems from small, curated datasets
- Maximizes diversity of final collection
- Retains unique contributions from each dataset
- Alternative: Quality-based priority (not used here)

## Output Structure

```
_deduplicated/
├── qa/
│   ├── phase1-intra/
│   │   ├── docqa-rl-verl/
│   │   │   ├── data/
│   │   │   │   └── train-00000.parquet
│   │   │   └── stats.json
│   │   ├── guru-logic-verl/
│   │   │   └── ...
│   │   └── phase1_summary.json
│   ├── phase2-inter/
│   │   ├── combined/
│   │   │   └── data/
│   │   │       ├── train-00000.parquet
│   │   │       └── ...
│   │   ├── splits/
│   │   │   ├── docqa-rl-verl.parquet
│   │   │   ├── guru-logic-verl.parquet
│   │   │   └── ...
│   │   └── stats/
│   │       └── phase2_stats.json
│   └── reports/
│       ├── deduplication_report.md
│       └── deduplication_summary.json
└── if/
    └── ... (same structure)
```

## Collections Configuration

### QA Collection

- **Datasets**: 5 (docqa-rl, guru-logic, toolrl-4k, guru-table, table-r1-zero)
- **Total Examples**: ~85K
- **License**: Apache 2.0 (with attribution for CC-BY-4.0 and MIT sources)
- **Task Categories**: question-answering, reasoning, table-qa, logic, tool-use

### IF Collection

- **Datasets**: 2 (ifeval-rlvr, ifbench)
- **Total Examples**: ~110K
- **License**: ODC-BY
- **Constraint Types**: 79 total (25 + 54)
- **Task Categories**: instruction-following, evaluation

## Tips and Best Practices

### Memory Management

- Use `--batch-size` to control memory usage (default: 10000)
- Large datasets automatically split into 500K row chunks
- Streaming mode for Phase 1 processing

### Performance

- Expected dedup rates:
  - QA: 5-15% (diverse datasets, less overlap)
  - IF: 10-30% (both IF evaluation, may have overlap)
- Processing time: ~10-30 minutes per collection (depends on size)

### Validation

```bash
# Check Phase 1 stats
cat _deduplicated/qa/phase1/phase1_summary.json

# Check Phase 2 stats
cat _deduplicated/qa/phase2/stats/phase2_stats.json

# Generate comprehensive report
python generate_report.py \
    --phase1-stats _deduplicated/qa/phase1/phase1_summary.json \
    --phase2-stats _deduplicated/qa/phase2/stats/phase2_stats.json \
    --output-dir _deduplicated/qa/reports
```

### Troubleshooting

**Import errors:**
```bash
# Make sure to run from correct directory
cd qa_data  # or if_data
python ../deduplicate_intra.py --all
```

**Missing datasets:**
- Check download completed successfully
- Verify data/ subdirectory exists with parquet files

**Schema validation errors:**
- Check VERL format compliance
- Verify prompt[0]['role'] == 'user'
- Ensure required fields present

## Example End-to-End Workflow

```bash
#!/bin/bash
# Complete QA collection deduplication workflow

# 1. Download
python download_datasets.py --collection qa --output-dir qa_data
cd qa_data

# 2. Phase 1
python ../deduplicate_intra.py --all --output-dir ../_deduplicated/qa/phase1 --verbose

# 3. Phase 2
python ../deduplicate_inter.py \
    --priority-file ../qa_priority.txt \
    --input-dir ../_deduplicated/qa/phase1 \
    --output-dir ../_deduplicated/qa/phase2 \
    --verbose

# 4. Generate report
cd ..
python generate_report.py \
    --phase1-stats _deduplicated/qa/phase1/phase1_summary.json \
    --phase2-stats _deduplicated/qa/phase2/stats/phase2_stats.json \
    --output-dir _deduplicated/qa/reports

# 5. Create splits
python create_splits.py \
    --combined-dir _deduplicated/qa/phase2/combined/data \
    --output-dir _deduplicated/qa/phase2/splits

# 6. Upload to Hub
python upload_unified.py \
    --splits-dir _deduplicated/qa/phase2/splits \
    --repo-name sungyub/qa-verl-unified \
    --collection-type qa \
    --dedup-stats _deduplicated/qa/phase1/phase1_summary.json

echo "✅ QA collection complete!"
```

## Related

- Original math deduplication: `~/data/math/scripts/`
- HuggingFace collections: https://huggingface.co/collections/sungyub/
- VERL format documentation: https://github.com/volcengine/verl
