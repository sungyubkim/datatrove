# Scripts Directory

This directory contains utility scripts for processing, cleaning, and managing VERL-format datasets. The scripts are organized into functional categories to support the complete dataset lifecycle from conversion to Hub upload.

## Directory Structure

```
scripts/
├── cleaning/           # Data cleaning and artifact removal
├── conversion/         # Format conversion utilities
├── deduplication/      # Duplicate detection and removal
├── hub/               # Hugging Face Hub management
├── processing/        # Combined processing workflows
├── upload/            # Dataset upload utilities
└── validation/        # Data quality verification
```

---

## 📁 cleaning/

Scripts for cleaning and normalizing VERL-format math datasets.

### `clean_math_dataset.py`
Streaming-based math dataset cleaner with configurable presets.

**Purpose**: Clean VERL math datasets by removing artifacts (problem numbers, contest metadata, etc.) while preserving ground truth answers.

**Features**:
- Streaming mode for memory efficiency
- Multiple cleaning presets (orz-math, openr1-math, skywork-or1, dapo-math)
- Comprehensive statistics and reporting
- Dataset shortcuts for common datasets

**Usage**:
```bash
# Clean ORZ dataset with auto-detected preset
python scripts/cleaning/clean_math_dataset.py \
    --dataset orz \
    --output ./output/orz-cleaned/

# Clean with specific preset and sample limit
python scripts/cleaning/clean_math_dataset.py \
    --dataset sungyub/openr1-math-verl \
    --preset openr1-math \
    --output ./output/openr1-cleaned/ \
    --max-samples 10000
```

**Key Parameters**:
- `--dataset`: Dataset name or shortcut (orz, openr1, skywork, dapo)
- `--preset`: Cleaning preset (auto-detected if not specified)
- `--output`: Output directory for cleaned parquet files
- `--max-samples`: Limit for testing (optional)

---

## 🔄 conversion/

Scripts for converting datasets to VERL format.

### `convert_dapo_to_verl.py`
Convert DAPO-Math-17K-cleaned to VERL format.

**Purpose**: Minimal schema conversion from Q&A format to VERL format without text cleaning.

**Features**:
- Streaming mode for large datasets
- Preserves original text content
- Schema-only transformation

**Usage**:
```bash
python scripts/conversion/convert_dapo_to_verl.py
```

**Output**: `./output/dapo-math-verl-converted/train.parquet`

---

### `convert_openr1_to_verl.py`
Convert OpenR1-Math-220k to VERL format (without correctness filtering).

**Purpose**: Convert all 220k samples from OpenR1-Math to VERL schema for cleaning/deduplication comparison.

**Features**:
- Preserves all samples (no correctness filtering)
- Batch processing with progress tracking
- PyArrow-based streaming writes
- Schema validation

**Usage**:
```bash
# Basic conversion
python scripts/conversion/convert_openr1_to_verl.py

# Custom output location
python scripts/conversion/convert_openr1_to_verl.py \
    --output data/openr1-raw.parquet \
    --batch-size 20000
```

**Key Parameters**:
- `--source`: Source dataset (default: open-r1/OpenR1-Math-220k)
- `--output`: Output parquet file path
- `--batch-size`: Processing batch size
- `--split`: Dataset split (default/all/extended)

---

### `convert_orz_to_verl_streaming.py`
Convert ORZ-Math-72k to VERL format (minimal version).

**Purpose**: Minimal schema conversion to investigate data quality issues (e.g., $ endings in source data).

**Features**:
- NO text cleaning or modification
- Detects problem numbering patterns
- Tracks $ ending occurrences
- Diagnostic reporting

**Usage**:
```bash
python scripts/conversion/convert_orz_to_verl_streaming.py
```

**Output**: `./output/orz-math-recreated/train.parquet` + diagnostic reports

---

### `prepare_for_hub_upload.py`
Prepare cleaned datasets for Hugging Face Hub upload.

**Purpose**: Add required metadata fields and normalize schema for Hub compatibility.

**Features**:
- Adds `original_dataset` field to extra_info
- Normalizes extra_info to standard 3-field schema
- Generates both individual and unified dataset versions
- Backward compatibility validation

**Usage**:
```bash
python scripts/conversion/prepare_for_hub_upload.py
```

**Outputs**:
- `output/hub-upload/openr1-math-verl/train.parquet` (individual dataset)
- `output/hub-upload/math-verl-unified/openr1-math-verl.parquet` (unified dataset)

---

## 🔍 deduplication/

Two-phase deduplication pipeline for removing exact duplicates.

### `deduplicate_intra.py` (Phase 1)
Remove duplicates within individual datasets.

**Purpose**: Deduplicate each dataset independently using SHA-256 hash of problem text.

**Features**:
- Streaming batch processing
- Collision detection
- Per-dataset statistics
- Top duplicate tracking
- Standard VERL schema output

**Usage**:
```bash
# Process single dataset
python scripts/deduplication/deduplicate_intra.py \
    --dataset docqa-rl-verl

# Process all datasets in directory
python scripts/deduplication/deduplicate_intra.py --all

# Dry run (count only, no output)
python scripts/deduplication/deduplicate_intra.py \
    --dataset toolrl-4k-verl \
    --dry-run
```

**Key Parameters**:
- `--dataset`: Dataset name to process
- `--all`: Process all datasets in input-dir
- `--input-dir`: Root directory containing datasets (default: .)
- `--output-dir`: Output directory (default: _deduplicated/phase1-intra)
- `--batch-size`: Rows per batch (default: 10000)
- `--dry-run`: Count duplicates without writing output

**Output Structure**:
```
_deduplicated/phase1-intra/
└── {dataset-name}/
    ├── data/
    │   ├── train-00000.parquet
    │   ├── train-00001.parquet
    │   └── ...
    └── stats.json
```

---

### `deduplicate_inter.py` (Phase 2)
Remove duplicates across datasets using priority order.

**Purpose**: Deduplicate across multiple datasets, preserving higher-priority datasets.

**Features**:
- Priority-based deduplication
- Cross-dataset duplicate tracking
- Combined output creation
- Detailed duplicate source tracking

**Usage**:
```bash
# Run with priority file
python scripts/deduplication/deduplicate_inter.py \
    --priority-file qa_priority.txt

# Use current directory datasets (not Phase 1 output)
python scripts/deduplication/deduplicate_inter.py \
    --priority-file if_priority.txt \
    --use-current-datasets
```

**Priority File Format** (one dataset per line, smallest to largest priority):
```
dataset-low-priority
dataset-medium-priority
dataset-high-priority
```

**Key Parameters**:
- `--input-dir`: Phase 1 output directory (default: _deduplicated/phase1-intra)
- `--output-dir`: Output directory (default: _deduplicated/phase2-inter)
- `--priority-file`: File containing dataset priority order (required)
- `--use-current-datasets`: Read from ./{dataset}/data/ instead of input-dir
- `--no-combine`: Skip creating combined output

**Output Structure**:
```
_deduplicated/phase2-inter/
├── combined/
│   └── data/
│       ├── train-00000.parquet
│       ├── train-00001.parquet
│       └── ...
└── stats/
    └── phase2_stats.json
```

---

### `create_splits.py`
Split combined dataset by original_dataset field.

**Purpose**: Create separate dataset files from combined deduplicated dataset for Hub upload as splits.

**Features**:
- Automatic grouping by original_dataset
- Distribution statistics
- Hub-compatible split naming

**Usage**:
```bash
python scripts/deduplication/create_splits.py \
    --combined-dir _deduplicated/phase2-inter/combined/data \
    --output-dir _deduplicated/phase2-inter/splits
```

**Output**: One parquet file per dataset split.

---

### `utils.py`
Shared utility functions for deduplication pipeline.

**Functions**:
- `compute_hash()`: SHA-256 hashing of problem text
- `extract_problem_text()`: Extract problem from VERL format
- `validate_verl_row()`: Schema validation
- `normalize_row_schema()`: Schema normalization
- `DuplicationStats`: Statistics tracking class
- `save_stats()`, `format_number()`, `format_percentage()`: Reporting utilities

---

## 🚀 hub/

Hugging Face Hub management and validation scripts.

### `fix_data_source.py`
Fix data_source field values to match Hub conventions.

**Purpose**: Update data_source field format (e.g., 'dapo-math-17k' → 'DAPO-Math-17K').

**Usage**:
```bash
python scripts/hub/fix_data_source.py \
    --input input.parquet \
    --output output.parquet
```

---

### `prepare_standalone_update.py` & `prepare_unified_update.py`
Prepare datasets for Hub upload with proper metadata.

**Purpose**: Add required fields and normalize schema for individual and unified datasets.

**Features**:
- Schema normalization
- Backward compatibility validation
- Dual output (individual + unified versions)

---

### `validate_before_upload.py` & `validate_unified_upload.py`
Pre-upload validation scripts.

**Purpose**: Comprehensive validation before pushing to Hub.

**Validations**:
- Schema structure
- Field presence and types
- Sample quality
- extra_info normalization
- Backward compatibility

---

## ⚙️ processing/

Combined processing workflows.

### `process_local_dataset.py`
Unified pipeline: Cleaning + Deduplication.

**Purpose**: Process local parquet files or HuggingFace Hub datasets through both cleaning and deduplication in one step.

**Features**:
- Supports local files AND Hub datasets
- Streaming mode for memory efficiency
- Combined cleaning + dedup statistics
- Sample collection for reporting
- Auto-detects input type (local vs Hub)

**Usage**:
```bash
# Process local file
python scripts/processing/process_local_dataset.py \
    --input output/orz-math-recreated/train.parquet \
    --output output/orz-math-cleaned-v5/train.parquet \
    --preset orz-math

# Process HuggingFace Hub dataset
python scripts/processing/process_local_dataset.py \
    --input sungyub/orz-math-72k-verl \
    --output output/orz-math-cleaned-hub/train.parquet \
    --preset openr1-math
```

**Key Parameters**:
- `--input`: Local parquet file OR Hub dataset identifier
- `--output`: Output parquet file path
- `--preset`: Cleaning preset (default: orz-math)
- `--sample-rate`: Collect sample every N documents

**Output Files**:
- `{output}`: Cleaned and deduplicated parquet file
- `processing_report.txt`: Detailed processing report with samples
- `processing_stats.json`: Statistics in JSON format

---

## 📤 upload/

Dataset upload utilities.

### `upload_to_hub.py`
Upload prepared datasets to Hugging Face Hub.

**Purpose**: Upload validated datasets to Hub with proper versioning and commit messages.

**Features**:
- Pre-upload validation
- User confirmation prompts
- Dry-run mode
- Post-upload verification
- Handles both individual and unified datasets

**Usage**:
```bash
# Dry run (show what would be uploaded)
python scripts/upload/upload_to_hub.py --dry-run

# Actual upload with confirmation
python scripts/upload/upload_to_hub.py

# Auto-confirm (skip prompts)
python scripts/upload/upload_to_hub.py --yes
```

**Key Parameters**:
- `--dry-run`: Show upload plan without uploading
- `--skip-verify`: Skip post-upload verification
- `--yes` / `-y`: Auto-confirm without prompting

**Requirements**:
- Must be logged in: `huggingface-cli login`
- Must have write access to target datasets

---

## ✅ validation/

Data quality verification and analysis tools.

### `validate_hub_upload.py`
Comprehensive Hub upload validation suite.

**Purpose**: Validate datasets before Hub upload with extensive checks.

**Validations**:
1. Schema structure (required fields, types)
2. Sample quality (prompt, reward_model, extra_info)
3. extra_info normalization (exactly 3 fields: index, original_dataset, split)
4. Backward compatibility
5. Statistics validation
6. File structure
7. Cross-validation (individual vs unified)

**Usage**:
```bash
python scripts/validation/validate_hub_upload.py
```

**Exit Codes**:
- 0: All validations passed
- 1: Some validations failed

---

### `verify_data_integrity.py`
Data integrity verification tool.

**Purpose**: Verify that problem text is not truncated and check for suspicious patterns.

**Features**:
- Length distribution analysis
- Truncation pattern detection
- Specific pattern search
- Detailed sample reports
- JSON statistics export

**Usage**:
```bash
python scripts/validation/verify_data_integrity.py \
    --files output/dapo-math-verl-converted/train.parquet \
            output/dapo-math-cleaned-deduped/train.parquet \
    --search-patterns "Regular hexagon" "orthocenter"
```

**Key Parameters**:
- `--files`: List of parquet files to verify
- `--output`: Output directory for reports
- `--search-patterns`: Specific text patterns to search for

**Outputs**:
- `data_integrity_report.txt`: Human-readable report
- `integrity_stats.json`: Statistics in JSON format
- `detailed_examples.json`: Found pattern examples

---

### Other Validation Scripts

**`analyze_dollar_endings.py`**: Analyze $ ending patterns in math problems
**`analyze_maxclean_quality.py`**: Analyze quality of maximum cleaning preset
**`compare_datasets.py`**: Compare multiple dataset versions
**`compare_openr1_versions.py`**: Compare OpenR1 dataset versions
**`compare_versions.py`**: Generic version comparison
**`inspect_unique_samples.py`**: Inspect unique samples in dataset
**`validate_multipart_pattern.py`**: Validate multi-part problem patterns

---

## 🔧 Common Workflows

### Complete Dataset Processing Pipeline

```bash
# 1. Convert to VERL format
python scripts/conversion/convert_openr1_to_verl.py \
    --output output/openr1-raw/train.parquet

# 2. Clean and deduplicate
python scripts/processing/process_local_dataset.py \
    --input output/openr1-raw/train.parquet \
    --output output/openr1-cleaned/train.parquet \
    --preset openr1-math

# 3. Prepare for Hub upload
python scripts/conversion/prepare_for_hub_upload.py

# 4. Validate before upload
python scripts/validation/validate_hub_upload.py

# 5. Upload to Hub (dry run first)
python scripts/upload/upload_to_hub.py --dry-run
python scripts/upload/upload_to_hub.py --yes
```

### Multi-Dataset Deduplication

```bash
# Phase 1: Intra-dataset deduplication
python scripts/deduplication/deduplicate_intra.py --all

# Phase 2: Inter-dataset deduplication with priority
echo -e "dataset-a\ndataset-b\ndataset-c" > priority.txt
python scripts/deduplication/deduplicate_inter.py \
    --priority-file priority.txt

# Create splits for Hub upload
python scripts/deduplication/create_splits.py \
    --combined-dir _deduplicated/phase2-inter/combined/data \
    --output-dir _deduplicated/phase2-inter/splits
```

---

## 📊 VERL Schema Standard

All scripts use the standard VERL schema:

```python
{
    "data_source": str,              # Dataset identifier
    "prompt": [                      # List of messages
        {"role": str, "content": str}
    ],
    "ability": str,                  # "math", "code", etc.
    "reward_model": {
        "style": str,                # "rule", "model", etc.
        "ground_truth": str          # Expected answer
    },
    "extra_info": {
        "index": int,                # Sample index
        "original_dataset": str,     # Source dataset name
        "split": str                 # "train", "test", etc.
    }
}
```

**Note**: `extra_info` should contain ONLY these 3 fields for Hub compatibility.

---

## 🛠️ Development Tips

### Python Environment
All scripts require the datatrove package and dependencies:
```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

### Common Dependencies
- `datasets`: HuggingFace datasets library
- `pyarrow`: Parquet file handling
- `pandas`: Data manipulation
- `tqdm`: Progress bars
- `huggingface_hub`: Hub API access

### Streaming Mode
Most scripts use streaming mode for memory efficiency:
- Processes data incrementally
- Prevents RAM OOM on large datasets
- Essential for datasets > 1GB

### Error Handling
Scripts follow consistent patterns:
- Clear error messages with ✗ prefix
- Progress indicators with ✓ prefix
- Detailed reports with statistics
- JSON output for programmatic access

---

## 📝 Contributing

When adding new scripts:
1. Place in appropriate category directory
2. Include comprehensive docstring at top
3. Add usage examples in `--help`
4. Generate statistics and reports
5. Update this README
6. Follow existing naming conventions

---

## 🔗 Related Documentation

- Main project: [README.md](../README.md)
- CLAUDE.md: [../CLAUDE.md](../CLAUDE.md)
- DataTrove docs: [src/datatrove/](../src/datatrove/)
- Examples: [examples/](../examples/)

---

**Last Updated**: 2025-11-08
**Maintained By**: DataTrove Contributors
