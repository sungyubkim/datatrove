# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataTrove is a library for processing, filtering, and deduplicating text data at very large scale. It provides prebuilt processing blocks and a framework for custom functionality. Pipelines are platform-agnostic, running locally or on slurm/ray clusters with relatively low memory usage.

## Development Commands

### Setup & Testing
```bash
# Setup (Python >= 3.10.0 required)
uv pip install -e ".[dev]"     # Recommended: Install with uv
uv run pre-commit install      # Setup pre-commit hooks
# Alternative: pip install -e ".[dev]" && pre-commit install

# Testing
uv run pytest -sv ./tests/                           # Run all tests
uv run pytest -sv ./tests/pipeline/test_filters.py   # Specific test file
make test                                            # Run tests via make

# Code Quality
make quality    # Check code quality (linter + formatter)
make style      # Auto-fix code style issues
```

### CLI Tools
After installation: `merge_stats`, `check_dataset`, `failed_logs`, `inspect_data`, `jobs_status`, `track_jobs`, `launch_pickled_pipeline`

## Architecture

### Core Concepts

**Document**: The fundamental data unit (`src/datatrove/data.py`)
- `text`: The actual text content
- `id`: Unique identifier (string)
- `metadata`: Dictionary for additional information
- `media`: List of associated media (future use)

**Pipeline**: A list of processing steps that transform documents
- Each step takes a generator of `Document` and yields `Document`
- Steps can be `PipelineStep` instances, custom callables, or sequences
- Data flows through pipeline via generator pattern (memory efficient)

**PipelineStep**: Base class for all processing blocks (`src/datatrove/pipeline/base.py`)
- `run(data, rank, world_size)`: Main processing method
- `stat_update()`: Track statistics during processing
- `track_time()`: Context manager for timing code blocks
- Automatically checks dependencies via `_requires_dependencies`

**Executor**: Manages pipeline execution across different platforms (`src/datatrove/executor/`)
- `LocalPipelineExecutor`: Multi-process execution on local machine
- `SlurmPipelineExecutor`: Distributed execution on slurm clusters
- `RayPipelineExecutor`: Distributed execution using Ray
- All executors share common interface: `run()`, `world_size`, task completion tracking

**Task & Sharding**: Parallelization is achieved by dividing work into tasks
- Each task processes a non-overlapping shard of input files
- Files are distributed: task `i` processes files `i, i+N, i+2N, ...` where N = world_size
- Completion tracking via empty marker files in `${logging_dir}/completions/`
- Failed tasks can be rerun by relaunching the same executor (don't change task count)

**DataFolder**: Abstraction over filesystem operations (`src/datatrove/io.py`)
- Wraps fsspec's `DirFileSystem` for local/remote file operations
- `get_shard(rank, world_size)`: Deterministic file sharding
- `list_files()`: List files with optional glob patterns
- `open()`: Open files with automatic parent directory creation
- Supports local, S3, HuggingFace Hub, and other fsspec backends

### Pipeline Block Types

All blocks in `src/datatrove/pipeline/`:

**Readers** (`readers/`): Read data and yield Documents
- `WarcReader`, `JsonlReader`, `CSVReader`, `ParquetReader`, `HuggingFaceReader`
- Common args: `data_folder`, `text_key`, `id_key`, `default_metadata`, `limit`, `glob_pattern`
- Implement `_get_document_from_dict()` to transform raw data to Documents

**Writers** (`writers/`): Save Documents to disk/cloud
- `JsonlWriter`, `ParquetWriter`, `HuggingFaceWriter`
- Use `output_filename` templates: `${rank}`, `${id}`, `${metadata_key}`
- Inherit from `DiskWriter` base class
- **IMPORTANT - macOS File Exclusion**: When uploading to Hugging Face Hub, always exclude macOS system files:
  - `.DS_Store`: Finder metadata files
  - `._*`: AppleDouble files (extended attributes on non-macOS filesystems like ExFAT)
  - `.Spotlight-V100`, `.Trashes`: Other macOS system files
  - Use `ignore_patterns` parameter in `HuggingFaceWriter` or filter files before upload
  - Example: When using external drives (ExFAT/NTFS), these files are automatically created by macOS

**Extractors** (`extractors/`): Extract text from raw formats
- `Trafilatura`: HTML text extraction (most common)
- Transform document text in-place

**Filters** (`filters/`): Remove documents based on criteria
- Return `True` to keep, `False` to remove
- Can save removed docs via `exclusion_writer` parameter
- Examples: `LanguageFilter`, `GopherQualityFilter`, `URLFilter`, `C4QualityFilter`
- Inherit from `BaseFilter`

**Formatters** (`formatters/`): Modify document content
- `PIIFormatter`: Remove personally identifiable information
- `FTFYFormatter`: Fix text encoding issues
- `SymbolLinesRemover`: Remove lines containing excessive symbols/special characters
- `RLVRFormatter`: Format data for RLVR (Reinforcement Learning from Verification and Reasoning)
- `MathDatasetCleaner`: Clean and normalize math datasets by removing artifacts and standardizing formats

**Dedup** (`dedup/`): Deduplication algorithms
- `MinhashDedup*`: Multi-stage minhash deduplication (signature → buckets → cluster → filter)
- `SentenceDedup`: Sentence-level exact deduplication
- `ExactSubstrings`: Substring deduplication
- `BloomFilter`: Memory-efficient probabilistic deduplication using bloom filters
- `URLDedup`: URL-based deduplication for web-scraped data
- Typically runs as multi-stage dependent pipelines

**Stats** (`stats/`): Collect dataset statistics
- Two-stage process: collect per-shard → merge across shards
- Groupings: `summary`, `fqdn`, `suffix`, `histogram`
- Results saved as `MetricStatsDict` JSON files

**Tokens** (`tokens/`): Tokenization and token operations
- `TokensCounter`: Count tokens in documents
- `DocumentTokenizer`: Tokenize and save tokens

**Inference** (`inference/`): Run LLM inference for synthetic data
- `InferenceRunner`: Supports vLLM, SGLang, and remote endpoints
  - **Local servers** (`server_type="vllm"` or `"sglang"`): Automatically spawns and manages server processes
  - **Remote servers** (`server_type="endpoint"`): Connects to existing external endpoints
- Automatic checkpointing via `checkpoints_local_dir` and `records_per_chunk`
- Server architecture:
  - `LocalInferenceServer`: Base for local server management (process spawning, port finding, logging)
  - `RemoteInferenceServer`: Base for external endpoint connections (health checks, no process management)

**Using External Endpoint:**
```python
from datatrove.pipeline.inference import InferenceRunner, InferenceConfig
from datatrove.pipeline.writers import JsonlWriter

# Connect to an existing inference endpoint instead of spawning a local one
config = InferenceConfig(
    server_type="endpoint",
    model_name_or_path="meta-llama/Llama-3-8B",
    endpoint_url="http://my-vllm-server.com:8000",  # Required for endpoint
    temperature=0.7,
    max_concurrent_requests=100,
)

runner = InferenceRunner(
    query_builder=my_query_builder,
    config=config,
    output_writer=JsonlWriter("output/synthetic_data"),
    checkpoints_local_dir="checkpoints/",
)
```

**Reward Scoring** (`utils/reward_score/`): Evaluation utilities for synthetic data
- Integrated from VERL project for scoring generated responses
- Supports multiple dataset types with automatic scoring method selection
- `compute_score()`: Main entry point that routes to appropriate scorer

**Preprocessing** (`preprocessing/`): Data preprocessing utilities
- `rlvr_to_ifbench`: Convert RLVR format to IFBench format for instruction-following benchmarks
- Located in `src/datatrove/preprocessing/`

**Supported Dataset Types:**

| Category | Datasets | Scorer | Key Requirements | Format Support |
|----------|----------|--------|------------------|----------------|
| **Math** | gsm8k, MATH, numina_* | `math_verify` | None | XML, GPT OSS (`format_type="auto"`) |
| **Code** | codecontests, apps, codeforces, taco | `sandbox_fusion` | `sandbox_fusion_url` | 24+ languages (auto-detect) |
| **Geometry** | geometry3k | `geo3k` | mathruler | N/A |
| **QA/SearchR1** | searchR1_nq, searchR1_triviaqa, etc. | `search_r1_like_qa_em` | None | N/A |
| **ToolRL** | rlla, toolrl, toolace, hammer, xlam | `toolrl` | None | XML, GPT OSS (`format_type="auto"`) |
| **IFEval** | ifeval, ifbench, IF_multi_constraints | `ifeval` | langdetect, nltk | XML, GPT OSS |
| **CodeV** | codev, codev-r1-verl | `codev` | `sandbox_fusion_url`, iverilog | XML, GPT OSS (`format_type="auto"`) |
| **Table** | WTQ, TabFact, FeTaQA, hitab, finqa | `table_*` | rouge-score, sacrebleu | N/A |
| **DocQA** | multihoprag, musique, docmath | `docqa`/`docmath` | None | N/A |
| **Long Context** | long_toc_choices | `long` | None | N/A |
| **Logic** | arcagi, puzzle datasets | `logic` | None | N/A |

**Key Features:**
- **Multi-language code execution**: Python, C++, Java, Go, Rust, JavaScript, and 20+ more (automatic language detection from code blocks)
- **Format auto-detection**: Supports both XML (`<think>`, `<tool_call>`) and GPT OSS (`<|channel|>`) formats with `format_type="auto"`
- **VERL compatibility**: Environment variables for reward tuning (WITHLENGTH, CORRECTMAX1, SCHEDULEREWARD, etc.)
- **CodeV requirements**: Icarus Verilog (`brew install icarus-verilog` on macOS) + Sandbox Fusion server

**Installation:**
```bash
pip install -e ".[reward_scoring]"  # Installs all reward scoring dependencies
# For CodeV: brew install icarus-verilog (macOS) or apt-get install iverilog (Ubuntu)
```

**Usage Examples:**
```python
from datatrove.utils.reward_score import compute_score

# Math dataset - Auto-format detection
score = compute_score(
    data_source="openai/gsm8k",
    solution_str="<think>Let me calculate...</think>\nThe answer is \\boxed{42}",
    ground_truth="\\boxed{42}",
    format_type="auto"  # Auto-detects XML or GPT OSS format
)

# Code execution - Multi-language support (Python, C++, Java, Go, etc.)
score = compute_score(
    data_source="codecontests",
    solution_str="```python\ndef solution(): return 42\n```",  # Also: ```cpp, ```java, etc.
    ground_truth={"inputs": ["5"], "outputs": ["42"]},
    sandbox_fusion_url="http://sandbox-server:5000"
)

# ToolRL - Format auto-detection with length reward
score = compute_score(
    data_source="toolrl",
    solution_str="<think>I need to search</think>\n<tool_call>{\"name\": \"search\", ...}</tool_call>",
    ground_truth="...",
    format_type="auto",  # Works with both XML and GPT OSS
    enable_length_reward=True
)

# CodeV - Verilog code generation (requires sandbox + iverilog)
score = compute_score(
    data_source="codev",
    solution_str="<think>Creating adder</think>\n<answer>```verilog\nmodule adder...\n```</answer>",
    ground_truth=json.dumps({
        "code": "module adder_gold...",
        "input_port_width": [1, 1],
        "output_port_width": [1],
        "clock_port_polarity": [],
        "reset_port_polarity_sync": []
    }),
    sandbox_fusion_url="http://sandbox-server:5000",
    format_type="auto"
)
```

**Supported Languages in Sandbox Fusion:**
- **Programming**: python, cpp, java, go, rust, javascript (nodejs), typescript, kotlin, swift, scala
- **Scripting**: bash, php, perl, ruby, lua, R
- **Testing**: pytest, junit, jest, go_test
- **Other**: csharp, sql, cuda, verilog, lean, racket, D_ut

See `examples/verl_data_processing.py` for complete VERL data processing pipeline with multi-response generation and scoring.

### Key Implementation Patterns

**Custom Pipeline Blocks**: Three approaches
1. List of Documents (for testing): `[Document(...), Document(...)]`
2. Custom function: `def process(data, rank, world_size) -> DocumentsPipeline`
3. Custom class inheriting from `PipelineStep` or subclass (`BaseFilter`, `BaseExtractor`, etc.)

**Statistics Tracking**:
```python
with self.track_time():
    # processing code
    self.stat_update("metric_name", value=count, unit="doc")
```

**Dependency Pipeline Execution**:
```python
stage2 = SlurmPipelineExecutor(..., depends=stage1)
stage3 = SlurmPipelineExecutor(..., depends=stage2)
stage3.run()  # Automatically runs stage1 → stage2 → stage3
```

**Multi-stage Deduplication**: See `examples/minhash_deduplication.py`
- Stage 1: Compute signatures (`MinhashDedupSignature`)
- Stage 2: Create buckets (`MinhashDedupBuckets`)
- Stage 3: Cluster duplicates (`MinhashDedupCluster`)
- Stage 4: Filter documents (`MinhashDedupFilter`)

## File Locations

- Core data structures: `src/datatrove/data.py`
- Pipeline base: `src/datatrove/pipeline/base.py`
- Executor base: `src/datatrove/executor/base.py`
- I/O utilities: `src/datatrove/io.py`
- Logging utilities: `src/datatrove/utils/logging.py`
- Stats handling: `src/datatrove/utils/stats.py`
- Reward scoring: `src/datatrove/utils/reward_score/` (main: `__init__.py`, subdirectories for each scorer)
- Preprocessing utilities: `src/datatrove/preprocessing/`
- Inference servers: `src/datatrove/pipeline/inference/servers/` (base classes and implementations)
- Command-line tools: `src/datatrove/tools/`

## Testing Strategy

Tests are organized by component in `tests/`:
- `tests/pipeline/` - Tests for readers, filters, extractors, dedup, etc.
- `tests/executor/` - Tests for executors
- `tests/test_io.py` - Tests for I/O operations
- Use `tests/utils.py` for test fixtures and helpers

## Important Implementation Details

**Logging Structure**: Each pipeline execution creates:
```
${logging_dir}/
├── executor.json          # Serialized executor config
├── ranks_to_run.json      # List of tasks being run
├── logs/
│   └── task_00000.log     # Individual task logs
├── completions/
│   └── 00000              # Empty marker files for completed tasks
├── stats/
│   └── 00000.json         # Per-task statistics
└── stats.json             # Merged global statistics
```

**Generator Pattern**: Pipelines use generators for memory efficiency
- Documents flow through pipeline without loading entire dataset into memory
- Use `deque(pipelined_data, maxlen=0)` to exhaust generator at pipeline end

**Sharding Guarantees**: File distribution is deterministic
- Same `world_size` always produces same sharding
- Never change `world_size` when re-running failed tasks
- Each file processed by exactly one task

**Dependency Checking**: `PipelineStep.__new__` checks `_requires_dependencies`
- Add `_requires_dependencies = ["package_name"]` to custom blocks
- Checked via `check_required_dependencies()` from `utils/_import_utils.py`

**Parquet Schema Consistency**: When writing to Parquet format, all fields must have consistent types across rows to prevent schema mismatch errors:
- **Critical Rule**: Use empty string `""` instead of `None` for string fields in fallback/error cases
- **Why**: PyArrow/Parquet cannot handle `None` (null type) mixed with `str` in same column
- **Examples**:
  - ✅ Success case: `{"text": "response", "inference_error": "", "score_error": ""}`
  - ✅ Failure case: `{"text": "", "inference_error": "timeout", "score_error": "scoring failed"}`
  - ❌ Wrong (causes schema mismatch): `{"text": None, "inference_error": None}`
- **Applies to**: All nullable string fields including text, error messages, metadata values, etc.
- **Related fields**: `text`, `inference_error`, `score_error`, and any custom string fields
- See `examples/verl_data_processing.py` for implementation example with unified response objects

## Example Scripts

**Data Processing**: `fineweb.py`, `process_common_crawl_dump.py`, `filter_hf_dataset.py`
**Deduplication**: `minhash_deduplication.py`, `sentence_deduplication.py`, `url_deduplication.py`, `exact_substrings.py`
**Tokenization**: `tokenize_c4.py`, `tokenize_from_hf_to_s3.py`
**LLM/VERL**: `inference_example_chunked.py`, `verl_data_processing.py`, `convert_toolrl_to_gpt_oss.py`
**Preprocessing**: `preprocess_codecontests_plus.py`
**Stats**: `summary_stats.py`

All located in `examples/` directory.

## Notes for Development

- DataFolder paths support local, S3, and HuggingFace Hub via fsspec
- Use `get_datafolder()` to parse various path formats: str, tuple, or DataFolder
- Executors save pickled versions of themselves for slurm job arrays
- Color logging can be controlled via `DATATROVE_COLORIZE_LOGS` env var
- All pipeline blocks should yield Documents, never return lists (memory efficiency)
