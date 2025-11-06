# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataTrove is a library for processing, filtering, and deduplicating text data at very large scale. It provides prebuilt processing blocks and a framework for custom functionality. Pipelines are platform-agnostic, running locally or on slurm/ray clusters with relatively low memory usage.

## Development Commands

### Setup

**Using uv (Recommended):**
```bash
uv pip install -e ".[dev]"  # Install with all dev dependencies
uv run pre-commit install   # Install pre-commit hooks

# Run scripts with uv
uv run python examples/example_script.py
```

**Using pip:**
```bash
pip install -e ".[dev]"  # Install with all dev dependencies
pre-commit install       # Install pre-commit hooks
```

Note: This project requires Python >= 3.10.0

### Testing

**Using uv:**
```bash
# Run all tests
uv run pytest -sv ./tests/

# Run specific test file
uv run pytest -sv ./tests/pipeline/test_filters.py

# Run specific test
uv run pytest -sv ./tests/pipeline/test_filters.py::test_filter_name
```

**Using make:**
```bash
# Run all tests
make test
```

### Code Quality
```bash
# Check code quality (linter + formatter)
make quality

# Auto-fix code style issues
make style
```

### CLI Tools
After installation, several command-line tools are available:
- `merge_stats` - Merge statistics from multiple tasks
- `check_dataset` - Validate dataset integrity
- `failed_logs` - View logs from failed tasks
- `inspect_data` - Inspect processed data
- `jobs_status` - Check status of pipeline jobs
- `track_jobs` - Track multiple pipeline jobs
- `launch_pickled_pipeline` - Execute pickled pipeline configurations

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

**Dedup** (`dedup/`): Deduplication algorithms
- `MinhashDedup*`: Multi-stage minhash deduplication (signature → buckets → cluster → filter)
- `SentenceDedup`: Sentence-level exact deduplication
- `ExactSubstrings`: Substring deduplication
- Typically runs as multi-stage dependent pipelines

**Stats** (`stats/`): Collect dataset statistics
- Two-stage process: collect per-shard → merge across shards
- Groupings: `summary`, `fqdn`, `suffix`, `histogram`
- Results saved as `MetricStatsDict` JSON files

**Tokens** (`tokens/`): Tokenization and token operations
- `TokensCounter`: Count tokens in documents
- `DocumentTokenizer`: Tokenize and save tokens

**Inference** (`inference/`): Run LLM inference for synthetic data
- `InferenceRunner`: Supports vLLM, SGLang, and remote vLLM endpoints
  - **Local servers** (`server_type="vllm"` or `"sglang"`): Automatically spawns and manages server processes
  - **Remote servers** (`server_type="vllm-remote"`): Connects to existing external vLLM endpoints
- Automatic checkpointing via `checkpoints_local_dir` and `records_per_chunk`
- Server architecture:
  - `LocalInferenceServer`: Base for local server management (process spawning, port finding, logging)
  - `RemoteInferenceServer`: Base for external endpoint connections (health checks, no process management)

**Using External vLLM Server:**
```python
from datatrove.pipeline.inference import InferenceRunner, InferenceConfig
from datatrove.pipeline.writers import JsonlWriter

# Connect to an existing vLLM server instead of spawning a local one
config = InferenceConfig(
    server_type="vllm-remote",
    model_name_or_path="meta-llama/Llama-3-8B",
    external_endpoint="http://my-vllm-server.com:8000",  # Required for vllm-remote
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

**Supported Dataset Types:**
- **Math datasets** (via `math_verify`):
  - GSM8K: `openai/gsm8k`
  - MATH: `lighteval/MATH`, `DigitalLearningGmbH/MATH-lighteval`, `HuggingFaceH4/MATH-500`
  - Numina datasets: `numina_aops_forum`, `numina_synthetic_math`, `numina_amc_aime`, etc.
  - Uses answer extraction and exact match comparison
  - Supports both XML (`<think>`, `\boxed{}`) and GPT OSS (`<|channel|>analysis`, `<|channel|>final`) formats
  - Auto-format detection via `format_type="auto"`

- **Code execution** (via `sandbox_fusion`):
  - `codecontests`, `apps`, `codeforces`, `taco`
  - Requires external sandbox fusion server for secure code execution
  - Must set `sandbox_fusion_url` parameter or will raise ValueError
  - **Multi-language support**: Python, C++, Java, Go, Rust, and 24+ more languages
  - Automatic language detection from code blocks (e.g., ````cpp`, ````java`)
  - Language mapping: `py3`/`py2` → `python`, `c++`/`c++17` → `cpp`
  - Fallback to Python for unsupported languages

- **Geometry** (via `geo3k`):
  - `hiyouga/geometry3k`
  - Uses mathruler for geometric problem evaluation

- **QA/SearchR1** (via `search_r1_like_qa_em`):
  - `searchR1_nq`, `searchR1_triviaqa`, `searchR1_popqa`, `searchR1_hotpotqa`, etc.
  - Exact match scoring for question-answering tasks
  - Ground truth must be dict format: `{"target": [answers]}`

- **ToolRL** (via `toolrl`):
  - `rlla`, `toolrl`, `tool_learning`, `toolace`, `hammer`, `xlam`, `sungyub/toolrl-verl`
  - Evaluates tool learning tasks with three components:
    - Format reward: Structure validation for XML (`<think>`, `<tool_call>`, `<response>`) or GPT OSS (`<|channel|>analysis`, `to=functions.X`, `<|channel|>final`) formats
    - Correctness reward: Tool name and parameter matching (frequency-based scoring)
    - Length reward (optional): Reasoning length in thinking sections
  - **Format Support**:
    - **XML format**: Traditional tags (`<think>`, `<tool_call>`, `<response>`) - Qwen/Llama default
    - **GPT OSS format**: Special tokens (`<|start|>`, `<|channel|>`, `<|message|>`, `<|end|>`, `<|call|>`, `<|return|>`) - GPT OSS 120B
    - Auto-detection by default via `format_type="auto"`
  - Supports Llama, Qwen, and GPT OSS chat templates with auto-detection
  - No external dependencies required (pure Python)
  - Returns dict with `score`, `reward_fmt`, `reward_correct`, `reward_length`, `reward_think`
  - **Note**: `toolrl_gpt_oss.py` is deprecated - use unified `toolrl.py` with `format_type="gpt_oss"` or `"auto"`

- **CodeV** (via `codev`):
  - `codev`, `sungyub/codev-r1-verl`
  - Verilog code generation with equivalence checking
  - **System Requirements**:
    - Icarus Verilog (`iverilog`) must be installed:
      - macOS: `brew install icarus-verilog`
      - Ubuntu/Debian: `apt-get install iverilog`
      - Verify: `which iverilog` should return a valid path
    - Python dependencies: `psutil`, `networkx` (auto-installed with `reward_scoring`)
  - **External Dependencies**:
    - Requires Sandbox Fusion server for Verilog simulation
    - Must set `sandbox_fusion_url` parameter or will raise ValueError
  - **Format Support**:
    - **XML format**: `<think>`, `<answer>` tags
    - **GPT OSS format**: `<|channel|>analysis`, `<|channel|>final` blocks
    - Auto-detection via `format_type="auto"`
  - Evaluation process:
    - Extracts Verilog code from markdown code blocks
    - Generates automatic testbenches (random + directed tests)
    - Runs equivalence checking via Sandbox Fusion
    - Supports multiple ground truth variants per problem
  - Returns dict with `score`, `reward_fmt`, `reward_think`, verification details

**Installation:**
```bash
pip install -e ".[reward_scoring]"  # Installs all reward scoring dependencies
# For CodeV: Also install Icarus Verilog (see system requirements above)
```

**Usage Example:**
```python
from datatrove.utils.reward_score import compute_score

# Math dataset scoring - XML format (default)
score = compute_score(
    data_source="openai/gsm8k",
    solution_str="<think>Let me calculate...</think>\nThe answer is \\boxed{42}",
    ground_truth="\\boxed{42}",
    format_type="auto"  # Auto-detect format (xml/gpt_oss)
)

# Math dataset scoring - GPT OSS format
score = compute_score(
    data_source="openai/gsm8k",
    solution_str=(
        "<|start|>assistant<|channel|>analysis<|message|>Let me calculate...<|end|>\n"
        "<|start|>assistant<|channel|>final<|message|>\\boxed{42}<|return|>"
    ),
    ground_truth="\\boxed{42}",
    format_type="gpt_oss"
)

# Code execution scoring - Python (requires sandbox)
score = compute_score(
    data_source="codecontests",
    solution_str="```python\ndef solution(): return 42\n```",
    ground_truth={"inputs": ["5"], "outputs": ["42"]},
    sandbox_fusion_url="http://sandbox-server:5000"
)

# Code execution scoring - C++ (automatic language detection)
score = compute_score(
    data_source="codecontests",
    solution_str="```cpp\n#include <iostream>\nint main() { std::cout << 42; }\n```",
    ground_truth={"inputs": ["5"], "outputs": ["42"]},
    sandbox_fusion_url="http://sandbox-server:5000"
)

# Code execution scoring - Java (automatic language detection)
score = compute_score(
    data_source="codecontests",
    solution_str="```java\npublic class Main { public static void main(String[] args) { System.out.println(42); } }\n```",
    ground_truth={"inputs": ["5"], "outputs": ["42"]},
    sandbox_fusion_url="http://sandbox-server:5000"
)

# ToolRL scoring - Tool learning tasks (XML format)
score = compute_score(
    data_source="toolrl",
    solution_str="<think>I need to search for information</think>\n<tool_call>\n{\"name\": \"search\", \"parameters\": {\"query\": \"AI\"}}\n</tool_call>",
    ground_truth="<think>...</think>\n<tool_call>\n{\"name\": \"search\", \"parameters\": {\"query\": \"AI\"}}\n</tool_call>",
    model_type="auto",  # Auto-detect chat template (llama/qwen)
    format_type="auto",  # Auto-detect format (xml/gpt_oss)
    enable_length_reward=True
)

# ToolRL scoring - GPT OSS format
score = compute_score(
    data_source="toolrl",
    solution_str=(
        "<|start|>assistant<|channel|>analysis<|message|>I need to search for information<|end|>\n"
        "<|start|>assistant to=functions.search<|channel|>commentary json<|message|>"
        "{\"query\": \"AI\"}<|call|>"
    ),
    ground_truth=(
        "<|start|>assistant<|channel|>analysis<|message|>...<|end|>\n"
        "<|start|>assistant to=functions.search<|channel|>commentary json<|message|>"
        "{\"query\": \"AI\"}<|call|>"
    ),
    format_type="gpt_oss",  # Explicit GPT OSS format
    enable_length_reward=True
)

# CodeV scoring - Verilog code generation (XML format, requires sandbox + iverilog)
# Note: New JSON format (recommended). Legacy pickle bytes format is still supported for backward compatibility.
score = compute_score(
    data_source="codev",
    solution_str="<think>Creating a simple adder</think>\n<answer>```verilog\nmodule adder(input a, input b, output sum); assign sum = a + b; endmodule\n```</answer>",
    ground_truth=json.dumps({
        "code": "module adder_gold(input a, input b, output sum); assign sum = a + b; endmodule",
        "input_port_width": [1, 1],  # Port widths as lists (JSON compatible)
        "output_port_width": [1],
        "clock_port_polarity": [],
        "reset_port_polarity_sync": []
    }),
    sandbox_fusion_url="http://sandbox-server:5000",
    format_type="auto"
)

# CodeV scoring - GPT OSS format
score = compute_score(
    data_source="codev",
    solution_str=(
        "<|start|>assistant<|channel|>analysis<|message|>Creating a simple adder<|end|>\n"
        "<|start|>assistant<|channel|>final<|message|>```verilog\n"
        "module adder(input a, input b, output sum); assign sum = a + b; endmodule\n"
        "```<|return|>"
    ),
    ground_truth=json.dumps({
        "code": "module adder_gold(input a, input b, output sum); assign sum = a + b; endmodule",
        "input_port_width": [1, 1],
        "output_port_width": [1],
        "clock_port_polarity": [],
        "reset_port_polarity_sync": []
    }),
    sandbox_fusion_url="http://sandbox-server:5000",
    format_type="gpt_oss"
)
```

**Supported Languages in Sandbox Fusion:**
- Programming: python, cpp, java, go, rust, javascript (nodejs), typescript, kotlin, swift, scala
- Scripting: bash, php, perl, ruby, lua, R
- Testing: pytest, junit, jest, go_test
- Other: csharp, sql, cuda, verilog, lean, racket, D_ut

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

## Common Patterns in Examples

All examples are in `examples/`:
- `fineweb.py`: Full reproduction of FineWeb dataset (filtering + minhash dedup)
- `process_common_crawl_dump.py`: CommonCrawl WARC processing pipeline
- `minhash_deduplication.py`: Complete minhash deduplication workflow
- `sentence_deduplication.py`: Sentence-level deduplication
- `tokenize_c4.py`: Tokenization from HuggingFace datasets
- `summary_stats.py`: Collecting and merging statistics

## Notes for Development

- DataFolder paths support local, S3, and HuggingFace Hub via fsspec
- Use `get_datafolder()` to parse various path formats: str, tuple, or DataFolder
- Executors save pickled versions of themselves for slurm job arrays
- Color logging can be controlled via `DATATROVE_COLORIZE_LOGS` env var
- All pipeline blocks should yield Documents, never return lists (memory efficiency)
