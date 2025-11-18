"""
VERL data processing pipeline example.

This is an EDUCATIONAL EXAMPLE with hardcoded configuration meant for learning
and experimentation. For production use with CLI interface, see:
    - CLI tool: scripts/processing/verl_data_processing.py
    - Usage examples: scripts/processing/run_verl_examples.sh
    - Documentation: scripts/README.md (processing section)

This example demonstrates how to process VERL-formatted data for RLHF training:
1. Read VERL parquet data (with fields: data_source, prompt, ability, reward_model, extra_info)
2. Generate multiple responses per prompt using InferenceRunner
3. Score each response against ground truth using VERL's reward_score utilities
   - Supports math datasets (GSM8K, MATH, etc.) via math-verify
   - Supports code execution (codecontests, apps, etc.) via sandbox_fusion
   - Supports QA datasets (SearchR1, etc.) via exact match
4. Calculate statistics (avg_score, success_rate, etc.)
5. Save results back to parquet format

Key Features (also in CLI version):
- Multi-response generation (configurable N per prompt)
- Append behavior: Re-processing adds responses instead of replacing
- Automatic checkpointing for resumption
- Parallel scoring with rate limiting
- Explicit PyArrow schema for field preservation (index, error fields)

VERL format reference: https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
"""

from typing import Any, AsyncGenerator

import pyarrow as pa

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import (
    GenerateFunction,
    InferenceConfig,
    InferenceError,
    InferenceResult,
    InferenceRunner,
)
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.writers import ParquetWriter
from datatrove.utils.reward_score import compute_score


# ==============================================================================
# Configuration
# ==============================================================================
INPUT_DATA_PATH = "path/to/verl/data"  # Path to VERL parquet files
OUTPUT_PATH = "output/verl_processed"  # Output directory
CHECKPOINTS_PATH = "checkpoints/verl"  # Checkpoint directory for resuming
LOGS_PATH = "logs/verl_processing"  # Logging directory
STATS_PATH = "stats/verl"  # Statistics output (optional)

# Inference settings
MODEL_NAME = "meta-llama/Llama-3-8B"
N_RESPONSES_PER_PROMPT = 10  # Number of responses to generate per prompt
TEMPERATURE = 0.7  # Sampling temperature for diversity
MAX_TOKENS = 2048  # Maximum tokens per response

# Reward scoring settings
SANDBOX_FUSION_URL = None  # Set to your sandbox URL for code execution scoring
# Example: SANDBOX_FUSION_URL = "http://your-sandbox-server.com:5000"


# ==============================================================================
# PyArrow Schema Definition for VERL Format
# ==============================================================================
# Explicit schema prevents PyArrow from dropping fields during automatic schema inference.
# This is critical for preserving error fields (inference_error, score_error) which may
# be empty strings in some rows and actual error messages in others.
VERL_SCHEMA = pa.schema([
    ('data_source', pa.string()),
    ('prompt', pa.list_(pa.struct([
        ('role', pa.string()),
        ('content', pa.string())
    ]))),
    ('ability', pa.string()),
    ('reward_model', pa.struct([
        ('style', pa.string()),
        ('ground_truth', pa.string())
    ])),
    ('extra_info', pa.struct([
        # Original VERL metadata field (preserved from input)
        ('index', pa.int64()),
        # Unified responses list with all inference + scoring fields
        ('responses', pa.list_(pa.struct([
            # Inference result fields
            ('text', pa.string()),
            ('finish_reason', pa.string()),
            ('usage', pa.struct([
                ('prompt_tokens', pa.int64()),
                ('completion_tokens', pa.int64()),
                ('total_tokens', pa.int64())
            ])),
            ('inference_error', pa.string()),  # CRITICAL: Must be explicitly defined
            ('is_success', pa.bool_()),
            # Scoring result fields
            ('score', pa.float64()),
            ('score_error', pa.string()),  # CRITICAL: Must be explicitly defined
            ('reward_think', pa.float64()),
            ('reward_fmt', pa.float64()),
            ('reward_correct', pa.float64()),
            ('reward_length', pa.float64())
        ]))),
        # Aggregate statistics
        ('avg_score', pa.float64()),
        ('max_score', pa.float64()),
        ('min_score', pa.float64()),
        ('success_rate', pa.float64()),
        ('num_correct', pa.int64()),
        ('num_responses', pa.int64()),
        ('num_failed', pa.int64())
    ]))
])


# ==============================================================================
# 1. VERL Data Adapter - Convert VERL format to Document
# ==============================================================================
def verl_to_document_adapter(
    self, data: dict, path: str, id_in_file: int | str
) -> dict:
    """
    Convert VERL parquet row to Document format.

    VERL format contains:
        - data_source: Dataset name
        - prompt: List of chat messages (huggingface chat template format)
        - ability: Task category (e.g., "math")
        - reward_model: Dict with "style" and "ground_truth" keys
        - extra_info: Additional metadata

    Args:
        self: Reader instance (provides access to self.text_key, self.id_key)
        data: VERL data row as dictionary
        path: Source file path
        id_in_file: Row index in file

    Returns:
        Dictionary with "text", "id", and "metadata" keys for Document creation
    """
    import json
    import pickle

    # Normalize ground_truth to JSON string format for idempotency
    # This ensures output format matches input format (JSON string)
    reward_model = data["reward_model"].copy()
    if "ground_truth" in reward_model:
        ground_truth = reward_model["ground_truth"]

        if isinstance(ground_truth, bytes):
            # Legacy pickle format → unpickle and convert to JSON string
            try:
                gt_data = pickle.loads(ground_truth)
                # Convert sets to lists for JSON compatibility
                if isinstance(gt_data, dict) and "answer" in gt_data:
                    answer = gt_data["answer"]
                    for key in ["input_port_width", "output_port_width",
                               "clock_port_polarity", "reset_port_polarity_sync"]:
                        if key in answer and isinstance(answer[key], set):
                            # Convert set of tuples to list of lists
                            answer[key] = [list(item) if isinstance(item, tuple) else item
                                          for item in answer[key]]
                reward_model["ground_truth"] = json.dumps(gt_data)
            except Exception as e:
                # If unpickling fails, keep as-is and log warning
                print(f"Warning: Failed to unpickle ground_truth: {e}")
        # else: Already a string (JSON or plain text), keep as-is

    return {
        "text": json.dumps(data["prompt"]),  # Serialize prompt for text field
        "id": f"{path}_{id_in_file}",
        "metadata": {
            "data_source": data["data_source"],
            "ability": data["ability"],
            "reward_model": reward_model,
            "extra_info": data.get("extra_info", {}),
            "original_prompt": data["prompt"],  # Keep for inference
        },
    }


# ==============================================================================
# 2. Unified Rollout Function - Generate N responses, score, and aggregate
# ==============================================================================
def normalize_usage(usage: dict | None) -> dict:
    """
    Normalize token usage to a consistent schema for Parquet compatibility.

    InferenceResult.usage may have varying keys depending on the model/server.
    This ensures all usage dicts have the same structure.

    Args:
        usage: Token usage dict from inference result, or None for failures

    Returns:
        Normalized dict with consistent schema:
        {
            "prompt_tokens": int,      # Always present (0 if missing)
            "completion_tokens": int,  # Always present (0 if missing)
            "total_tokens": int        # Always present (0 if missing)
        }
    """
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


# Note: The normalize_score() function has been removed as of the latest update.
# compute_score() now automatically returns normalized scores with consistent schema:
# {
#     "score": float,
#     "error": str,  # Empty string "" on success, error message on failure
#     "reward_think": float,  # 0.0 if not applicable/not computed
#     "reward_fmt": float,    # 0.0 if not applicable/not computed
#     "reward_correct": float,  # 0.0 if not applicable/not computed
#     "reward_length": float    # 0.0 if not applicable/not computed
# }
# All fields are guaranteed to be present, preventing Parquet schema mismatch errors.
# NOTE: error uses empty string "" (not None), reward_* use 0.0 (not None), for Parquet compatibility.


def reconstruct_inference_result(
    result: dict | InferenceResult | InferenceError,
) -> InferenceResult | InferenceError:
    """
    Reconstruct InferenceResult/InferenceError objects from dictionaries.

    When documents are loaded from checkpoints, inference_results are
    deserialized as plain dictionaries. This function converts them back
    to proper dataclass instances for isinstance() checks to work.

    Args:
        result: Either a dictionary (from checkpoint) or already an object

    Returns:
        InferenceResult or InferenceError object
    """
    # Already an object, return as-is
    if isinstance(result, (InferenceResult, InferenceError)):
        return result

    # Dictionary from checkpoint - reconstruct based on fields
    if "error" in result:
        return InferenceError(error=result["error"])
    else:
        return InferenceResult(
            text=result.get("text", ""),
            finish_reason=result.get("finish_reason", ""),
            usage=result.get("usage", {}),
        )


async def verl_rollout_fn(
    document: Document,
    generate: GenerateFunction,
    scoring_semaphore,
    sandbox_url: str | None,
) -> dict:
    """
    Unified rollout function: generate N responses, score, and compute statistics.

    This replaces the old query_builder + postprocess_fn pattern with a single
    function that handles the entire workflow for VERL data processing.

    This function:
    1. Generates N responses by calling generate() multiple times
    2. Scores ALL responses in PARALLEL against ground truth using VERL's compute_score
       - Automatically selects appropriate scorer based on data_source
       - Supports math (GSM8K, MATH), code execution, geometry, and QA datasets
       - Uses asyncio.gather() for concurrent execution
       - Uses scoring_semaphore to prevent overwhelming external services
    3. Creates unified response objects merging inference results + scores
       - Each response contains: text, finish_reason, usage, inference_error, is_success,
         score, score_error, reward_think, reward_fmt, reward_correct, reward_length
       - Uses empty strings ("") instead of None for schema consistency (Parquet compatible)
    4. Computes aggregate statistics (avg, max, success rate, etc.)
    5. Stores unified_responses in document.metadata for checkpoints and output

    Args:
        document: Input document with VERL data in metadata
        generate: Callback function to send requests to the inference server
        scoring_semaphore: Semaphore to rate-limit concurrent scoring requests
        sandbox_url: Sandbox Fusion URL for code execution scoring (optional)

    Returns:
        Summary dict with processing status (stored in metadata_key by framework)
    """
    import asyncio

    # 1. Extract VERL metadata
    original_prompt = document.metadata["original_prompt"]
    ground_truth = document.metadata["reward_model"].get("ground_truth", "")
    data_source = document.metadata["data_source"]

    # Read existing responses from previous processing runs (for append behavior)
    existing_responses = document.metadata.get("extra_info", {}).get("responses", [])

    # 2. Generate N responses by calling generate() multiple times
    results = []
    for i in range(N_RESPONSES_PER_PROMPT):
        try:
            result = await generate({
                "messages": original_prompt,  # Chat messages in VERL format
                "max_tokens": MAX_TOKENS,
                # temperature is in default_generation_params, no need to pass here
            })
            results.append(result)
        except InferenceError as e:
            # Store errors alongside successful results
            results.append(e)

    # Handle different ground truth formats based on dataset type
    # Ground truth is now always a JSON string (converted by input adapter)
    # SearchR1 datasets expect dict format: {"target": [answers]}
    if data_source.startswith("searchR1_") and isinstance(ground_truth, str):
        ground_truth = {"target": [ground_truth]}

    # 3. Score all responses in parallel using asyncio.gather()
    # Note: compute_score() now returns normalized scores with consistent schema automatically
    async def score_single_response(result):
        """Score a single response with semaphore rate limiting."""
        if isinstance(result, InferenceResult):
            try:
                # Use semaphore to limit concurrent scoring requests
                async with scoring_semaphore:
                    # Run synchronous compute_score in thread pool to avoid blocking event loop
                    score_dict = await asyncio.to_thread(
                        compute_score,
                        data_source,
                        result.text,
                        ground_truth,
                        sandbox_fusion_url=sandbox_url,
                    )
                return score_dict
            except Exception as e:
                # Handle scoring errors - return normalized error score
                return {
                    "score": 0.0,
                    "error": f"Scoring error: {str(e)}",
                    "reward_think": 0.0,
                    "reward_fmt": 0.0,
                    "reward_correct": 0.0,
                    "reward_length": 0.0,
                }
        else:
            # Failed inference gets zero score with normalized schema
            error_msg = result.error if hasattr(result, "error") else "unknown"
            return {
                "score": 0.0,
                "error": f"Inference error: {error_msg}",
                "reward_think": 0.0,
                "reward_fmt": 0.0,
                "reward_correct": 0.0,
                "reward_length": 0.0,
            }

    # Score all responses concurrently
    scores = await asyncio.gather(*[
        score_single_response(result)
        for result in results
    ])

    # 4. Create unified response objects (merge inference results + scores)
    # This unified structure is stored in checkpoints and used for output
    unified_responses = []
    for result, score in zip(results, scores):
        if isinstance(result, InferenceResult):
            unified_responses.append(
                {
                    # Response fields
                    "text": result.text,
                    "finish_reason": result.finish_reason,
                    "usage": normalize_usage(result.usage),
                    "inference_error": "",  # Empty string for success (schema consistent)
                    "is_success": True,
                    # Score fields
                    "score": score.get("score", 0.0),
                    "score_error": score.get("error", ""),
                    "reward_think": score.get("reward_think", 0.0),
                    "reward_fmt": score.get("reward_fmt", 0.0),
                    "reward_correct": score.get("reward_correct", 0.0),
                    "reward_length": score.get("reward_length", 0.0),
                }
            )
        else:
            # Failed inference
            error_msg = result.error if hasattr(result, "error") else "unknown"
            unified_responses.append(
                {
                    # Response fields (use empty strings for None to maintain schema)
                    "text": "",  # Empty string instead of None (schema consistent)
                    "finish_reason": "error",
                    "usage": normalize_usage(None),
                    "inference_error": error_msg,  # Inference error message
                    "is_success": False,
                    # Score fields (0.0 for failed inference)
                    "score": score.get("score", 0.0),
                    "score_error": score.get("error", ""),
                    "reward_think": score.get("reward_think", 0.0),
                    "reward_fmt": score.get("reward_fmt", 0.0),
                    "reward_correct": score.get("reward_correct", 0.0),
                    "reward_length": score.get("reward_length", 0.0),
                }
            )

    # 5. Merge existing responses (from previous runs) with new responses (append behavior)
    # This allows incremental response generation: run 1 generates 10, run 2 adds 10 more = 20 total
    all_responses = existing_responses + unified_responses

    # 6. Compute aggregate statistics from ALL responses (existing + new)
    if all_responses:
        valid_scores = [r["score"] for r in all_responses]
        # Store ALL responses (used by both checkpoints and output adapter)
        document.metadata["unified_responses"] = all_responses
        # Recalculate statistics from ALL responses (not just new ones)
        document.metadata["avg_score"] = sum(valid_scores) / len(valid_scores)
        document.metadata["max_score"] = max(valid_scores)
        document.metadata["min_score"] = min(valid_scores)
        document.metadata["num_correct"] = sum(
            int(r["score"] > 0) for r in all_responses
        )
        document.metadata["success_rate"] = document.metadata["num_correct"] / len(
            all_responses
        )
        document.metadata["num_responses"] = len(all_responses)
        document.metadata["num_failed"] = sum(
            1 for r in all_responses if r["score_error"]
        )
    else:
        # No responses generated
        document.metadata["unified_responses"] = []
        document.metadata["avg_score"] = 0.0
        document.metadata["max_score"] = 0.0
        document.metadata["min_score"] = 0.0
        document.metadata["num_correct"] = 0
        document.metadata["success_rate"] = 0.0
        document.metadata["num_responses"] = 0
        document.metadata["num_failed"] = 0

    # 7. Return summary dict (will be stored in doc.metadata[metadata_key])
    # The actual VERL data is already stored in document.metadata by this function
    return {
        "status": "processed",
        "num_new_responses": len(unified_responses),
        "num_total_responses": len(all_responses),
        "avg_score": document.metadata["avg_score"],
    }


# ==============================================================================
# 4. Custom Stats Block - Collect statistics across dataset (optional)
# ==============================================================================
class ResponseScoreStats(BaseStats):
    """
    Collect statistics on response scores across the dataset.

    This block computes statistics for:
    - Average score per document
    - Success rate (% of correct responses)
    - Number of responses generated
    - Distribution of scores (via histogram grouping)

    Results are saved in: {output_folder}/{group}/{stat_name}/{rank}.json
    """

    name = "Response Score Statistics"

    def extract_stats(self, doc: Document) -> dict[str, int | float]:
        """
        Extract statistics from a single document.

        Args:
            doc: Document with response score metadata

        Returns:
            Dictionary of statistics to aggregate
        """
        return {
            "avg_score": doc.metadata.get("avg_score", 0.0),
            "max_score": doc.metadata.get("max_score", 0.0),
            "min_score": doc.metadata.get("min_score", 0.0),
            "success_rate": doc.metadata.get("success_rate", 0.0),
            "num_responses": doc.metadata.get("num_responses", 0),
            "num_correct": doc.metadata.get("num_correct", 0),
            "num_failed": doc.metadata.get("num_failed", 0),
        }


# ==============================================================================
# 5. Output Adapter - Convert Document back to VERL format + results
# ==============================================================================
def document_to_verl_adapter(self, document: Document) -> dict:
    """
    Convert processed Document back to VERL parquet format with results in extra_info.

    Output follows VERL standard (5 required fields) with all generation results
    and metrics stored in the extra_info field:
    - responses: List of unified response objects (Parquet compatible)
        Each response merges inference result + score into a single object:
        {
            # Response fields
            "text": str,               # "" for failures (not None - schema consistent)
            "finish_reason": str,
            "usage": dict,
            "inference_error": str,    # "" for success (not None - schema consistent)
            "is_success": bool,

            # Score fields
            "score": float,
            "score_error": str,        # "" for success
            "reward_think": float,
            "reward_fmt": float,
            "reward_correct": float,
            "reward_length": float
        }
    - avg_score, max_score, min_score: Aggregate statistics
    - success_rate, num_correct, num_responses, num_failed: Success metrics

    Args:
        document: Processed document with inference results

    Returns:
        Dictionary for parquet row with VERL standard fields only
    """
    # Check if unified_responses already exists (from postprocess_and_score)
    unified_responses = document.metadata.get("unified_responses", None)

    # If not present, create from separate lists (backward compatibility with old checkpoints)
    if unified_responses is None:
        # Extract inference results and scores
        inference_results = document.metadata.get("inference_results", [])
        # Reconstruct objects from checkpoint dictionaries
        inference_results = [
            reconstruct_inference_result(r) for r in inference_results
        ]
        response_scores = document.metadata.get("response_scores", [])

        # Validate that inference results and scores are aligned
        if len(inference_results) != len(response_scores):
            raise ValueError(
                f"Length mismatch: inference_results ({len(inference_results)}) "
                f"!= response_scores ({len(response_scores)}). "
                f"Lists must be synchronized."
            )

        # Create unified response objects (merge inference result + score)
        # All responses have the same fields regardless of success/failure
        unified_responses = []
        for result, score in zip(inference_results, response_scores):
            if isinstance(result, InferenceResult):
                unified_responses.append(
                    {
                        # Response fields
                        "text": result.text,
                        "finish_reason": result.finish_reason,
                        "usage": normalize_usage(result.usage),
                        "inference_error": "",  # Empty string for success (schema consistent)
                        "is_success": True,
                        # Score fields
                        "score": score.get("score", 0.0),
                        "score_error": score.get("error", ""),
                        "reward_think": score.get("reward_think", 0.0),
                        "reward_fmt": score.get("reward_fmt", 0.0),
                        "reward_correct": score.get("reward_correct", 0.0),
                        "reward_length": score.get("reward_length", 0.0),
                    }
                )
            else:
                unified_responses.append(
                    {
                        # Response fields (use empty strings for schema consistency)
                        "text": "",  # Empty string instead of None (schema consistent)
                        "finish_reason": "error",
                        "usage": normalize_usage(None),
                        "inference_error": result.error
                        if hasattr(result, "error")
                        else "unknown",
                        "is_success": False,
                        # Score fields (0.0 for failed inference)
                        "score": score.get("score", 0.0),
                        "score_error": score.get("error", ""),
                        "reward_think": score.get("reward_think", 0.0),
                        "reward_fmt": score.get("reward_fmt", 0.0),
                        "reward_correct": score.get("reward_correct", 0.0),
                        "reward_length": score.get("reward_length", 0.0),
                    }
                )

    # Copy existing extra_info and add generation results
    extra_info = document.metadata.get("extra_info", {}).copy()
    extra_info.update(
        {
            # Unified responses (single list with both inference + score data)
            "responses": unified_responses,
            # Aggregate statistics
            "avg_score": document.metadata.get("avg_score", 0.0),
            "max_score": document.metadata.get("max_score", 0.0),
            "min_score": document.metadata.get("min_score", 0.0),
            "success_rate": document.metadata.get("success_rate", 0.0),
            "num_correct": document.metadata.get("num_correct", 0),
            "num_responses": document.metadata.get("num_responses", 0),
            "num_failed": document.metadata.get("num_failed", 0),
        }
    )

    # Keep ground_truth in JSON string format (idempotent)
    # Input adapter already normalized it to JSON string format
    reward_model = document.metadata["reward_model"].copy()

    return {
        # VERL standard fields (5 required fields only)
        "data_source": document.metadata["data_source"],
        "prompt": document.metadata["original_prompt"],
        "ability": document.metadata["ability"],
        "reward_model": reward_model,  # JSON string format (same as input)
        "extra_info": extra_info,  # All results stored here
    }


# ==============================================================================
# 6. Pipeline Construction
# ==============================================================================
import asyncio

# Create semaphore for scoring rate limiting (replaces old max_concurrent_scoring parameter)
SCORING_SEMAPHORE = asyncio.Semaphore(50)  # Adjust based on sandbox capacity

pipeline = [
    # Step 1: Read VERL parquet data
    ParquetReader(
        data_folder=INPUT_DATA_PATH,
        adapter=verl_to_document_adapter,
        batch_size=100,  # Read 100 rows at a time
        recursive=True,
        glob_pattern="*.parquet",
    ),
    # Step 2: Generate multiple responses and score them
    InferenceRunner(
        rollout_fn=verl_rollout_fn,  # Unified function (replaces query_builder + postprocess_fn)
        config=InferenceConfig(
            server_type="vllm",  # Options: "vllm", "sglang", "endpoint"
            model_name_or_path=MODEL_NAME,
            default_generation_params={"temperature": TEMPERATURE},  # Dict format (was direct param)
            max_concurrent_generations=100,  # Renamed from max_concurrent_requests
            max_concurrent_documents=200,  # Renamed from max_concurrent_tasks
            metric_interval=120,  # Report metrics every 2 minutes
        ),
        output_writer=ParquetWriter(
            output_folder=OUTPUT_PATH,
            adapter=document_to_verl_adapter,
            output_filename="${rank}_chunk_${chunk_index}.parquet",
            compression="snappy",
            schema=VERL_SCHEMA,  # Explicit schema preserves error fields
        ),
        shared_context={  # NEW: Pass dependencies as kwargs to rollout_fn
            "scoring_semaphore": SCORING_SEMAPHORE,
            "sandbox_url": SANDBOX_FUSION_URL,
        },
        checkpoints_local_dir=CHECKPOINTS_PATH,  # Enable checkpointing
        records_per_chunk=500,  # Save checkpoint every 500 documents
        # metadata_key default is "rollout_results" (not used by VERL output adapter)
    ),
    # Step 3 (Optional): Collect statistics
    # Uncomment to enable statistics collection
    # ResponseScoreStats(
    #     output_folder=STATS_PATH,
    #     groups_to_compute=["summary", "histogram"],  # Aggregate + distribution
    # ),
]


# ==============================================================================
# 7. Executor Setup and Execution
# ==============================================================================
if __name__ == "__main__":
    # Local execution with multiprocessing
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=LOGS_PATH,
        tasks=10,  # Number of parallel tasks
        workers=5,  # Number of concurrent workers
    )

    executor.run()

    # For distributed execution on Slurm:
    # from datatrove.executor import SlurmPipelineExecutor
    # executor = SlurmPipelineExecutor(
    #     pipeline=pipeline,
    #     logging_dir=LOGS_PATH,
    #     tasks=100,
    #     time="24:00:00",
    #     partition="gpu",
    #     job_name="verl_processing",
    #     sbatch_args={"gres": "gpu:1"},  # Request 1 GPU per task
    # )
    # executor.run()
