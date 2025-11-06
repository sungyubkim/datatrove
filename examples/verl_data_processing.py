"""
VERL data processing pipeline example.

This example demonstrates how to process VERL-formatted data for RLHF training:
1. Read VERL parquet data (with fields: data_source, prompt, ability, reward_model, extra_info)
2. Generate multiple responses per prompt using InferenceRunner
3. Score each response against ground truth using VERL's reward_score utilities
   - Supports math datasets (GSM8K, MATH, etc.) via math-verify
   - Supports code execution (codecontests, apps, etc.) via sandbox_fusion
   - Supports QA datasets (SearchR1, etc.) via exact match
4. Calculate statistics (avg_score, success_rate, etc.)
5. Save results back to parquet format

VERL format reference: https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
"""

from typing import Any, AsyncGenerator

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import (
    InferenceConfig,
    InferenceError,
    InferenceRunner,
    InferenceSuccess,
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
    import base64
    import json

    # Convert bytes ground_truth to base64 for checkpoint JSON serialization
    # Note: New JSON format datasets (e.g., codev) already have string ground_truth, no conversion needed
    reward_model = data["reward_model"].copy()
    if "ground_truth" in reward_model and isinstance(reward_model["ground_truth"], bytes):
        # Legacy format: pickle bytes → base64 string for JSON compatibility
        reward_model["ground_truth"] = base64.b64encode(reward_model["ground_truth"]).decode('ascii')

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
# 2. Multi-Response Query Builder - Generate N responses per prompt
# ==============================================================================
async def multi_response_query_builder(
    runner: InferenceRunner, document: Document
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Generate multiple inference requests for a single document.

    This async generator yields N inference requests for each document,
    enabling the generation of multiple diverse responses per prompt.

    Args:
        runner: InferenceRunner instance
        document: Input document with VERL data in metadata

    Yields:
        Inference request payloads (OpenAI chat completion format)
    """
    # Extract original prompt from metadata
    original_prompt = document.metadata["original_prompt"]

    # Generate N requests for diverse responses
    for i in range(N_RESPONSES_PER_PROMPT):
        yield {
            "messages": original_prompt,  # Chat messages in VERL format
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            # Optional: add response index to track which response is which
            "metadata": {"response_index": i},
        }


# ==============================================================================
# 3. Postprocessing - Score all responses and compute statistics
# ==============================================================================
def normalize_usage(usage: dict | None) -> dict:
    """
    Normalize token usage to a consistent schema for Parquet compatibility.

    InferenceSuccess.usage may have varying keys depending on the model/server.
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


def normalize_score(
    score_result: dict, is_success: bool = True, error_msg: str = None
) -> dict:
    """
    Normalize score results to a consistent schema for Parquet compatibility.

    compute_score now always returns a dict with score, reward_think, and reward_fmt fields.
    This function adds the error field for consistent Parquet schema.

    Args:
        score_result: Score dict from compute_score (with score, reward_think, reward_fmt)
        is_success: Whether inference succeeded
        error_msg: Error message if inference failed

    Returns:
        Normalized dict with consistent schema:
        {
            "score": float,               # Always present
            "error": str | None,          # None for success, error message for failure
            "reward_think": float | None, # For math datasets, None/1.0 otherwise
            "reward_fmt": float | None    # For math datasets, None/1.0 otherwise
        }
    """
    if is_success:
        return {
            "score": score_result.get("score", 0.0),
            "error": None,
            "reward_think": score_result.get("reward_think", None),
            "reward_fmt": score_result.get("reward_fmt", None),
        }
    else:
        # Failure case
        return {
            "score": 0.0,
            "error": error_msg if error_msg else "unknown",
            "reward_think": None,
            "reward_fmt": None,
        }


def reconstruct_inference_result(
    result: dict | InferenceSuccess | InferenceError,
) -> InferenceSuccess | InferenceError:
    """
    Reconstruct InferenceSuccess/InferenceError objects from dictionaries.

    When documents are loaded from checkpoints, inference_results are
    deserialized as plain dictionaries. This function converts them back
    to proper dataclass instances for isinstance() checks to work.

    Args:
        result: Either a dictionary (from checkpoint) or already an object

    Returns:
        InferenceSuccess or InferenceError object
    """
    # Already an object, return as-is
    if isinstance(result, (InferenceSuccess, InferenceError)):
        return result

    # Dictionary from checkpoint - reconstruct based on fields
    if "error" in result:
        return InferenceError(error=result["error"])
    else:
        return InferenceSuccess(
            text=result.get("text", ""),
            finish_reason=result.get("finish_reason", ""),
            usage=result.get("usage", {}),
        )


def postprocess_and_score(runner: InferenceRunner, document: Document) -> Document:
    """
    Post-process document after inference: score responses and compute statistics.

    This function:
    1. Retrieves all generated responses from inference_results
    2. Scores each response against ground truth using VERL's compute_score
       - Automatically selects appropriate scorer based on data_source
       - Supports math (GSM8K, MATH), code execution, geometry, and QA datasets
    3. Computes aggregate statistics (avg, max, success rate, etc.)
    4. Adds results to document metadata

    Args:
        runner: InferenceRunner instance
        document: Document with inference results

    Returns:
        Updated document with scoring results (or None to skip saving)
    """
    import base64

    inference_results = document.metadata.get("inference_results", [])
    # Reconstruct objects from checkpoint dictionaries
    inference_results = [reconstruct_inference_result(r) for r in inference_results]
    ground_truth = document.metadata["reward_model"].get("ground_truth", "")
    data_source = document.metadata["data_source"]

    # Handle different ground truth formats:
    # - Legacy pickle format: base64 string → decode to bytes
    # - New JSON format: plain string → keep as-is (e.g., codev JSON)
    # - Math/QA datasets: plain string → keep as-is
    if isinstance(ground_truth, str):
        try:
            ground_truth = base64.b64decode(ground_truth)  # Try decoding base64
        except Exception:
            pass  # Not base64, use as string (JSON or plain text)

    # Handle different ground truth formats based on dataset type
    # SearchR1 datasets expect dict format: {"target": [answers]}
    if data_source.startswith("searchR1_") and isinstance(ground_truth, str):
        ground_truth = {"target": [ground_truth]}

    # Score each response
    scores = []
    for result in inference_results:
        if isinstance(result, InferenceSuccess):
            try:
                score_dict = compute_score(
                    data_source,
                    result.text,
                    ground_truth,
                    sandbox_fusion_url=SANDBOX_FUSION_URL,
                )
                # Normalize score to consistent schema for Parquet compatibility
                normalized_score = normalize_score(score_dict, is_success=True)
            except Exception as e:
                # Handle scoring errors (e.g., invalid format, API errors)
                normalized_score = normalize_score(
                    None, is_success=False, error_msg=f"Scoring error: {str(e)}"
                )
            scores.append(normalized_score)
        else:
            # Failed inference gets zero score with normalized schema
            error_msg = result.error if hasattr(result, "error") else "unknown"
            normalized_score = normalize_score(
                None, is_success=False, error_msg=error_msg
            )
            scores.append(normalized_score)

    # Compute aggregate statistics
    if scores:
        valid_scores = [s["score"] for s in scores]
        document.metadata["response_scores"] = scores
        document.metadata["avg_score"] = sum(valid_scores) / len(valid_scores)
        document.metadata["max_score"] = max(valid_scores)
        document.metadata["min_score"] = min(valid_scores)
        document.metadata["num_correct"] = sum(int(s["score"] > 0) for s in scores)
        document.metadata["success_rate"] = document.metadata["num_correct"] / len(
            scores
        )
        document.metadata["num_responses"] = len(scores)
        document.metadata["num_failed"] = sum(1 for s in scores if "error" in s)
    else:
        # No responses generated
        document.metadata["response_scores"] = []
        document.metadata["avg_score"] = 0.0
        document.metadata["max_score"] = 0.0
        document.metadata["min_score"] = 0.0
        document.metadata["num_correct"] = 0
        document.metadata["success_rate"] = 0.0
        document.metadata["num_responses"] = 0
        document.metadata["num_failed"] = 0

    return document


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
def document_to_verl_adapter(document: Document) -> dict:
    """
    Convert processed Document back to VERL parquet format with results in extra_info.

    Output follows VERL standard (5 required fields) with all generation results
    and metrics stored in the extra_info field:
    - generated_responses: List of responses with unified schema (Parquet compatible)
        Each response has: text, finish_reason, usage, error, is_success
        Success: text/finish_reason/usage filled, error=None, is_success=True
        Failure: text/usage=None, finish_reason="error", error filled, is_success=False
    - response_scores: Scores for each response
    - avg_score, max_score, min_score: Aggregate statistics
    - success_rate, num_correct, num_responses, num_failed: Success metrics

    Args:
        document: Processed document with inference results

    Returns:
        Dictionary for parquet row with VERL standard fields only
    """
    import base64

    # Extract inference results
    inference_results = document.metadata.get("inference_results", [])
    # Reconstruct objects from checkpoint dictionaries
    inference_results = [reconstruct_inference_result(r) for r in inference_results]

    # Format responses for output with unified schema (for Parquet compatibility)
    # All responses have the same fields regardless of success/failure
    generated_responses = []
    for result in inference_results:
        if isinstance(result, InferenceSuccess):
            generated_responses.append(
                {
                    "text": result.text,
                    "finish_reason": result.finish_reason,
                    "usage": normalize_usage(result.usage),
                    "error": None,
                    "is_success": True,
                }
            )
        else:
            generated_responses.append(
                {
                    "text": None,
                    "finish_reason": "error",
                    "usage": normalize_usage(None),
                    "error": result.error if hasattr(result, "error") else "unknown",
                    "is_success": False,
                }
            )

    # Copy existing extra_info and add generation results
    extra_info = document.metadata.get("extra_info", {}).copy()
    extra_info.update(
        {
            # Generated responses and scores
            "generated_responses": generated_responses,
            "response_scores": document.metadata.get("response_scores", []),
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

    # Handle output ground_truth format:
    # - Legacy format: base64 string → decode back to bytes for parquet
    # - New JSON format: keep as string (no base64 encoding)
    reward_model = document.metadata["reward_model"].copy()
    if "ground_truth" in reward_model and isinstance(reward_model["ground_truth"], str):
        try:
            reward_model["ground_truth"] = base64.b64decode(reward_model["ground_truth"])
        except Exception:
            pass  # Not base64 (e.g., JSON string), keep as-is

    return {
        # VERL standard fields (5 required fields only)
        "data_source": document.metadata["data_source"],
        "prompt": document.metadata["original_prompt"],
        "ability": document.metadata["ability"],
        "reward_model": reward_model,  # Use decoded version with original bytes format
        "extra_info": extra_info,  # All results stored here
    }


# ==============================================================================
# 6. Pipeline Construction
# ==============================================================================
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
        query_builder=multi_response_query_builder,
        config=InferenceConfig(
            server_type="vllm",  # Options: "vllm", "sglang", "vllm-remote"
            model_name_or_path=MODEL_NAME,
            temperature=TEMPERATURE,  # Will be overridden by query_builder
            max_concurrent_requests=100,  # Adjust based on GPU memory
            max_concurrent_tasks=200,  # Higher if query_builder is slow
            metric_interval=120,  # Report metrics every 2 minutes
        ),
        output_writer=ParquetWriter(
            output_folder=OUTPUT_PATH,
            adapter=document_to_verl_adapter,
            output_filename="${rank}_chunk_${chunk_index}.parquet",
            compression="snappy",
        ),
        checkpoints_local_dir=CHECKPOINTS_PATH,  # Enable checkpointing
        records_per_chunk=500,  # Save checkpoint every 500 documents
        postprocess_fn=postprocess_and_score,  # Score responses
        skip_bad_requests=True,  # Skip documents that cause BadRequestError
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
