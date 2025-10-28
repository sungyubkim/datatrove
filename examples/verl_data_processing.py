"""
VERL data processing pipeline example.

This example demonstrates how to process VERL-formatted data for RLHF training:
1. Read VERL parquet data (with fields: data_source, prompt, ability, reward_model, extra_info)
2. Generate multiple responses per prompt using InferenceRunner
3. Score each response against ground truth
4. Calculate statistics (avg_score, success_rate, etc.)
5. Save results back to parquet format

VERL format reference: https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
"""

from typing import Any, AsyncGenerator

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner, InferenceSuccess
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.stats.base import BaseStats
from datatrove.pipeline.writers import ParquetWriter


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


# ==============================================================================
# 1. VERL Data Adapter - Convert VERL format to Document
# ==============================================================================
def verl_to_document_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
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

    return {
        "text": json.dumps(data["prompt"]),  # Serialize prompt for text field
        "id": f"{path}_{id_in_file}",
        "metadata": {
            "data_source": data["data_source"],
            "ability": data["ability"],
            "reward_model": data["reward_model"],
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
# 3. Response Scoring - Evaluate generated responses
# ==============================================================================
def score_response(ground_truth: str, generated_text: str, reward_style: str) -> dict:
    """
    Score a generated response against ground truth.

    This is a simple example implementation. In practice, you might:
    - Use a reward model for scoring
    - Implement domain-specific evaluation metrics
    - Use exact match, fuzzy matching, or semantic similarity

    Args:
        ground_truth: Expected answer
        generated_text: Model-generated response
        reward_style: Scoring style from VERL reward_model field

    Returns:
        Dictionary with scoring metrics
    """
    # Example: Simple exact match scoring
    # TODO: Replace with your custom scoring logic
    generated_clean = generated_text.strip().lower()
    ground_truth_clean = ground_truth.strip().lower()

    # Exact match
    exact_match = generated_clean == ground_truth_clean

    # Containment match
    contains_answer = ground_truth_clean in generated_clean

    # Compute score (customize based on your needs)
    if exact_match:
        score = 1.0
    elif contains_answer:
        score = 0.7
    else:
        score = 0.0

    return {
        "score": score,
        "exact_match": exact_match,
        "contains_answer": contains_answer,
        "response_length": len(generated_text),
        "is_correct": score > 0.5,
    }


# ==============================================================================
# 4. Postprocessing - Score all responses and compute statistics
# ==============================================================================
def postprocess_and_score(runner: InferenceRunner, document: Document) -> Document:
    """
    Post-process document after inference: score responses and compute statistics.

    This function:
    1. Retrieves all generated responses from inference_results
    2. Scores each response against ground truth
    3. Computes aggregate statistics (avg, max, success rate, etc.)
    4. Adds results to document metadata

    Args:
        runner: InferenceRunner instance
        document: Document with inference results

    Returns:
        Updated document with scoring results (or None to skip saving)
    """
    inference_results = document.metadata.get("inference_results", [])
    ground_truth = document.metadata["reward_model"].get("ground_truth", "")
    reward_style = document.metadata["reward_model"].get("style", "rule")

    # Score each response
    scores = []
    for result in inference_results:
        if isinstance(result, InferenceSuccess):
            score_dict = score_response(ground_truth, result.text, reward_style)
            scores.append(score_dict)
        else:
            # Failed inference gets zero score
            scores.append(
                {
                    "score": 0.0,
                    "exact_match": False,
                    "contains_answer": False,
                    "response_length": 0,
                    "is_correct": False,
                    "error": result.error if hasattr(result, "error") else "unknown",
                }
            )

    # Compute aggregate statistics
    if scores:
        valid_scores = [s["score"] for s in scores]
        document.metadata["response_scores"] = scores
        document.metadata["avg_score"] = sum(valid_scores) / len(valid_scores)
        document.metadata["max_score"] = max(valid_scores)
        document.metadata["min_score"] = min(valid_scores)
        document.metadata["num_correct"] = sum(s["is_correct"] for s in scores)
        document.metadata["success_rate"] = document.metadata["num_correct"] / len(scores)
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
# 5. Custom Stats Block - Collect statistics across dataset (optional)
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
# 6. Output Adapter - Convert Document back to VERL format + results
# ==============================================================================
def document_to_verl_adapter(document: Document) -> dict:
    """
    Convert processed Document back to VERL parquet format with results in extra_info.

    Output follows VERL standard (5 required fields) with all generation results
    and metrics stored in the extra_info field:
    - generated_responses: List of all generated responses
    - response_scores: Scores for each response
    - avg_score, max_score, min_score: Aggregate statistics
    - success_rate, num_correct, num_responses, num_failed: Success metrics

    Args:
        document: Processed document with inference results

    Returns:
        Dictionary for parquet row with VERL standard fields only
    """
    # Extract inference results
    inference_results = document.metadata.get("inference_results", [])

    # Format responses for output
    generated_responses = []
    for result in inference_results:
        if isinstance(result, InferenceSuccess):
            generated_responses.append(
                {
                    "text": result.text,
                    "finish_reason": result.finish_reason,
                    "usage": result.usage,
                }
            )
        else:
            generated_responses.append({"error": result.error if hasattr(result, "error") else "unknown"})

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

    return {
        # VERL standard fields (5 required fields only)
        "data_source": document.metadata["data_source"],
        "prompt": document.metadata["original_prompt"],
        "ability": document.metadata["ability"],
        "reward_model": document.metadata["reward_model"],
        "extra_info": extra_info,  # All results stored here
    }


# ==============================================================================
# 7. Pipeline Construction
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
            output_filename="${rank}.parquet",
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
# 8. Executor Setup and Execution
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
