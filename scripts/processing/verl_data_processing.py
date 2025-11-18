#!/usr/bin/env python3
"""
VERL Data Processing Pipeline - Production CLI Tool

Process VERL-formatted data for RLHF training with multi-response generation and reward scoring.

This tool provides a command-line interface to:
1. Read VERL parquet data (data_source, prompt, ability, reward_model, extra_info)
2. Generate multiple responses per prompt using vLLM/SGLang/remote inference servers
3. Score responses against ground truth using dataset-specific scorers
4. Calculate aggregate statistics (avg_score, success_rate, etc.)
5. Save results with automatic checkpointing for resumption

Supported datasets:
- Math: GSM8K, MATH, Numina datasets (via math-verify)
- Code: codecontests, apps (via sandbox_fusion) - multi-language support
- QA: SearchR1 datasets (exact match)
- ToolRL: Tool learning tasks (XML/GPT OSS formats)
- IFEval: Instruction-following benchmarks
- CodeV: Verilog code generation (requires iverilog + sandbox)
- Table reasoning: HiTab, WikiTableQuestions, TabFact, FeTaQA
- Logic: Ordering/zebra puzzles, ARC-AGI tasks
- And more...

Key Features:
- Append behavior: Re-processing adds responses instead of replacing
- Automatic checkpointing with configurable frequency
- Parallel inference and scoring with semaphore-based rate limiting
- Explicit PyArrow schema ensures field preservation (index, error fields)
- Comprehensive statistics and logging

Usage Examples:

    # Basic usage with default settings
    python scripts/processing/verl_data_processing.py \\
        --input-data data/math-verl/train.parquet \\
        --output-dir output/math-processed \\
        --model-name-or-path meta-llama/Llama-3-8B

    # Code dataset with Sandbox Fusion
    python scripts/processing/verl_data_processing.py \\
        --input-data data/codecontests.parquet \\
        --output-dir output/code-processed \\
        --model-name-or-path deepseek-ai/deepseek-coder-7b \\
        --num-responses-per-prompt 5 \\
        --sandbox-fusion-url http://localhost:5000 \\
        --max-concurrent-scoring 20

    # Remote vLLM server with custom settings
    python scripts/processing/verl_data_processing.py \\
        --input-data data/large-dataset \\
        --output-dir output/remote-processed \\
        --model-name-or-path meta-llama/Llama-3-70B \\
        --inference-server-type vllm-remote \\
        --remote-vllm-endpoint http://vllm-cluster:8000 \\
        --num-responses-per-prompt 15 \\
        --max-concurrent-inference 200

    # Production run with full control
    python scripts/processing/verl_data_processing.py \\
        --input-data data/production \\
        --output-dir output/prod \\
        --model-name-or-path Qwen/Qwen2.5-Math-7B \\
        --num-responses-per-prompt 20 \\
        --sampling-temperature 0.8 \\
        --max-tokens-per-response 4096 \\
        --num-parallel-tasks 50 \\
        --num-concurrent-workers 10 \\
        --checkpoint-frequency 1000 \\
        --checkpoint-dir checkpoints/prod \\
        --log-dir logs/prod \\
        --stats-output-dir stats/prod

For more examples, see: scripts/processing/run_verl_examples.sh
For implementation details, see: examples/verl_data_processing.py

VERL format reference: https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
"""

import argparse
from pathlib import Path
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
# Picklable Rollout Function for Multiprocessing
# ==============================================================================
# This callable class captures configuration at creation time and is picklable,
# solving the issue where global variables modified at runtime don't propagate
# to worker processes in multiprocessing.


class VERLRolloutFunction:
    """
    Picklable unified rollout function with captured configuration.

    This class replaces the old MultiResponseQueryBuilder + PostprocessAndScore
    pattern with a single callable that handles query generation, inference,
    and scoring in one unified workflow.
    """

    def __init__(self, num_responses: int, temperature: float, max_tokens: int, sandbox_url: str | None):
        """
        Initialize rollout function with configuration.

        Args:
            num_responses: Number of responses to generate per prompt
            temperature: Sampling temperature for diversity (stored for reference, passed via default_generation_params)
            max_tokens: Maximum tokens per response
            sandbox_url: Sandbox Fusion URL for code execution scoring
        """
        self.num_responses = num_responses
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sandbox_url = sandbox_url

    async def __call__(
        self,
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
            sandbox_url: Sandbox Fusion URL for code execution scoring (from shared_context)

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
        for i in range(self.num_responses):
            try:
                result = await generate({
                    "messages": original_prompt,  # Chat messages in VERL format
                    "max_tokens": self.max_tokens,
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
        else:
            # No responses generated
            document.metadata["unified_responses"] = []
            document.metadata["avg_score"] = 0.0
            document.metadata["max_score"] = 0.0
            document.metadata["min_score"] = 0.0
            document.metadata["num_correct"] = 0
            document.metadata["success_rate"] = 0.0
            document.metadata["num_responses"] = 0

        # 7. Return summary dict (will be stored in doc.metadata[metadata_key])
        # The actual VERL data is already stored in document.metadata by this function
        return {
            "status": "processed",
            "num_new_responses": len(unified_responses),
            "num_total_responses": len(all_responses),
            "avg_score": document.metadata["avg_score"],
        }


class ResponseScoreStats(BaseStats):
    """
    Collect statistics on response scores and success rates.

    This class is defined at module level (not inside build_pipeline) to ensure
    it can be properly pickled for multiprocessing. Local classes defined inside
    functions cannot be pickled by Python's pickle module.
    """

    def extract_stats(self, doc: Document):
        """
        Extract response score statistics from a document.

        Args:
            doc: Document with response score metadata

        Returns:
            Dict mapping statistic names to values
        """
        avg_score = doc.metadata.get("avg_score", 0.0)
        success_rate = doc.metadata.get("success_rate", 0.0)
        num_responses = doc.metadata.get("num_responses", 0)
        return {
            "avg_score": avg_score,
            "success_rate": success_rate,
            "num_responses": num_responses,
        }


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
        # Aggregate statistics (displayed first for better UX in parquet viewers)
        ('avg_score', pa.float64()),
        ('max_score', pa.float64()),
        ('min_score', pa.float64()),
        ('success_rate', pa.float64()),
        ('num_correct', pa.int64()),
        ('num_responses', pa.int64()),
        # Unified responses list with all inference + scoring fields (at end to not block view)
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
        ])))
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
# 2. Usage Normalization Helper
# ==============================================================================
def normalize_usage(usage: dict | None) -> dict:
    """
    Normalize usage dictionary to consistent format.

    Handles None values and missing fields, returning a dictionary
    with guaranteed keys: prompt_tokens, completion_tokens, total_tokens.

    Args:
        usage: Usage dictionary from inference result (or None)

    Returns:
        Normalized usage dictionary with all fields as integers
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


# ==============================================================================
# 3. Output Adapter - Convert Document back to VERL format + results
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
# Pipeline Builder - Construct pipeline from CLI arguments
# ==============================================================================
def build_pipeline(args):
    """
    Build processing pipeline from CLI arguments.

    Args:
        args: Parsed argparse arguments

    Returns:
        List of pipeline steps
    """
    import asyncio

    # Create picklable callable instance with configuration from CLI arguments
    # This instance captures configuration at creation time and is properly
    # pickled/unpickled in worker processes (unlike module-level globals)
    rollout_fn = VERLRolloutFunction(
        num_responses=args.num_responses_per_prompt,
        temperature=args.sampling_temperature,
        max_tokens=args.max_tokens_per_response,
        sandbox_url=args.sandbox_fusion_url,
    )

    # Create shared context with dependencies to pass as kwargs to rollout_fn
    # This replaces the old max_concurrent_scoring parameter
    shared_context = {
        "scoring_semaphore": asyncio.Semaphore(args.max_concurrent_scoring),
        "sandbox_url": args.sandbox_fusion_url,
    }

    pipeline = [
        # Step 1: Read VERL parquet data
        ParquetReader(
            data_folder=args.input_data,
            adapter=verl_to_document_adapter,
            batch_size=args.batch_size_for_reading,
            recursive=True,
            glob_pattern="*.parquet",
        ),
        # Step 2: Generate multiple responses and score them
        # Use .run_with_yield method reference to enable pipeline chaining
        InferenceRunner(
            rollout_fn=rollout_fn,  # Unified function (replaces query_builder + postprocess_fn)
            config=InferenceConfig(
                server_type="endpoint" if args.inference_server_type == "vllm-remote" else args.inference_server_type,
                model_name_or_path=args.model_name_or_path,
                default_generation_params={"temperature": args.sampling_temperature},  # Dict format
                max_concurrent_generations=args.max_concurrent_inference,  # Renamed
                max_concurrent_documents=100,  # Renamed (reduced to prevent thread pool exhaustion)
                metric_interval=120,  # Report metrics every 2 minutes
                endpoint_url=args.remote_vllm_endpoint if args.inference_server_type == "vllm-remote" else None,
            ),
            output_writer=ParquetWriter(
                output_folder=args.output_dir,
                adapter=document_to_verl_adapter,
                output_filename=args.output_filename_pattern,
                compression=args.output_compression,
                schema=VERL_SCHEMA,  # Explicit schema preserves error fields
                batch_size=args.checkpoint_frequency,  # Match checkpoint frequency for consistent flushing
            ),
            shared_context=shared_context,  # NEW: Pass dependencies as kwargs
            checkpoints_local_dir=args.checkpoint_dir,
            records_per_chunk=args.checkpoint_frequency,
            # metadata_key default is "rollout_results" (not used by VERL output adapter)
        ),
    ]

    # Optional: Add statistics collection if requested
    if args.stats_output_dir:
        pipeline.append(
            ResponseScoreStats(
                output_folder=args.stats_output_dir,
                groups_to_compute=["summary", "histogram"],
            )
        )

    return pipeline


# ==============================================================================
# Main CLI Function
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="VERL data processing with multi-response generation and reward scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/processing/verl_data_processing.py \\
      --input-data data/math.parquet \\
      --output-dir output/math \\
      --model-name-or-path meta-llama/Llama-3-8B

  # With code execution scoring
  python scripts/processing/verl_data_processing.py \\
      --input-data data/code.parquet \\
      --output-dir output/code \\
      --model-name-or-path deepseek-coder-7b \\
      --sandbox-fusion-url http://localhost:5000

For more examples: scripts/processing/run_verl_examples.sh
For details: examples/verl_data_processing.py
        """
    )

    # ============================================================
    # Required Arguments
    # ============================================================
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--input-data',
        required=True,
        type=str,
        metavar='PATH',
        help='Path to input VERL parquet file or directory'
    )
    required.add_argument(
        '--output-dir',
        required=True,
        type=str,
        metavar='PATH',
        help='Output directory for processed parquet files'
    )
    required.add_argument(
        '--model-name-or-path',
        required=True,
        type=str,
        metavar='TEXT',
        help='Model name (HuggingFace Hub) or local path (e.g., meta-llama/Llama-3-8B)'
    )

    # ============================================================
    # Response Generation Settings
    # ============================================================
    generation = parser.add_argument_group('response generation settings')
    generation.add_argument(
        '--num-responses-per-prompt',
        type=int,
        default=10,
        metavar='INT',
        help='Number of responses to generate per prompt (default: 10)'
    )
    generation.add_argument(
        '--sampling-temperature',
        type=float,
        default=0.7,
        metavar='FLOAT',
        help='Sampling temperature for response diversity (default: 0.7)'
    )
    generation.add_argument(
        '--max-tokens-per-response',
        type=int,
        default=2048,
        metavar='INT',
        help='Maximum tokens per generated response (default: 2048)'
    )

    # ============================================================
    # Inference Server Settings
    # ============================================================
    server = parser.add_argument_group('inference server settings')
    server.add_argument(
        '--inference-server-type',
        type=str,
        choices=['vllm', 'sglang', 'vllm-remote'],
        default='vllm',
        help='Type of inference server to use (default: vllm)'
    )
    server.add_argument(
        '--remote-vllm-endpoint',
        type=str,
        metavar='URL',
        help='Remote vLLM server endpoint URL (required when --inference-server-type=vllm-remote)'
    )
    server.add_argument(
        '--max-concurrent-inference',
        type=int,
        default=100,
        metavar='INT',
        help='Maximum concurrent inference requests (default: 100)'
    )

    # ============================================================
    # Reward Scoring Settings
    # ============================================================
    scoring = parser.add_argument_group('reward scoring settings')
    scoring.add_argument(
        '--sandbox-fusion-url',
        type=str,
        metavar='URL',
        help='Sandbox Fusion server URL for code execution scoring (optional, required for code datasets like codecontests)'
    )
    scoring.add_argument(
        '--max-concurrent-scoring',
        type=int,
        default=50,
        metavar='INT',
        help='Maximum concurrent scoring requests to sandbox server (default: 50)'
    )

    # ============================================================
    # Parallel Execution Settings
    # ============================================================
    execution = parser.add_argument_group('parallel execution settings')
    execution.add_argument(
        '--num-parallel-tasks',
        type=int,
        default=10,
        metavar='INT',
        help='Number of parallel processing tasks for data sharding (default: 10)'
    )
    execution.add_argument(
        '--num-concurrent-workers',
        type=int,
        default=5,
        metavar='INT',
        help='Number of concurrent workers per task (default: 5)'
    )

    # ============================================================
    # Checkpointing & Logging
    # ============================================================
    checkpointing = parser.add_argument_group('checkpointing and logging')
    checkpointing.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/verl',
        metavar='PATH',
        help='Directory for saving processing checkpoints (default: checkpoints/verl)'
    )
    checkpointing.add_argument(
        '--log-dir',
        type=str,
        default='logs/verl_processing',
        metavar='PATH',
        help='Directory for saving execution logs (default: logs/verl_processing)'
    )
    checkpointing.add_argument(
        '--stats-output-dir',
        type=str,
        metavar='PATH',
        help='Directory for statistics output (optional, enables statistics collection if specified)'
    )
    checkpointing.add_argument(
        '--checkpoint-frequency',
        type=int,
        default=500,
        metavar='INT',
        help='Save checkpoint every N processed documents (default: 500). '
             'Also controls ParquetWriter batch size for consistent flushing.'
    )

    # ============================================================
    # Data Processing Options
    # ============================================================
    processing = parser.add_argument_group('data processing options')
    processing.add_argument(
        '--batch-size-for-reading',
        type=int,
        default=100,
        metavar='INT',
        help='Batch size for reading input parquet files (default: 100)'
    )
    processing.add_argument(
        '--output-compression',
        type=str,
        choices=['snappy', 'gzip', 'none'],
        default='snappy',
        help='Compression algorithm for output parquet files (default: snappy)'
    )
    processing.add_argument(
        '--output-filename-pattern',
        type=str,
        default='${rank}_chunk_${chunk_index}.parquet',
        metavar='TEXT',
        help='Output filename pattern with variables ${rank} and ${chunk_index} (default: ${rank}_chunk_${chunk_index}.parquet)'
    )

    # ============================================================
    # Error Handling
    # ============================================================
    error_handling = parser.add_argument_group('error handling')
    error_handling.add_argument(
        '--stop-on-bad-request',
        action='store_true',
        help='Stop processing when a BadRequestError occurs (default: skip problematic documents and continue)'
    )

    args = parser.parse_args()

    # Validation
    if args.inference_server_type == 'vllm-remote' and not args.remote_vllm_endpoint:
        parser.error('--remote-vllm-endpoint is required when --inference-server-type=vllm-remote')

    # Create output directories if they don't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    if args.stats_output_dir:
        Path(args.stats_output_dir).mkdir(parents=True, exist_ok=True)

    # Configure ThreadPoolExecutor for asyncio.to_thread()
    # Increase from default (~32) to 2000 to prevent deadlock when
    # max_concurrent_tasks × responses_per_prompt > default thread pool size
    import asyncio
    import concurrent.futures

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2000)
    loop.set_default_executor(executor)

    # Build and run pipeline
    print(f"Building pipeline with configuration:")
    print(f"  Input: {args.input_data}")
    print(f"  Output: {args.output_dir}")
    print(f"  Model: {args.model_name_or_path}")
    print(f"  Responses per prompt: {args.num_responses_per_prompt}")
    print(f"  Temperature: {args.sampling_temperature}")
    print(f"  Parallel tasks: {args.num_parallel_tasks}")
    print(f"  ThreadPool size: 2000")
    print(f"  Checkpoint frequency: {args.checkpoint_frequency}")
    print()

    pipeline = build_pipeline(args)

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=args.log_dir,
        tasks=args.num_parallel_tasks,
        workers=args.num_concurrent_workers,
    )

    print("Starting pipeline execution...")
    executor.run()
    print("Pipeline execution completed!")


if __name__ == "__main__":
    main()
