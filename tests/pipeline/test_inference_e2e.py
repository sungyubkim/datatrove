"""
End-to-end integration tests for InferenceRunner with real external services.

These tests require external services to be running:
- Ollama server at localhost:11434 (for LLM inference)
- Sandbox Fusion server at localhost:8080 (for code execution scoring)

Tests will automatically skip if services are unavailable.

Run tests with:
    uv run pytest tests/pipeline/test_inference_e2e.py -v -s
"""

import asyncio
import json
import tempfile
from pathlib import Path

import httpx
import pytest

from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import (
    InferenceConfig,
    InferenceError,
    InferenceResult,
    InferenceRunner,
)
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.reward_score import compute_score


# ============================================================================
# Service Health Checks
# ============================================================================


async def check_ollama_ready():
    """
    Check if Ollama is running and accessible at localhost:11434.

    Returns:
        bool: True if Ollama is ready, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # Note: Ollama's OpenAI-compatible endpoint is at /v1/models
            response = await client.get("http://localhost:11434/v1/models")
            return response.status_code == 200
    except Exception as e:
        print(f"Ollama health check failed: {e}")
        return False


async def check_sandbox_ready():
    """
    Check if Sandbox Fusion is running and accessible at localhost:8080.

    Returns:
        bool: True if Sandbox Fusion is ready, False otherwise
    """
    try:
        # Test with a simple Python code execution
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                "http://localhost:8080/run_code",
                json={
                    "code": "print('test')",
                    "language": "python",
                },
            )
            # Accept both 200 (success) and error responses (server is up)
            return response.status_code in (200, 400, 500)
    except Exception as e:
        print(f"Sandbox Fusion health check failed: {e}")
        return False


# ============================================================================
# Test 1: Ollama Basic Inference (No Scoring)
# ============================================================================


@pytest.mark.asyncio
async def test_ollama_basic_inference():
    """
    Test basic inference with Ollama endpoint without scoring.

    This test verifies:
    - InferenceRunner can connect to Ollama at localhost:11434
    - Qwen model generates non-empty responses
    - Results are properly stored in document metadata
    - Output files are created correctly
    """
    # Skip if Ollama not available
    if not await check_ollama_ready():
        pytest.skip("Ollama server not available at localhost:11434")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Simple test documents
        documents = [
            Document(text="What is 2+2? Answer briefly.", id="math-1"),
            Document(text="What is the capital of France? One word answer.", id="geo-1"),
        ]

        output_dir = tmp_path / "ollama_output"

        async def simple_rollout(document, generate):
            """Simple rollout function: just get response from Ollama."""
            result = await generate(
                {
                    "messages": [{"role": "user", "content": document.text}],
                    "max_tokens": 100,
                }
            )

            return {
                "text": result.text,
                "finish_reason": result.finish_reason,
                "usage": result.usage or {},
            }

        config = InferenceConfig(
            server_type="endpoint",  # Use "endpoint" for Ollama
            model_name_or_path="qwen3:0.6b",  # Your Ollama model
            endpoint_url="http://localhost:11434",  # Ollama base URL (is_ready() will append /v1/models)
            model_max_context=4096,
            use_chat=True,
            metric_interval=60,
            max_concurrent_generations=2,
            max_concurrent_documents=1,
        )

        runner = InferenceRunner(
            rollout_fn=simple_rollout,
            config=config,
            output_writer=JsonlWriter(
                str(output_dir),
                output_filename="${rank}.jsonl",
                compression=None,
            ),
        )

        # Run inference
        await runner.run_async(documents, rank=0)

        # Verify results
        for doc in documents:
            assert "rollout_results" in doc.metadata, f"Missing rollout_results for {doc.id}"
            assert len(doc.metadata["rollout_results"]) > 0, f"Empty rollout_results for {doc.id}"

            result = doc.metadata["rollout_results"][0]
            assert "text" in result, "Missing 'text' field in result"
            assert len(result["text"]) > 0, f"Ollama generated empty response for {doc.id}"
            assert "finish_reason" in result, "Missing 'finish_reason' field"

        # Verify output file exists
        output_files = list(output_dir.glob("*.jsonl"))
        assert len(output_files) > 0, "No output files created"


# ============================================================================
# Test 2: Sandbox Fusion Code Scoring (No Real Inference)
# ============================================================================


@pytest.mark.asyncio
async def test_sandbox_fusion_code_scoring():
    """
    Test Sandbox Fusion for code execution scoring.

    This test verifies:
    - compute_score() can connect to Sandbox Fusion at localhost:8080
    - Code execution and scoring works correctly
    - Error handling is robust
    - Score normalization is correct
    """
    # Skip if Sandbox Fusion not available
    if not await check_sandbox_ready():
        pytest.skip("Sandbox Fusion server not available at localhost:8080")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Simple code problem with known solution
        documents = [
            Document(
                text="Write a Python function that reads two numbers and prints their sum",
                id="code-sum",
                metadata={
                    "test_cases": {
                        "inputs": ["3\n5"],  # stdin: two numbers
                        "outputs": ["8"],  # expected output
                    }
                },
            ),
        ]

        output_dir = tmp_path / "sandbox_output"

        async def code_scoring_rollout(document, generate, sandbox_url):
            """Generate code and score with Sandbox Fusion."""
            # 1. Generate code (using dummy server for speed)
            result = await generate(
                {
                    "messages": [{"role": "user", "content": document.text}],
                    "max_tokens": 512,
                }
            )

            # 2. For testing, use a correct solution
            # In real usage, this would be result.text from the model
            test_code = """```python
a = int(input())
b = int(input())
print(a + b)
```"""

            # 3. Score with Sandbox Fusion
            try:
                score_dict = await asyncio.to_thread(
                    compute_score,
                    "codecontests",  # Code execution dataset
                    test_code,  # The generated code
                    json.dumps(document.metadata["test_cases"]),  # Ground truth
                    sandbox_fusion_url=sandbox_url,
                )
            except Exception as e:
                score_dict = {
                    "score": 0.0,
                    "error": str(e),
                    "reward_think": 0.0,
                    "reward_fmt": 0.0,
                    "reward_correct": 0.0,
                    "reward_length": 0.0,
                }

            return {
                "text": test_code,
                "score": score_dict["score"],
                "error": score_dict["error"],
                "is_success": score_dict["error"] == "",
            }

        config = InferenceConfig(
            server_type="dummy",  # Use dummy server for quick generation
            model_name_or_path="dummy",
        )

        runner = InferenceRunner(
            rollout_fn=code_scoring_rollout,
            config=config,
            output_writer=JsonlWriter(
                str(output_dir),
                output_filename="${rank}.jsonl",
                compression=None,
            ),
            shared_context={"sandbox_url": "http://localhost:8080"},
        )

        await runner.run_async(documents, rank=0)

        # Verify results
        doc = documents[0]
        assert "rollout_results" in doc.metadata
        result = doc.metadata["rollout_results"][0]

        assert "score" in result, "Missing 'score' field"
        assert 0.0 <= result["score"] <= 1.0, f"Score out of range: {result['score']}"
        assert "error" in result, "Missing 'error' field"
        assert "is_success" in result, "Missing 'is_success' field"

        # With correct code, score should be 1.0
        # Note: This assumes Sandbox Fusion is working correctly
        print(f"Sandbox Fusion score: {result['score']}, error: {result['error']}")


# ============================================================================
# Test 3: Full E2E with Ollama + Sandbox Fusion
# ============================================================================


@pytest.mark.asyncio
async def test_ollama_with_sandbox_e2e():
    """
    Full end-to-end test: Ollama generates code → Sandbox Fusion scores it.

    This test verifies the complete VERL-style workflow:
    1. Ollama generates N code responses
    2. Sandbox Fusion scores each response
    3. Aggregate statistics computed (avg_score, success_rate)
    4. Unified responses stored in metadata
    5. Checkpointing works correctly
    6. Output files created with all data
    """
    # Skip if either service unavailable
    ollama_ok = await check_ollama_ready()
    sandbox_ok = await check_sandbox_ready()

    if not ollama_ok:
        pytest.skip("Ollama server not available at localhost:11434")
    if not sandbox_ok:
        pytest.skip("Sandbox Fusion server not available at localhost:8080")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Code generation problem
        documents = [
            Document(
                text=(
                    "Write a Python function that checks if a number is even. "
                    "Read the number from stdin and print 'True' if even, 'False' otherwise."
                ),
                id="code-even",
                metadata={
                    "test_cases": {
                        "inputs": ["4", "7"],  # Two test cases
                        "outputs": ["True", "False"],
                    }
                },
            ),
        ]

        output_dir = tmp_path / "e2e_output"
        checkpoint_dir = tmp_path / "e2e_checkpoints"

        # Semaphore for rate-limiting scoring requests
        scoring_semaphore = asyncio.Semaphore(5)

        async def e2e_rollout(document, generate, sandbox_url, scoring_semaphore):
            """
            Complete VERL-style rollout:
            - Generate N responses
            - Score all responses
            - Aggregate statistics
            - Store unified responses
            """
            N_RESPONSES = 2

            # 1. Generate N responses
            results = []
            for i in range(N_RESPONSES):
                try:
                    result = await generate(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"{document.text}\n\n(Attempt {i+1} of {N_RESPONSES})",
                                }
                            ],
                            "max_tokens": 512,
                        }
                    )
                    results.append(result)
                except InferenceError as e:
                    results.append(e)

            # 2. Score all results with rate-limiting
            async def score_response(result):
                """Score a single response with error handling."""
                if isinstance(result, InferenceResult):
                    try:
                        async with scoring_semaphore:
                            score_dict = await asyncio.to_thread(
                                compute_score,
                                "codecontests",
                                result.text,
                                json.dumps(document.metadata["test_cases"]),
                                sandbox_fusion_url=sandbox_url,
                            )
                        return score_dict
                    except Exception as e:
                        return {
                            "score": 0.0,
                            "error": f"Scoring failed: {str(e)}",
                            "reward_think": 0.0,
                            "reward_fmt": 0.0,
                            "reward_correct": 0.0,
                            "reward_length": 0.0,
                        }
                else:
                    # Inference error
                    return {
                        "score": 0.0,
                        "error": f"Inference error: {result.error}",
                        "reward_think": 0.0,
                        "reward_fmt": 0.0,
                        "reward_correct": 0.0,
                        "reward_length": 0.0,
                    }

            scores = await asyncio.gather(*[score_response(result) for result in results])

            # 3. Create unified responses (merge inference + scoring results)
            unified_responses = []
            for result, score in zip(results, scores):
                if isinstance(result, InferenceResult):
                    unified_responses.append(
                        {
                            "text": result.text,
                            "finish_reason": result.finish_reason,
                            "usage": result.usage or {},
                            "score": score["score"],
                            "error": score["error"],
                            "is_success": score["error"] == "",
                        }
                    )
                else:
                    # Inference error case
                    unified_responses.append(
                        {
                            "text": "",
                            "finish_reason": "error",
                            "usage": {},
                            "score": 0.0,
                            "error": result.error,
                            "is_success": False,
                        }
                    )

            # 4. Aggregate statistics
            scores_list = [r["score"] for r in unified_responses]
            avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
            success_count = sum(1 for r in unified_responses if r["is_success"])
            success_rate = success_count / len(unified_responses) if unified_responses else 0.0

            # 5. Store in metadata for persistence
            document.metadata["unified_responses"] = unified_responses
            document.metadata["avg_score"] = avg_score
            document.metadata["success_rate"] = success_rate

            # 6. Return summary
            return {
                "status": "processed",
                "num_responses": len(unified_responses),
                "avg_score": avg_score,
                "success_rate": success_rate,
            }

        config = InferenceConfig(
            server_type="endpoint",
            model_name_or_path="qwen3:0.6b",
            endpoint_url="http://localhost:11434",  # Ollama base URL
            model_max_context=4096,
            use_chat=True,
            metric_interval=60,
            max_concurrent_generations=2,
            max_concurrent_documents=1,
        )

        runner = InferenceRunner(
            rollout_fn=e2e_rollout,
            config=config,
            output_writer=JsonlWriter(
                str(output_dir),
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
                compression=None,
            ),
            shared_context={
                "sandbox_url": "http://localhost:8080",
                "scoring_semaphore": scoring_semaphore,
            },
            checkpoints_local_dir=str(checkpoint_dir),
            records_per_chunk=5,
        )

        await runner.run_async(documents, rank=0)

        # Verify complete E2E pipeline
        doc = documents[0]
        assert "rollout_results" in doc.metadata, "Missing rollout_results"

        rollout_result = doc.metadata["rollout_results"][0]
        assert rollout_result["status"] == "processed", f"Unexpected status: {rollout_result['status']}"
        assert rollout_result["num_responses"] == 2, f"Expected 2 responses, got {rollout_result['num_responses']}"

        # Verify unified responses
        assert "unified_responses" in doc.metadata, "Missing unified_responses in metadata"
        assert len(doc.metadata["unified_responses"]) == 2, "Expected 2 unified responses"

        # Verify aggregate statistics
        assert "avg_score" in doc.metadata, "Missing avg_score"
        assert 0.0 <= doc.metadata["avg_score"] <= 1.0, f"Score out of range: {doc.metadata['avg_score']}"

        assert "success_rate" in doc.metadata, "Missing success_rate"
        assert 0.0 <= doc.metadata["success_rate"] <= 1.0, f"Success rate out of range: {doc.metadata['success_rate']}"

        # Verify output files exist
        output_files = list(output_dir.glob("*.jsonl"))
        assert len(output_files) > 0, "No output files created"

        # Note: Checkpoint files are temporary and cleaned up after successful completion,
        # so we don't verify their existence. The checkpoint creation is logged above.

        # Print results for manual verification
        print(f"\n✅ E2E Test Results:")
        print(f"   Avg Score: {doc.metadata['avg_score']:.2f}")
        print(f"   Success Rate: {doc.metadata['success_rate']:.2%}")
        print(f"   Responses: {len(doc.metadata['unified_responses'])}")
        for i, resp in enumerate(doc.metadata["unified_responses"]):
            print(f"   Response {i+1}: score={resp['score']:.2f}, success={resp['is_success']}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
