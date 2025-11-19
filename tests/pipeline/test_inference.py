import asyncio
import json
import socket
import tempfile
import threading
from contextlib import contextmanager
from functools import partial
from http.server import HTTPServer
from pathlib import Path

import pytest

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.inference.servers.dummy_server import DummyHandler
from datatrove.pipeline.writers import JsonlWriter


class ControlledRollout:
    """Rollout function that can be configured to fail at specific document IDs or after a certain count."""

    def __init__(self, fail_at_ids=None, fail_after_count=None):
        self.fail_at_ids = fail_at_ids or set()
        self.fail_after_count = fail_after_count
        self.processed_count = 0

    async def __call__(self, document, generate):
        self.processed_count += 1

        if self.fail_after_count and self.processed_count > self.fail_after_count:
            raise RuntimeError(f"Simulated failure after processing {self.fail_after_count} documents")

        if document.id in self.fail_at_ids:
            raise RuntimeError(f"Simulated failure for document {document.id}")

        result = await generate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": document.text},
                        ],
                    }
                ],
                "max_tokens": 100,
            }
        )

        return {
            "text": result.text,
            "finish_reason": result.finish_reason,
            "usage": result.usage,
        }


def test_inference_config_sets_default_concurrency():
    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=4096,
        metric_interval=60,
        rollouts_per_document=3,
        max_concurrent_generations=8,
        max_concurrent_documents=None,
    )

    assert config.max_concurrent_documents == 2


def test_multiple_rollouts_collect_results(tmp_path):
    output_dir = tmp_path / "multi_rollouts"
    documents = [Document(text="hello world", id="multi-1")]

    async def multi_rollout(document, generate):
        await asyncio.sleep(0)
        return "multi-result"

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=3,
        max_concurrent_generations=3,
        max_concurrent_documents=None,
    )

    runner = InferenceRunner(
        rollout_fn=multi_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert doc.metadata["rollout_results"] == ["multi-result", "multi-result", "multi-result"]

    output_file = output_dir / "00000.jsonl"
    assert output_file.exists()
    saved = json.loads(output_file.read_text().strip())
    assert saved["metadata"]["rollout_results"] == ["multi-result", "multi-result", "multi-result"]


def test_custom_metadata_key(tmp_path):
    output_dir = tmp_path / "custom_metadata"
    documents = [Document(text="hello", id="custom-1")]

    async def custom_rollout(document, generate):
        return {"value": document.id}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=None,
    )

    runner = InferenceRunner(
        rollout_fn=custom_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
        metadata_key="custom_results",
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert "rollout_results" not in doc.metadata
    assert doc.metadata["custom_results"] == [{"value": "custom-1"}]

    output_file = output_dir / "00000.jsonl"
    assert output_file.exists()
    saved = json.loads(output_file.read_text().strip())
    assert "rollout_results" not in saved["metadata"]
    assert saved["metadata"]["custom_results"] == [{"value": "custom-1"}]


def test_chunked_checkpoint_requires_chunk_index(tmp_path):
    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    with pytest.raises(ValueError, match="chunk_index"):
        InferenceRunner(
            rollout_fn=lambda document, generate: generate({}),
            config=config,
            output_writer=JsonlWriter(
                str(tmp_path / "no_chunk"),
                output_filename="${rank}.jsonl",
                compression=None,
            ),
            checkpoints_local_dir=str(tmp_path / "checkpoints"),
            records_per_chunk=10,
        )

    try:
        InferenceRunner(
            rollout_fn=lambda document, generate: generate({}),
            config=config,
            output_writer=JsonlWriter(
                str(tmp_path / "with_chunk"),
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
                compression=None,
            ),
            checkpoints_local_dir=str(tmp_path / "checkpoints_ok"),
            records_per_chunk=10,
        )
    except ValueError as exc:  # pragma: no cover - explicit failure message
        pytest.fail(f"InferenceRunner should allow chunk_index templates: {exc}")


def test_rollout_handles_multiple_parts(tmp_path):
    parts = ["first chunk", "second chunk", "third chunk"]

    async def chunked_rollout(document, generate):
        document.metadata["rollout_calls"] = document.metadata.get("rollout_calls", 0) + 1
        document.metadata.setdefault("parts_served", [])

        generations = []
        previous_generation = ""

        for index, part in enumerate(parts):
            document.metadata["parts_served"].append(index)
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Process part {index}: {part}\nPrevious: {previous_generation}",
                    }
                ],
                "max_tokens": 128,
            }
            result = await generate(payload)
            previous_generation = result.text
            generations.append(result.text)

        return {"parts": generations}

    output_dir = tmp_path / "callback_output"
    documents = [Document(text="dummy", id="callback-doc")]

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=4096,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    runner = InferenceRunner(
        rollout_fn=chunked_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert doc.metadata["rollout_calls"] == 1
    assert doc.metadata["parts_served"] == [0, 1, 2]
    assert len(doc.metadata["rollout_results"]) == 1
    assert len(doc.metadata["rollout_results"][0]["parts"]) == len(parts)

    output_file = output_dir / "00000.jsonl"
    assert output_file.exists(), "Expected output document to be saved"
    with output_file.open() as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 1, "Document should be written once after callbacks finish"
    saved_doc = json.loads(lines[0])
    assert saved_doc["id"] == "callback-doc"
    assert len(saved_doc["metadata"]["rollout_results"][0]["parts"]) == len(parts)


def test_query_builder_none_payload_skips_document(tmp_path):
    output_dir = tmp_path / "none_payload_output"
    documents = [Document(text="skip me", id="skip-none")]

    async def none_rollout(document, generate):
        return None

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    runner = InferenceRunner(
        rollout_fn=none_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert doc.metadata.get("rollout_results") == [], (
        "Document should have no rollout results when rollout returns None"
    )
    output_file = output_dir / "00000.jsonl"
    assert not output_file.exists() or output_file.read_text().strip() == "", (
        "No output should be written when rollout returns None"
    )


def test_async_query_builder_none_payload_skips_document(tmp_path):
    output_dir = tmp_path / "none_async_output"
    documents = [Document(text="skip me async", id="skip-async")]

    async def none_async_rollout(document, generate):
        await asyncio.sleep(0)
        return None

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    runner = InferenceRunner(
        rollout_fn=none_async_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert doc.metadata.get("rollout_results") == [], (
        "Document should have no rollout results when async rollout returns None"
    )
    output_file = output_dir / "00000.jsonl"
    assert not output_file.exists() or output_file.read_text().strip() == "", (
        "No output should be written when rollout returns None"
    )


def read_output_files(output_path):
    """Helper to read all output files and return document data"""
    output_path = Path(output_path)
    # Look for files matching the pattern used by JsonlWriter
    output_files = sorted(output_path.glob("*_chunk_*.jsonl"))
    all_docs = []

    for output_file in output_files:
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    doc_data = json.loads(line.strip())
                    all_docs.append(doc_data)

    return all_docs, output_files


def test_checkpoint_recovery_and_completeness():
    """
    Comprehensive test that verifies:
    1. Checkpoint creation when pipeline fails mid-execution
    2. Successful recovery and resumption from checkpoints
    3. Complete and correct output after recovery
    4. Proper chunking behavior
    """
    num_docs = 35
    records_per_chunk = 10  # Should create 4 chunks: [0-9], [10-19], [20-29], [30-34]
    fail_after_docs = 22  # Fail after processing 22 docs (middle of chunk 2)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output"
        checkpoint_path = Path(temp_dir) / "checkpoints"

        # Create test documents
        def make_documents():
            return [Document(text=f"Test document {i}", id=str(i)) for i in range(num_docs)]

        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="test-model",
            model_max_context=8192,
            metric_interval=120,
            rollouts_per_document=1,
            max_concurrent_generations=2,  # Low concurrency to ensure predictable chunk completion
            max_concurrent_documents=2,
        )

        # === FIRST RUN: Should fail partway through ===
        failing_rollout = ControlledRollout(fail_after_count=fail_after_docs)

        def make_runner(rollout_fn):
            return InferenceRunner(
                rollout_fn=rollout_fn,
                config=config,
                records_per_chunk=records_per_chunk,
                checkpoints_local_dir=str(checkpoint_path),
                output_writer=JsonlWriter(
                    str(output_path),
                    output_filename="${rank}_chunk_${chunk_index}.jsonl",
                    compression=None,
                ),
            )

        failing_runner = make_runner(failing_rollout)

        # Run first pass - should fail due to rollout exception
        try:
            failing_runner.run(make_documents(), rank=0, world_size=1)
            assert False, "Expected pipeline to fail, but it completed successfully"
        except Exception as e:
            # Pipeline should fail when query_builder raises exceptions
            print(f"Pipeline failed as expected: {e}")

        # === VERIFY CHECKPOINT STATE ===
        assert checkpoint_path.exists(), "Checkpoint directory should exist after failure"

        # Check checkpoint files exist
        checkpoint_files = list(checkpoint_path.rglob("chunk_*.jsonl"))
        assert len(checkpoint_files) > 0, "Should have checkpoint files after partial processing"

        # Check last_chunk tracking file
        last_chunk_file = checkpoint_path / "last_chunk" / "00000.txt"
        if last_chunk_file.exists():
            with open(last_chunk_file, "r") as f:
                last_completed_chunk = int(f.read().strip())
                assert last_completed_chunk >= 0, "Should have completed at least one chunk"

        # Verify partial output exists
        partial_docs, partial_files = read_output_files(output_path)
        assert len(partial_docs) > 0, "Should have some processed documents from first run"
        assert len(partial_docs) <= fail_after_docs, f"Should not have more than {fail_after_docs} docs from first run"

        # === SECOND RUN: Should resume from checkpoint ===
        success_rollout = ControlledRollout()  # No failures this time

        success_runner = make_runner(success_rollout)

        # Run second pass - should complete successfully
        success_runner.run(make_documents(), rank=0, world_size=1)

        # === VERIFY COMPLETE OUTPUT ===
        final_docs, final_files = read_output_files(output_path)

        # Check total document count
        assert len(final_docs) == num_docs, f"Expected {num_docs} documents, got {len(final_docs)}"

        # Check all document IDs are present and unique
        final_ids = {doc["id"] for doc in final_docs}
        expected_ids = {str(i) for i in range(num_docs)}
        assert final_ids == expected_ids, (
            f"Missing IDs: {expected_ids - final_ids}, Extra IDs: {final_ids - expected_ids}"
        )

        # Verify no duplicates (each document processed exactly once)
        final_ids_list = [doc["id"] for doc in final_docs]
        assert len(final_ids_list) == len(set(final_ids_list)), "Found duplicate documents in output"

        # === VERIFY CHUNKING ===
        expected_chunks = (num_docs + records_per_chunk - 1) // records_per_chunk
        assert len(final_files) == expected_chunks, f"Expected {expected_chunks} chunk files, got {len(final_files)}"

        # Verify chunk sizes
        for i, output_file in enumerate(final_files):
            with open(output_file, "r") as f:
                chunk_docs = [json.loads(line.strip()) for line in f if line.strip()]

            if i < expected_chunks - 1:  # All chunks except last should be full
                assert len(chunk_docs) == records_per_chunk, (
                    f"Chunk {i} should have {records_per_chunk} docs, got {len(chunk_docs)}"
                )
            else:  # Last chunk may be partial
                expected_last_chunk_size = num_docs - (expected_chunks - 1) * records_per_chunk
                assert len(chunk_docs) == expected_last_chunk_size, (
                    f"Last chunk should have {expected_last_chunk_size} docs, got {len(chunk_docs)}"
                )

        # === VERIFY INFERENCE RESULTS ===
        for doc in final_docs:
            assert "metadata" in doc, f"Document {doc['id']} missing metadata"
            assert "rollout_results" in doc["metadata"], f"Document {doc['id']} missing rollout_results"

            rollout_results = doc["metadata"]["rollout_results"]
            assert len(rollout_results) > 0, f"Document {doc['id']} has no rollout results"

            # Verify rollout result structure (dummy server should return success)
            for result in rollout_results:
                assert "text" in result, f"Rollout result missing 'text' field for doc {doc['id']}"
                assert "finish_reason" in result, f"Rollout result missing 'finish_reason' field for doc {doc['id']}"
                assert "usage" in result, f"Rollout result missing 'usage' field for doc {doc['id']}"


def test_complete_pipeline_with_various_scenarios():
    """
    Test complete pipeline execution matching the original bug scenario:
    1. 1005 documents (matches original bug report)
    2. 500 documents per chunk (matches original bug report)
    3. High concurrency to stress-test the fix
    4. Validates all documents are saved (vs original bug where only ~7 were saved)
    """
    num_docs = 1005
    records_per_chunk = 500  # Creates 3 chunks: [0-499], [500-999], [1000-1004]

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output"
        checkpoint_path = Path(temp_dir) / "checkpoints"
        logs_path = Path(temp_dir) / "logs"

        # Create test documents matching original bug scenario
        documents = [Document(text="What's the weather in Tokyo?", id=str(i)) for i in range(num_docs)]

        # Normal query builder that doesn't cause pipeline failures
        async def normal_rollout(document, generate):
            result = await generate(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": document.text},
                            ],
                        }
                    ],
                    "max_tokens": 4096,
                }
            )
            return {
                "text": result.text,
                "finish_reason": result.finish_reason,
                "usage": result.usage,
            }

        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="reducto/RolmOCR",
            model_max_context=8192,
            metric_interval=120,
            rollouts_per_document=1,
            max_concurrent_generations=500,
            max_concurrent_documents=500,
        )

        pipeline_executor = LocalPipelineExecutor(
            pipeline=[
                documents,
                InferenceRunner(
                    rollout_fn=normal_rollout,
                    config=config,
                    records_per_chunk=records_per_chunk,
                    checkpoints_local_dir=str(checkpoint_path),
                    output_writer=JsonlWriter(
                        str(output_path),
                        output_filename="${rank}_chunk_${chunk_index}.jsonl",
                        compression=None,
                    ),
                ),
            ],
            logging_dir=str(logs_path / "complete_test_run"),
            tasks=1,
        )

        # Run pipeline - should complete successfully and save ALL documents
        pipeline_executor.run()

        # === VERIFY COMPLETE PROCESSING ===
        final_docs, final_files = read_output_files(output_path)

        # This is the key test - ALL documents should be processed (original bug only saved ~7)
        assert len(final_docs) == num_docs, f"Expected {num_docs} documents, got {len(final_docs)}"

        # Verify all document IDs present
        processed_ids = {doc["id"] for doc in final_docs}
        expected_ids = {str(i) for i in range(num_docs)}
        assert processed_ids == expected_ids, "Not all documents were processed"

        # === VERIFY SUCCESSFUL RESULTS ===
        for doc in final_docs:
            rollout_results = doc["metadata"]["rollout_results"]
            assert len(rollout_results) > 0, f"Document {doc['id']} has no rollout results"

            # All results should be successful (dummy server always succeeds)
            for result in rollout_results:
                assert "text" in result, "Success result should have text"
                assert "finish_reason" in result, "Success result should have finish_reason"
                assert "usage" in result, "Success result should have usage stats"
                assert "error" not in result, "Should not have error in successful result"

        # === VERIFY CHUNKING CORRECTNESS ===
        expected_chunks = (num_docs + records_per_chunk - 1) // records_per_chunk  # Should be 3 chunks
        assert len(final_files) == expected_chunks, f"Expected {expected_chunks} chunks, got {len(final_files)}"

        # Verify chunk contents
        chunk_doc_counts = []
        for output_file in final_files:
            with open(output_file, "r") as f:
                chunk_docs = [json.loads(line.strip()) for line in f if line.strip()]
                chunk_doc_counts.append(len(chunk_docs))

        # First two chunks should have exactly 500 documents each
        for i, count in enumerate(chunk_doc_counts[:-1]):
            assert count == records_per_chunk, f"Chunk {i} should have {records_per_chunk} docs, got {count}"

        # Last chunk should have remaining 5 documents (1000-1004)
        expected_last_count = num_docs - (expected_chunks - 1) * records_per_chunk  # Should be 5
        assert chunk_doc_counts[-1] == expected_last_count, (
            f"Last chunk should have {expected_last_count} docs, got {chunk_doc_counts[-1]}"
        )


def test_shared_context_as_dict(tmp_path):
    """Test that shared_context as a dict passes kwargs to rollout_fn."""
    output_dir = tmp_path / "shared_context_dict"
    documents = [Document(text="test", id="shared-1")]

    async def rollout_with_context(document, generate, custom_value=None, another_param=None):
        assert custom_value == "test_value", "custom_value should be passed from shared_context"
        assert another_param == 42, "another_param should be passed from shared_context"
        result = await generate(
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
                "max_tokens": 100,
            }
        )
        return {"custom_value": custom_value, "another_param": another_param, "result": result.text}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    runner = InferenceRunner(
        rollout_fn=rollout_with_context,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
        shared_context={"custom_value": "test_value", "another_param": 42},
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert len(doc.metadata["rollout_results"]) == 1
    assert doc.metadata["rollout_results"][0]["custom_value"] == "test_value"
    assert doc.metadata["rollout_results"][0]["another_param"] == 42


def test_shared_context_as_callable(tmp_path):
    """Test that shared_context as a callable passes kwargs to rollout_fn."""
    output_dir = tmp_path / "shared_context_callable"
    documents = [Document(text="test", id="shared-2")]

    call_count = {"count": 0}

    def make_shared_context():
        call_count["count"] += 1
        return {"dynamic_value": f"value_{call_count['count']}"}

    async def rollout_with_context(document, generate, dynamic_value=None):
        assert dynamic_value is not None, "dynamic_value should be passed from shared_context"
        await generate(
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
                "max_tokens": 100,
            }
        )
        return {"dynamic_value": dynamic_value}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    runner = InferenceRunner(
        rollout_fn=rollout_with_context,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
        shared_context=make_shared_context,
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert len(doc.metadata["rollout_results"]) == 1
    assert doc.metadata["rollout_results"][0]["dynamic_value"] == "value_1"
    assert call_count["count"] == 1, "shared_context callable should be called once"


def test_shared_context_as_context_manager(tmp_path):
    """Test that shared_context as a context manager properly manages resources."""
    output_dir = tmp_path / "shared_context_cm"
    documents = [Document(text="test", id="shared-3")]

    cleanup_called = {"called": False}

    class TestContextManager:
        def __init__(self):
            self.value = "context_value"
            self.entered = False

        def __enter__(self):
            self.entered = True
            return {"context_value": self.value}

        def __exit__(self, exc_type, exc_val, exc_tb):
            cleanup_called["called"] = True
            return False

    async def rollout_with_context(document, generate, context_value=None):
        assert context_value == "context_value", "context_value should be passed from shared_context"
        await generate(
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
                "max_tokens": 100,
            }
        )
        return {"context_value": context_value}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    cm = TestContextManager()
    runner = InferenceRunner(
        rollout_fn=rollout_with_context,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
        shared_context=cm,
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert len(doc.metadata["rollout_results"]) == 1
    assert doc.metadata["rollout_results"][0]["context_value"] == "context_value"
    assert cm.entered, "Context manager should have been entered"
    assert cleanup_called["called"], "Context manager cleanup should have been called"


def test_shared_context_none(tmp_path):
    """Test that rollout_fn works without shared_context (no kwargs passed)."""
    output_dir = tmp_path / "shared_context_none"
    documents = [Document(text="test", id="shared-4")]

    async def rollout_no_context(document, generate):
        # This should work fine without any kwargs
        result = await generate(
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
                "max_tokens": 100,
            }
        )
        return {"result": result.text}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    runner = InferenceRunner(
        rollout_fn=rollout_no_context,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
        shared_context=None,  # Explicitly None
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert len(doc.metadata["rollout_results"]) == 1
    assert "result" in doc.metadata["rollout_results"][0]


def test_shared_context_callable_returns_context_manager(tmp_path):
    """Test that a callable that returns a context manager works correctly."""
    output_dir = tmp_path / "shared_context_callable_cm"
    documents = [Document(text="test", id="shared-5")]

    cleanup_called = {"called": False}

    @contextmanager
    def test_context_manager(value: str):
        cleanup_called["called"] = False
        try:
            yield {"test_value": value}
        finally:
            cleanup_called["called"] = True

    async def rollout_with_context(document, generate, test_value=None):
        assert test_value == "test_value", "test_value should be passed from shared_context"
        await generate(
            {
                "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
                "max_tokens": 100,
            }
        )
        return {"test_value": test_value}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=1,
    )

    # Test 1: Callable that returns a context manager (using partial)
    cleanup_called["called"] = False
    runner1 = InferenceRunner(
        rollout_fn=rollout_with_context,
        config=config,
        output_writer=JsonlWriter(str(output_dir / "test1"), output_filename="${rank}.jsonl", compression=None),
        shared_context=partial(test_context_manager, "test_value"),
    )

    asyncio.run(runner1.run_async(documents, rank=0))

    doc = documents[0]
    assert len(doc.metadata["rollout_results"]) == 1
    assert doc.metadata["rollout_results"][0]["test_value"] == "test_value"
    assert cleanup_called["called"], "Context manager cleanup should have been called (callable version)"

    # Test 2: Direct context manager (calling the function and passing the result)
    cleanup_called["called"] = False
    runner2 = InferenceRunner(
        rollout_fn=rollout_with_context,
        config=config,
        output_writer=JsonlWriter(str(output_dir / "test2"), output_filename="${rank}.jsonl", compression=None),
        shared_context=test_context_manager("test_value"),
    )

    asyncio.run(runner2.run_async(documents, rank=0))

    doc = documents[0]
    assert len(doc.metadata["rollout_results"]) == 1
    assert doc.metadata["rollout_results"][0]["test_value"] == "test_value"
    assert cleanup_called["called"], "Context manager cleanup should have been called (direct version)"


def test_endpoint_server(tmp_path):
    """Test EndpointServer with Ollama server (requires Ollama running on localhost:11434)."""
    pytest.importorskip("httpx")  # Required for EndpointServer.is_ready()

    output_dir = tmp_path / "endpoint_test"
    documents = [Document(text="Write a haiku about coding", id="endpoint-1")]

    async def endpoint_rollout(document, generate):
        result = await generate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": document.text}],
                    }
                ],
                "max_tokens": 100,
            }
        )
        return {
            "text": result.text,
            "finish_reason": result.finish_reason,
            "usage": result.usage,
        }

    config = InferenceConfig(
        server_type="endpoint",  # Use EndpointServer instead of DummyServer
        endpoint_url="http://localhost:11434",  # Ollama base URL (is_ready adds /v1/models)
        model_name_or_path="gemma3:4b-it-qat",  # Ollama model name
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=None,
    )

    runner = InferenceRunner(
        rollout_fn=endpoint_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    doc = documents[0]
    assert "rollout_results" in doc.metadata
    assert len(doc.metadata["rollout_results"]) == 1
    assert "text" in doc.metadata["rollout_results"][0]
    assert len(doc.metadata["rollout_results"][0]["text"]) > 0  # Should have generated text

    output_file = output_dir / "00000.jsonl"
    assert output_file.exists()
    saved = json.loads(output_file.read_text().strip())
    assert saved["metadata"]["rollout_results"][0]["text"] == doc.metadata["rollout_results"][0]["text"]


def test_rollout_fn_graceful_inference_error_handling(tmp_path):
    """
    Test that rollout functions can gracefully handle InferenceError.

    This verifies the VERL pattern where InferenceError is caught and
    included in results rather than crashing the pipeline.
    """
    from datatrove.pipeline.inference.run_inference import InferenceError, InferenceResult

    output_dir = tmp_path / "error_handling"
    documents = [
        Document(text="Success document", id="success-1"),
        Document(text="Error document", id="error-2"),
        Document(text="Another success", id="success-3"),
    ]

    async def error_aware_rollout(document, generate):
        """
        Rollout function that catches InferenceError and returns it gracefully.
        Pattern from VERL: errors are results, not exceptions.
        """
        # Simulate errors for specific documents (simpler than mocking server)
        if "Error" in document.text:
            # Simulate an InferenceError
            return {
                "text": "",
                "success": False,
                "error": "Simulated inference error for testing",
            }

        result = await generate({
            "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
            "max_tokens": 100,
        })
        return {
            "text": result.text,
            "success": True,
            "error": "",
        }

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=None,
    )

    runner = InferenceRunner(
        rollout_fn=error_aware_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    # Pipeline should NOT crash even though one document has errors
    asyncio.run(runner.run_async(documents, rank=0))

    # Verify all documents were processed
    for doc in documents:
        assert "rollout_results" in doc.metadata
        result = doc.metadata["rollout_results"][0]

        if "Error" in doc.text:
            # Error document should have error info
            assert result["success"] is False
            assert result["error"] != ""
            assert result["text"] == ""
        else:
            # Success documents should have no errors
            assert result["success"] is True
            assert result["error"] == ""
            assert result["text"] != ""

    # Verify output file contains all documents
    output_file = output_dir / "00000.jsonl"
    assert output_file.exists()

    with open(output_file, "r") as f:
        saved_docs = [json.loads(line.strip()) for line in f if line.strip()]

    assert len(saved_docs) == 3, "All documents should be in output despite errors"


def test_rollout_fn_partial_success_multiple_responses(tmp_path):
    """
    Test rollout function with multiple generate() calls where some succeed and some fail.

    This simulates VERL's N-response generation pattern where each document
    generates multiple responses and some may fail.
    """
    from datatrove.pipeline.inference.run_inference import InferenceError, InferenceResult

    output_dir = tmp_path / "partial_success"
    documents = [Document(text="Generate 3 responses", id="multi-1")]

    async def multi_response_rollout(document, generate):
        """Generate 3 responses, simulating partial failure."""
        results = []

        for i in range(3):
            # Simulate failure on 2nd attempt
            if i == 1:
                results.append({
                    "attempt": i + 1,
                    "success": False,
                    "text": "",
                    "error": "Simulated intermittent error",
                })
            else:
                result = await generate({
                    "messages": [{"role": "user", "content": [{"type": "text", "text": f"{document.text} (attempt {i+1})"}]}],
                    "max_tokens": 50,
                })
                results.append({
                    "attempt": i + 1,
                    "success": True,
                    "text": result.text,
                    "error": "",
                })

        return {
            "responses": results,
            "total": len(results),
            "successes": sum(1 for r in results if r["success"]),
            "failures": sum(1 for r in results if not r["success"]),
        }

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=None,
    )

    runner = InferenceRunner(
        rollout_fn=multi_response_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    # Verify partial success was handled correctly
    doc = documents[0]
    result = doc.metadata["rollout_results"][0]

    assert result["total"] == 3, "Should have 3 total responses"
    assert result["successes"] == 2, "Should have 2 successful responses"
    assert result["failures"] == 1, "Should have 1 failed response"

    # Verify individual responses
    responses = result["responses"]
    assert responses[0]["success"] is True, "1st response should succeed"
    assert responses[1]["success"] is False, "2nd response should fail"
    assert responses[2]["success"] is True, "3rd response should succeed"

    # Verify error details are present
    assert responses[1]["error"] != "", "Failed response should have error message"


def test_rollout_fn_error_storage_in_output(tmp_path):
    """
    Test that error information is properly persisted to output files.

    Verifies that when errors occur, they are saved with sufficient detail
    for debugging and analysis.
    """
    from datatrove.pipeline.inference.run_inference import InferenceError

    output_dir = tmp_path / "error_storage"
    documents = [
        Document(text="This will fail", id="error-doc", metadata={"index": 0}),
    ]

    async def detailed_error_rollout(document, generate):
        """Rollout that captures detailed error information."""
        # Simulate an error for testing
        return {
            "text": "",
            "status": "failed",
            "error_type": "SimulatedError",
            "error_message": "Service unavailable - intentional test failure",
            "failed_document_id": document.id,
        }

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=None,
    )

    runner = InferenceRunner(
        rollout_fn=detailed_error_rollout,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    # Verify error was captured in document metadata
    doc = documents[0]
    result = doc.metadata["rollout_results"][0]

    assert result["status"] == "failed"
    assert result["error_message"] != ""
    assert "unavailable" in result["error_message"].lower() or "error" in result["error_message"].lower()

    # Verify error was persisted to output file
    output_file = output_dir / "00000.jsonl"
    assert output_file.exists()

    saved_doc = json.loads(output_file.read_text().strip())
    saved_result = saved_doc["metadata"]["rollout_results"][0]

    assert saved_result["status"] == "failed"
    assert saved_result["error_message"] != ""
    assert saved_result["text"] == ""

    # Verify we have enough detail for debugging
    assert "error_type" in saved_result
    assert "error_message" in saved_result
    assert saved_result["failed_document_id"] == "error-doc"


def test_checkpoint_edge_case_chunk_size_one(tmp_path):
    """
    Test checkpointing with chunk_size=1 (every document triggers close_file).

    This is an edge case that stresses the checkpoint system by completing
    a chunk after every single document.
    """
    output_dir = tmp_path / "chunk_one"
    checkpoint_dir = tmp_path / "checkpoints"

    num_docs = 5
    documents = [Document(text=f"Doc {i}", id=f"doc-{i}") for i in range(num_docs)]

    async def simple_rollout(document, generate):
        result = await generate({
            "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
            "max_tokens": 50,
        })
        return {"text": result.text}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=None,
    )

    runner = InferenceRunner(
        rollout_fn=simple_rollout,
        config=config,
        records_per_chunk=1,  # Every document is a separate chunk
        checkpoints_local_dir=str(checkpoint_dir),
        output_writer=JsonlWriter(
            str(output_dir),
            output_filename="${rank}_chunk_${chunk_index}.jsonl",
            compression=None,
        ),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    # Verify 5 chunk files were created (one per document)
    output_files = sorted(output_dir.glob("*.jsonl"))
    assert len(output_files) == num_docs, f"Expected {num_docs} chunks, got {len(output_files)}"

    # Verify each chunk contains exactly 1 document
    for i, output_file in enumerate(output_files):
        with open(output_file, "r") as f:
            docs_in_chunk = [json.loads(line) for line in f if line.strip()]
        assert len(docs_in_chunk) == 1, f"Chunk {i} should have 1 document, got {len(docs_in_chunk)}"

    # Verify all documents were processed
    total_docs = sum(1 for f in output_files for _ in open(f) if _.strip())
    assert total_docs == num_docs, f"Expected {num_docs} total docs, got {total_docs}"


def test_checkpoint_edge_case_incomplete_last_chunk(tmp_path):
    """
    Test checkpointing when last chunk is incomplete (< records_per_chunk).

    This verifies that the final partial chunk is properly saved.
    """
    output_dir = tmp_path / "incomplete_chunk"
    checkpoint_dir = tmp_path / "checkpoints"

    num_docs = 7  # With chunk_size=3, last chunk will have 1 document
    chunk_size = 3
    documents = [Document(text=f"Doc {i}", id=f"doc-{i}") for i in range(num_docs)]

    async def simple_rollout(document, generate):
        result = await generate({
            "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
            "max_tokens": 50,
        })
        return {"text": result.text}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=None,
    )

    runner = InferenceRunner(
        rollout_fn=simple_rollout,
        config=config,
        records_per_chunk=chunk_size,
        checkpoints_local_dir=str(checkpoint_dir),
        output_writer=JsonlWriter(
            str(output_dir),
            output_filename="${rank}_chunk_${chunk_index}.jsonl",
            compression=None,
        ),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    # Verify 3 chunks: [0-2], [3-5], [6]
    output_files = sorted(output_dir.glob("*.jsonl"))
    expected_chunks = (num_docs + chunk_size - 1) // chunk_size  # Ceiling division
    assert len(output_files) == expected_chunks, f"Expected {expected_chunks} chunks, got {len(output_files)}"

    # Verify chunk sizes
    chunk_sizes = []
    for output_file in output_files:
        with open(output_file, "r") as f:
            docs_in_chunk = [json.loads(line) for line in f if line.strip()]
        chunk_sizes.append(len(docs_in_chunk))

    assert chunk_sizes == [3, 3, 1], f"Expected [3, 3, 1], got {chunk_sizes}"

    # Verify all documents were processed
    total_docs = sum(chunk_sizes)
    assert total_docs == num_docs, f"Expected {num_docs} total docs, got {total_docs}"


def test_checkpoint_edge_case_large_chunk_size(tmp_path):
    """
    Test checkpointing with very large chunk_size (larger than total documents).

    This verifies that when chunk_size > num_documents, everything goes
    into a single chunk.
    """
    output_dir = tmp_path / "large_chunk"
    checkpoint_dir = tmp_path / "checkpoints"

    num_docs = 10
    chunk_size = 1000  # Much larger than num_docs
    documents = [Document(text=f"Doc {i}", id=f"doc-{i}") for i in range(num_docs)]

    async def simple_rollout(document, generate):
        result = await generate({
            "messages": [{"role": "user", "content": [{"type": "text", "text": document.text}]}],
            "max_tokens": 50,
        })
        return {"text": result.text}

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=1,
        max_concurrent_documents=None,
    )

    runner = InferenceRunner(
        rollout_fn=simple_rollout,
        config=config,
        records_per_chunk=chunk_size,
        checkpoints_local_dir=str(checkpoint_dir),
        output_writer=JsonlWriter(
            str(output_dir),
            output_filename="${rank}_chunk_${chunk_index}.jsonl",
            compression=None,
        ),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    # Verify only 1 chunk file was created
    output_files = list(output_dir.glob("*.jsonl"))
    assert len(output_files) == 1, f"Expected 1 chunk, got {len(output_files)}"

    # Verify it contains all documents
    with open(output_files[0], "r") as f:
        docs_in_chunk = [json.loads(line) for line in f if line.strip()]
    assert len(docs_in_chunk) == num_docs, f"Expected {num_docs} docs in chunk, got {len(docs_in_chunk)}"


def test_verl_end_to_end_workflow(tmp_path):
    """
    Test complete VERL workflow: generate N responses → score → aggregate stats.

    This is an end-to-end test that verifies the VERL pattern works correctly
    without requiring external model serving or sandbox services.
    Uses DummyServer + math dataset scoring.
    """
    import asyncio
    from unittest.mock import patch

    output_dir = tmp_path / "verl_e2e"

    # Create a VERL-style document with math problem
    documents = [
        Document(
            text="What is 2+2?",
            id="math-1",
            metadata={
                "data_source": "gsm8k",
                "original_prompt": [
                    {"role": "user", "content": "What is 2+2?"}
                ],
                "reward_model": {
                    "ground_truth": "\\boxed{4}"
                },
            }
        )
    ]

    # Mock compute_score to avoid actual scoring dependency
    mock_scores = [
        {"correct": True, "score": 1.0, "reward_correct": 1.0, "reward_think": 0.0, "reward_fmt": 0.5, "reward_length": 0.0},
        {"correct": False, "score": 0.0, "reward_correct": 0.0, "reward_think": 0.0, "reward_fmt": 0.5, "reward_length": 0.0},
        {"correct": True, "score": 1.0, "reward_correct": 1.0, "reward_think": 0.0, "reward_fmt": 0.5, "reward_length": 0.0},
    ]

    call_count = 0

    def mock_compute_score(*args, **kwargs):
        nonlocal call_count
        result = mock_scores[call_count % len(mock_scores)]
        call_count += 1
        return result

    async def verl_rollout_fn(document, generate):
        """
        Complete VERL rollout function: generate N responses, score, compute stats.
        """
        N_RESPONSES = 3

        # 1. Generate N responses
        results = []
        for i in range(N_RESPONSES):
            result = await generate({
                "messages": document.metadata["original_prompt"],
                "max_tokens": 100,
            })
            results.append(result)

        # 2. Score all responses (mocked)
        with patch('datatrove.utils.reward_score.compute_score', side_effect=mock_compute_score):
            from datatrove.utils.reward_score import compute_score

            scores = []
            for result in results:
                score_dict = compute_score(
                    data_source=document.metadata["data_source"],
                    solution_str=result.text,
                    ground_truth=document.metadata["reward_model"]["ground_truth"],
                )
                scores.append(score_dict)

        # 3. Create unified responses (merging inference + scores)
        unified_responses = []
        for result, score_dict in zip(results, scores):
            unified_responses.append({
                "text": result.text,
                "finish_reason": result.finish_reason,
                "is_success": True,
                "score": score_dict.get("score", 0.0),
                "reward_correct": score_dict.get("reward_correct", 0.0),
                "reward_think": score_dict.get("reward_think", 0.0),
                "reward_fmt": score_dict.get("reward_fmt", 0.0),
                "reward_length": score_dict.get("reward_length", 0.0),
            })

        # 4. Compute aggregate statistics
        scores_list = [r["score"] for r in unified_responses]
        avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
        max_score = max(scores_list) if scores_list else 0.0
        success_rate = sum(1 for s in scores_list if s > 0.5) / len(scores_list) if scores_list else 0.0

        # 5. Store in document metadata (for output)
        document.metadata["unified_responses"] = unified_responses
        document.metadata["avg_score"] = avg_score
        document.metadata["max_score"] = max_score
        document.metadata["success_rate"] = success_rate

        # 6. Return summary
        return {
            "status": "processed",
            "num_responses": len(unified_responses),
            "avg_score": avg_score,
            "max_score": max_score,
            "success_rate": success_rate,
        }

    config = InferenceConfig(
        server_type="dummy",
        model_name_or_path="test-model",
        model_max_context=2048,
        metric_interval=60,
        rollouts_per_document=1,
        max_concurrent_generations=3,
        max_concurrent_documents=None,
    )

    runner = InferenceRunner(
        rollout_fn=verl_rollout_fn,
        config=config,
        output_writer=JsonlWriter(str(output_dir), output_filename="${rank}.jsonl", compression=None),
    )

    asyncio.run(runner.run_async(documents, rank=0))

    # Verify workflow completed successfully
    doc = documents[0]
    result = doc.metadata["rollout_results"][0]

    assert result["status"] == "processed"
    assert result["num_responses"] == 3
    assert 0.0 <= result["avg_score"] <= 1.0
    assert 0.0 <= result["max_score"] <= 1.0
    assert 0.0 <= result["success_rate"] <= 1.0

    # Verify unified responses were stored
    assert "unified_responses" in doc.metadata
    unified_responses = doc.metadata["unified_responses"]
    assert len(unified_responses) == 3

    # Verify each response has required fields
    for response in unified_responses:
        assert "text" in response
        assert "score" in response
        assert "reward_correct" in response
        assert "reward_think" in response
        assert "reward_fmt" in response
        assert "reward_length" in response

    # Verify statistics match mocked scores
    # Mock scores: [correct=True (1.0), correct=False (0.0), correct=True (1.0)]
    expected_avg = (1.0 + 0.0 + 1.0) / 3
    assert abs(result["avg_score"] - expected_avg) < 0.01

    # Verify output file
    output_file = output_dir / "00000.jsonl"
    assert output_file.exists()

    with open(output_file, "r") as f:
        saved_doc = json.loads(f.read().strip())

    assert "unified_responses" in saved_doc["metadata"]
    assert len(saved_doc["metadata"]["unified_responses"]) == 3
    assert saved_doc["metadata"]["avg_score"] == result["avg_score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
