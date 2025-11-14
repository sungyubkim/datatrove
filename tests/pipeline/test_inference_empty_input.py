"""
Test that InferenceRunner properly handles empty input (0 documents).

This edge case occurs when:
- Number of tasks > number of input files
- Some tasks get no files to process (empty shard)

Expected behavior:
- Tasks with 0 documents should terminate gracefully
- No NameError or hanging
- Proper cleanup of resources (queues, thread pools, files)
"""

import tempfile
from pathlib import Path

import pytest

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.writers import JsonlWriter


class SimpleQueryBuilder:
    """Minimal query builder for testing"""

    def __call__(self, runner, document):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": document.text},
                    ],
                }
            ],
            "max_tokens": 10,
        }


def test_empty_input_no_hang():
    """
    Test that InferenceRunner handles empty input (0 documents) gracefully.

    This test verifies the fix for the bug where chunk_index was undefined
    when no documents were processed, causing tasks to hang.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output"
        checkpoint_path = Path(temp_dir) / "checkpoints"
        logs_path = Path(temp_dir) / "logs"

        # Empty document list - simulates a task with no files to process
        documents = []

        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="test-model",
            temperature=0.0,
            model_max_context=8192,
            max_concurrent_requests=2,
            max_concurrent_tasks=2,
            metric_interval=120,
        )

        query_builder = SimpleQueryBuilder()

        pipeline_executor = LocalPipelineExecutor(
            pipeline=[
                documents,  # Empty list!
                InferenceRunner(
                    query_builder=query_builder,
                    config=config,
                    records_per_chunk=10,
                    checkpoints_local_dir=str(checkpoint_path),
                    output_writer=JsonlWriter(
                        output_folder=str(output_path),
                        output_filename="${rank}_chunk_${chunk_index}.jsonl",
                    ),
                ),
            ],
            logging_dir=str(logs_path),
            tasks=1,
            workers=1,
        )

        # This should complete without hanging or errors
        pipeline_executor.run()

        # Verify no output files were created (since no documents were processed)
        output_files = list(output_path.glob("*.jsonl")) if output_path.exists() else []
        assert len(output_files) == 0, "No output files should be created for empty input"

        # Verify checkpoint directory exists but is empty (or only has metadata)
        if checkpoint_path.exists():
            checkpoint_files = list(checkpoint_path.glob("*.json"))
            # Only metadata files (if any) should exist, no actual checkpoint data
            # Since no documents were processed, there should be no checkpoint files
            assert all(
                "metadata" in f.name or "progress" in f.name for f in checkpoint_files
            ), "Only metadata files should exist for empty input"


def test_multiple_tasks_with_fewer_files():
    """
    Test the realistic scenario: More tasks than input files.

    This simulates the user's actual use case where some tasks get 0 files
    due to sharding.

    Example: 3 input files, 5 tasks
    - Task 0: processes file 0
    - Task 1: processes file 1
    - Task 2: processes file 2
    - Task 3: empty (no files) - should terminate gracefully
    - Task 4: empty (no files) - should terminate gracefully
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input"
        output_path = Path(temp_dir) / "output"
        checkpoint_path = Path(temp_dir) / "checkpoints"
        logs_path = Path(temp_dir) / "logs"
        input_path.mkdir()

        # Create 3 input files (one document per file)
        num_files = 3
        num_tasks = 5

        import json

        for i in range(num_files):
            input_file = input_path / f"doc_{i}.jsonl"
            with open(input_file, "w") as f:
                doc_data = {
                    "text": f"Test document {i}",
                    "id": str(i),
                }
                f.write(json.dumps(doc_data) + "\n")

        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="test-model",
            temperature=0.0,
            model_max_context=8192,
            max_concurrent_requests=2,
            max_concurrent_tasks=2,
            metric_interval=120,
        )

        query_builder = SimpleQueryBuilder()

        from datatrove.pipeline.readers import JsonlReader

        pipeline_executor = LocalPipelineExecutor(
            pipeline=[
                JsonlReader(str(input_path), glob_pattern="*.jsonl"),
                InferenceRunner(
                    query_builder=query_builder,
                    config=config,
                    output_writer=JsonlWriter(
                        output_folder=str(output_path),
                        output_filename="${rank}.jsonl",  # One file per rank
                    ),
                    # No checkpoints - direct write to output
                ),
            ],
            logging_dir=str(logs_path),
            tasks=num_tasks,  # More tasks than files
            workers=2,
        )

        # This should complete without hanging
        # Tasks 3 and 4 will have empty input and should terminate gracefully
        pipeline_executor.run()

        # Main verification: All tasks completed successfully without hanging
        # This is the critical fix - tasks with no input files should terminate
        # gracefully instead of hanging indefinitely
        completion_files = sorted(logs_path.glob("completions/*"))
        assert len(completion_files) == num_tasks, f"Expected {num_tasks} completion markers, got {len(completion_files)}"

        # Verify logs show proper termination for empty tasks
        for rank in [3, 4]:  # These ranks should have no files
            log_file = logs_path / "logs" / f"task_{rank:05d}.log"
            if log_file.exists():
                log_content = log_file.read_text()
                # Should see either "No files found" or successful completion
                assert (
                    "No files found" in log_content or "Processing done" in log_content
                ), f"Rank {rank} should terminate gracefully"


if __name__ == "__main__":
    # Run tests manually for debugging
    test_empty_input_no_hang()
    print("✓ test_empty_input_no_hang passed")

    test_multiple_tasks_with_fewer_files()
    print("✓ test_multiple_tasks_with_fewer_files passed")

    print("\nAll tests passed!")
