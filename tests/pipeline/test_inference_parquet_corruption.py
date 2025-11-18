"""
Test to reproduce and verify Parquet file corruption issue when using InferenceRunner with ParquetWriter.

The issue occurs when chunks are completed (records_per_chunk is reached):
1. InferenceRunner closes the file handler
2. But ParquetWriter's internal state (_batches, _writers) is not properly flushed
3. This results in Parquet files missing their footer (magic bytes, metadata)
4. Such files cannot be read by Parquet readers
"""

import tempfile
from pathlib import Path

import pytest

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.writers import ParquetWriter


def check_parquet_file_validity(file_path):
    """
    Check if a Parquet file is valid by:
    1. Checking for magic bytes (PAR1 at start and end)
    2. Attempting to read it with pyarrow

    Returns: (is_valid, error_message)
    """
    try:
        import pyarrow.parquet as pq

        # Check magic bytes
        with open(file_path, "rb") as f:
            # Parquet files should start with "PAR1"
            header = f.read(4)
            if header != b"PAR1":
                return False, f"Missing magic bytes at start. Got: {header}"

            # Parquet files should end with "PAR1"
            f.seek(-4, 2)  # Seek to 4 bytes before end
            footer = f.read(4)
            if footer != b"PAR1":
                return False, f"Missing magic bytes at end. Got: {footer}"

        # Try to read the file
        table = pq.read_table(file_path)
        num_rows = len(table)

        return True, f"Valid Parquet file with {num_rows} rows"

    except Exception as e:
        return False, f"Error reading Parquet: {str(e)}"


def test_parquet_corruption_with_chunking():
    """
    Test that reproduces the Parquet corruption issue:
    - Uses ParquetWriter as output_writer
    - Sets records_per_chunk to a small value (5) to force chunk completion
    - Processes 20 documents (4 complete chunks) to test multiple chunk closes
    - Verifies that all generated Parquet files are valid and readable

    The bug occurs because CheckpointManager closes files by only calling
    output_mg.pop().close(), which doesn't flush ParquetWriter's internal
    _batches or properly close _writers, resulting in files without magic bytes.
    """
    num_docs = 20
    records_per_chunk = 5  # Force 4 complete chunk closures

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output"
        checkpoint_path = Path(temp_dir) / "checkpoints"
        logs_path = Path(temp_dir) / "logs"

        # Create test documents
        documents = [
            Document(text=f"Test document {i}", id=f"doc_{i}")
            for i in range(num_docs)
        ]

        async def rollout_fn(document, generate):
            result = await generate({
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": document.text}],
                    }
                ],
                "max_tokens": 100,
            })
            return result

        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="test-model",
            default_generation_params={"temperature": 0.0},
            max_concurrent_generations=2,  # Low to ensure sequential processing
            max_concurrent_documents=2,
            metric_interval=120,
        )

        pipeline_executor = LocalPipelineExecutor(
            pipeline=[
                documents,
                InferenceRunner(
                    rollout_fn=rollout_fn,
                    config=config,
                    records_per_chunk=records_per_chunk,
                    checkpoints_local_dir=str(checkpoint_path),
                    output_writer=ParquetWriter(
                        output_folder=str(output_path),
                        output_filename="${rank}_chunk_${chunk_index}.parquet",
                        compression="snappy",
                    ),
                ),
            ],
            logging_dir=str(logs_path),
            tasks=1,
        )

        # Run pipeline
        pipeline_executor.run()

        # === VERIFY PARQUET FILES ===
        parquet_files = sorted(output_path.glob("*_chunk_*.parquet"))

        print(f"\n\nGenerated {len(parquet_files)} Parquet files:")
        for pf in parquet_files:
            print(f"  - {pf.name} ({pf.stat().st_size} bytes)")

        # Note: Due to ParquetWriter's file splitting, we may get more files than expected
        # Let's just verify that we got at least the expected number
        expected_chunks = 3
        assert len(parquet_files) >= expected_chunks, (
            f"Expected at least {expected_chunks} chunk files, got {len(parquet_files)}"
        )

        # Check each file for validity
        all_valid = True
        validation_results = {}

        for parquet_file in parquet_files:
            is_valid, message = check_parquet_file_validity(parquet_file)
            validation_results[parquet_file.name] = (is_valid, message)

            if not is_valid:
                all_valid = False
                print(f"\n❌ INVALID: {parquet_file.name}")
                print(f"   Error: {message}")
            else:
                print(f"\n✓ VALID: {parquet_file.name}")
                print(f"   {message}")

        # This assertion will FAIL if the bug exists
        assert all_valid, (
            f"Some Parquet files are corrupted:\n" +
            "\n".join(
                f"  - {name}: {msg}"
                for name, (valid, msg) in validation_results.items()
                if not valid
            )
        )

        # === VERIFY CONTENT ===
        # Try to read and verify all documents are present
        import pyarrow.parquet as pq

        all_docs = []
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            all_docs.extend(table.to_pylist())

        assert len(all_docs) == num_docs, (
            f"Expected {num_docs} documents total, got {len(all_docs)}"
        )

        # Verify all document IDs are present
        processed_ids = {doc["id"] for doc in all_docs}
        expected_ids = {f"doc_{i}" for i in range(num_docs)}
        assert processed_ids == expected_ids, (
            f"Missing IDs: {expected_ids - processed_ids}"
        )


def test_parquet_corruption_with_batch_size():
    """
    Test with different batch_size to ensure the issue is not related to batching.
    ParquetWriter has internal batching (default 1000), so we test with batch_size < records_per_chunk.
    """
    num_docs = 20
    records_per_chunk = 10
    batch_size = 3  # Smaller than records_per_chunk

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output"
        checkpoint_path = Path(temp_dir) / "checkpoints"
        logs_path = Path(temp_dir) / "logs"

        documents = [
            Document(text=f"Test document {i}", id=f"doc_{i}")
            for i in range(num_docs)
        ]

        async def rollout_fn(document, generate):
            result = await generate({
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": document.text}],
                    }
                ],
                "max_tokens": 100,
            })
            return result

        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="test-model",
            default_generation_params={"temperature": 0.0},
            max_concurrent_generations=2,
            max_concurrent_documents=2,
            metric_interval=120,
        )

        pipeline_executor = LocalPipelineExecutor(
            pipeline=[
                documents,
                InferenceRunner(
                    rollout_fn=rollout_fn,
                    config=config,
                    records_per_chunk=records_per_chunk,
                    checkpoints_local_dir=str(checkpoint_path),
                    output_writer=ParquetWriter(
                        output_folder=str(output_path),
                        output_filename="${rank}_chunk_${chunk_index}.parquet",
                        compression="snappy",
                        batch_size=batch_size,  # Test with small batch size
                    ),
                ),
            ],
            logging_dir=str(logs_path),
            tasks=1,
        )

        pipeline_executor.run()

        # Check all generated files
        parquet_files = sorted(output_path.glob("*_chunk_*.parquet"))

        for parquet_file in parquet_files:
            is_valid, message = check_parquet_file_validity(parquet_file)
            assert is_valid, f"File {parquet_file.name} is corrupted: {message}"


def test_parquet_file_split_by_size():
    """
    Test that verifies file splitting behavior when max_file_size is reached.
    This tests both chunking (records_per_chunk) and file size splitting (max_file_size).
    """
    num_docs = 20
    records_per_chunk = 10
    max_file_size = 5000  # Very small to force file splits

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output"
        checkpoint_path = Path(temp_dir) / "checkpoints"
        logs_path = Path(temp_dir) / "logs"

        documents = [
            Document(
                text=f"Test document {i} " * 100,  # Make documents larger
                id=f"doc_{i}",
                metadata={"index": i}
            )
            for i in range(num_docs)
        ]

        async def rollout_fn(document, generate):
            result = await generate({
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": document.text}],
                    }
                ],
                "max_tokens": 100,
            })
            return result

        config = InferenceConfig(
            server_type="dummy",
            model_name_or_path="test-model",
            default_generation_params={"temperature": 0.0},
            max_concurrent_generations=2,
            max_concurrent_documents=2,
            metric_interval=120,
        )

        pipeline_executor = LocalPipelineExecutor(
            pipeline=[
                documents,
                InferenceRunner(
                    rollout_fn=rollout_fn,
                    config=config,
                    records_per_chunk=records_per_chunk,
                    checkpoints_local_dir=str(checkpoint_path),
                    output_writer=ParquetWriter(
                        output_folder=str(output_path),
                        output_filename="${rank}_chunk_${chunk_index}.parquet",
                        compression="snappy",
                        max_file_size=max_file_size,
                    ),
                ),
            ],
            logging_dir=str(logs_path),
            tasks=1,
        )

        pipeline_executor.run()

        # Check all generated files (including 000_, 001_ prefixed files)
        parquet_files = sorted(output_path.glob("*.parquet"))

        print(f"\nGenerated {len(parquet_files)} Parquet files:")
        for pf in parquet_files:
            print(f"  - {pf.name} ({pf.stat().st_size} bytes)")

        # Verify all files are valid
        for parquet_file in parquet_files:
            is_valid, message = check_parquet_file_validity(parquet_file)
            assert is_valid, f"File {parquet_file.name} is corrupted: {message}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
