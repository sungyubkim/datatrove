"""
Test to reproduce Parquet corruption during pipeline execution (not after completion).

This simulates checking Parquet files WHILE the pipeline is still running,
which is when users typically encounter the corruption issue.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import (
    CheckpointManager,
    InferenceConfig,
    InferenceRunner,
)
from datatrove.pipeline.writers import ParquetWriter


def check_parquet_file_validity(file_path):
    """Check if a Parquet file has valid magic bytes and can be read."""
    try:
        import pyarrow.parquet as pq

        with open(file_path, "rb") as f:
            header = f.read(4)
            if header != b"PAR1":
                return False, f"Missing magic bytes at start. Got: {header!r}"

            f.seek(-4, 2)
            footer = f.read(4)
            if footer != b"PAR1":
                return False, f"Missing magic bytes at end. Got: {footer!r}"

        table = pq.read_table(file_path)
        return True, f"Valid with {len(table)} rows"

    except Exception as e:
        return False, f"Error: {str(e)}"


def test_parquet_corruption_during_execution():
    """
    Directly test CheckpointManager's file closing behavior with ParquetWriter.

    This reproduces the exact scenario:
    1. ParquetWriter opens and writes to files
    2. CheckpointManager closes files using output_mg.pop().close()
    3. Files are checked BEFORE the writer context exits
    """
    num_docs = 20
    records_per_chunk = 5

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output"
        checkpoint_path = Path(temp_dir) / "checkpoints"

        # Create documents
        documents = [
            Document(text=f"Test document {i}", id=f"doc_{i}", metadata={})
            for i in range(num_docs)
        ]

        # Create ParquetWriter
        writer = ParquetWriter(
            output_folder=str(output_path),
            output_filename="${rank}_chunk_${chunk_index}.parquet",
            compression="snappy",
        )

        # Create CheckpointManager
        checkpoint_manager = CheckpointManager(
            checkpoints_local_dir=str(checkpoint_path),
            records_per_chunk=records_per_chunk,
        )

        chunk_index_gen = checkpoint_manager.chunk_index_gen()

        # Simulate InferenceRunner's behavior - WITHIN the writer context
        with writer:
            for i, doc in enumerate(documents):
                chunk_index = next(chunk_index_gen)

                # Write document
                writer.write(doc, rank=0, chunk_index=chunk_index)
                checkpoint_manager.per_chunk_counts[chunk_index] += 1

                # Check if chunk is complete
                if checkpoint_manager.per_chunk_counts[chunk_index] == records_per_chunk:
                    print(f"\n🔄 Chunk {chunk_index} complete ({records_per_chunk} docs)")

                    filename = writer._get_output_filename(doc, rank=0, chunk_index=chunk_index)
                    print(f"   Closing file: {filename}")

                    # THIS IS THE BUG: Only closes file handler, not ParquetWriter internals
                    writer.output_mg.pop(filename).close()

                    # Check file immediately after "closing" (simulating user checking mid-execution)
                    parquet_files = list(output_path.glob("*.parquet"))
                    print(f"   Files on disk: {len(parquet_files)}")

                    for pf in parquet_files:
                        if f"chunk_{chunk_index}" in pf.name:
                            is_valid, msg = check_parquet_file_validity(pf)
                            print(f"   {pf.name}: {'✓' if is_valid else '❌'} {msg}")

                            if not is_valid:
                                print(f"\n⚠️  CORRUPTION DETECTED: {pf.name}")
                                print(f"   This file was 'closed' but is still corrupt!")
                                print(f"   Size: {pf.stat().st_size} bytes")

            # After loop completes, check all files BEFORE context exit
            print(f"\n\n📊 Final check BEFORE writer.close():")
            all_files = sorted(output_path.glob("*.parquet"))

            corruption_found = False
            for pf in all_files:
                is_valid, msg = check_parquet_file_validity(pf)
                status = "✓ VALID" if is_valid else "❌ CORRUPT"
                print(f"   {status}: {pf.name} - {msg}")
                if not is_valid:
                    corruption_found = True

            assert corruption_found, (
                "Expected to find corrupted files, but all files are valid. "
                "This means the bug might be fixed or test conditions are wrong."
            )

        # After context exit, check again
        print(f"\n\n📊 After writer.close():")
        all_files = sorted(output_path.glob("*.parquet"))

        for pf in all_files:
            is_valid, msg = check_parquet_file_validity(pf)
            status = "✓ VALID" if is_valid else "❌ CORRUPT"
            print(f"   {status}: {pf.name} - {msg}")


async def test_async_checkpoint_scenario():
    """
    Test the async scenario that happens in actual InferenceRunner.
    This simulates concurrent document processing and chunk completion.
    """
    num_docs = 15
    records_per_chunk = 5

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output"
        checkpoint_path = Path(temp_dir) / "checkpoints"

        writer = ParquetWriter(
            output_folder=str(output_path),
            output_filename="${rank}_chunk_${chunk_index}.parquet",
            compression="snappy",
        )

        checkpoint_manager = CheckpointManager(
            checkpoints_local_dir=str(checkpoint_path),
            records_per_chunk=records_per_chunk,
        )

        documents = [
            Document(text=f"Test {i}", id=f"doc_{i}", metadata={})
            for i in range(num_docs)
        ]

        chunk_index_gen = checkpoint_manager.chunk_index_gen()

        async def process_doc(doc, chunk_idx):
            """Simulate async document processing."""
            await asyncio.sleep(0.01)  # Simulate some async work
            return doc, chunk_idx

        with writer:
            chunk_idx = -1
            for i, doc in enumerate(documents):
                chunk_idx = next(chunk_index_gen)

                # Process document
                processed_doc, ci = await process_doc(doc, chunk_idx)

                # Write using CheckpointManager's write_document method
                await checkpoint_manager.write_document(
                    processed_doc, rank=0, chunk_index=ci, output_writer_context=writer
                )

            # Check files mid-execution
            print(f"\n📊 Files during execution:")
            all_files = sorted(output_path.glob("*.parquet"))

            corrupted_files = []
            for pf in all_files:
                is_valid, msg = check_parquet_file_validity(pf)
                status = "✓" if is_valid else "❌"
                print(f"   {status} {pf.name}: {msg}")
                if not is_valid:
                    corrupted_files.append((pf.name, msg))

            if corrupted_files:
                print(f"\n⚠️  Found {len(corrupted_files)} corrupted file(s):")
                for name, msg in corrupted_files:
                    print(f"   - {name}: {msg}")

            # This should find corruption
            assert len(corrupted_files) > 0, "Expected corrupted files during execution"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
