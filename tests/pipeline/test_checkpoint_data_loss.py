"""
Tests for checkpoint data loss detection and prevention.

This module contains tests that reproduce the data loss issue where:
- Checkpoint JSONL files correctly save 100 documents
- But Parquet output only contains 0-90 documents due to race conditions

Test strategy:
1. test_OLD_implementation: Uses old asyncio.Lock + aiofiles (SHOULD FAIL/SKIP)
2. test_QUEUE_implementation: Uses new Queue-based approach (SHOULD PASS)
"""

import asyncio
import os
import tempfile
from pathlib import Path
from collections import Counter, defaultdict
import concurrent.futures

import pytest
import pyarrow.parquet as pq

from datatrove.data import Document
from datatrove.pipeline.writers.parquet import ParquetWriter


@pytest.mark.asyncio
async def test_OLD_implementation_causes_data_loss():
    """
    Reproduces data loss with OLD asyncio.Lock + aiofiles implementation.

    This test recreates the buggy behavior to prove the problem exists.
    Expected result: SKIP (successfully demonstrates data loss)
    """
    import aiofiles
    import orjson
    import dataclasses

    # Force small thread pool (2 workers) to dramatically increase race condition probability
    loop = asyncio.get_event_loop()
    small_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(small_executor)

    try:
        # OLD implementation classes (buggy - no thread locks)
        class OldCheckpointManager:
            def __init__(self, checkpoints_local_dir, records_per_chunk):
                self.checkpoints_local_dir = checkpoints_local_dir
                self.records_per_chunk = records_per_chunk
                self.file_locks = defaultdict(asyncio.Lock)  # OLD: asyncio.Lock
                self.per_chunk_counts = Counter()

            async def write_document_old(self, document, rank, chunk_index, output_writer_context):
                """OLD buggy implementation with asyncio.Lock + aiofiles"""
                async with self.file_locks[chunk_index]:  # asyncio.Lock
                    output_writer_context.write(document, rank=rank, chunk_index=chunk_index)
                    self.per_chunk_counts[chunk_index] += 1

                    if self.checkpoints_local_dir is not None:
                        save_path = os.path.join(
                            self.checkpoints_local_dir,
                            f"{rank:05d}/chunk_{chunk_index:05d}.jsonl"
                        )
                        save_dir = os.path.dirname(save_path)

                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)

                        # BUG: aiofiles uses thread pool, not protected by asyncio.Lock
                        async with aiofiles.open(save_path, "ab") as f:
                            await f.write(orjson.dumps(dataclasses.asdict(document), option=orjson.OPT_APPEND_NEWLINE))

                    if self.per_chunk_counts[chunk_index] == self.records_per_chunk:
                        filename = output_writer_context._get_output_filename(
                            document, rank, chunk_index=chunk_index
                        )
                        output_writer_context.close_file(filename)

        class BadParquetWriter(ParquetWriter):
            """ParquetWriter WITHOUT threading.Lock to simulate old buggy behavior"""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if hasattr(self, '_write_lock'):
                    del self._write_lock

            def _write(self, document, file_handler, filename):
                """Override without lock protection"""
                import pyarrow as pa
                import pyarrow.parquet as pq

                # NO LOCK - race condition possible
                if filename not in self._writers:
                    self._writers[filename] = pq.ParquetWriter(
                        file_handler,
                        schema=self.schema if self.schema is not None else pa.RecordBatch.from_pylist([document]).schema,
                        compression=self.compression,
                    )
                self._batches[filename].append(document)
                if len(self._batches[filename]) == self.batch_size:
                    self._write_batch(filename)

            def close_file(self, original_name):
                """Override without lock protection"""
                if original_name in self._batches:
                    self._write_batch(original_name)
                if original_name in self._writers:
                    self._writers.pop(original_name).close()
                from datatrove.pipeline.writers.disk_base import DiskWriter
                DiskWriter.close_file(self, original_name)

            def close(self):
                """Override without lock protection"""
                for filename in list(self._batches.keys()):
                    self._write_batch(filename)
                for writer in self._writers.values():
                    writer.close()
                self._batches.clear()
                self._writers.clear()
                from datatrove.pipeline.writers.disk_base import DiskWriter
                DiskWriter.close(self)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            output_dir = Path(tmpdir) / "output"

            num_docs_to_write = 100
            records_per_chunk = 200

            checkpoint_mgr = OldCheckpointManager(
                checkpoints_local_dir=str(checkpoint_dir),
                records_per_chunk=records_per_chunk
            )

            output_writer = BadParquetWriter(
                output_folder=str(output_dir),
                output_filename="test.parquet",
                batch_size=25,
                schema=None
            )

            test_docs = [
                Document(text=f"Test document {i}", id=f"doc_{i:03d}", metadata={"index": i})
                for i in range(num_docs_to_write)
            ]

            # Simulate concurrent writes
            with output_writer:
                async def write_document_concurrent(doc, task_id):
                    await checkpoint_mgr.write_document_old(
                        doc, rank=0, chunk_index=0,
                        output_writer_context=output_writer
                    )
                    # Increase race probability with delays
                    await asyncio.sleep(0.001 * (task_id % 10))

                tasks = [write_document_concurrent(doc, i) for i, doc in enumerate(test_docs)]
                await asyncio.gather(*tasks)

            # Verify checkpoint JSONL
            checkpoint_file = checkpoint_dir / "00000" / "chunk_00000.jsonl"
            jsonl_lines = sum(1 for _ in open(checkpoint_file))

            # Verify Parquet output
            parquet_files = list(output_dir.glob("*.parquet"))
            if parquet_files:
                tables = [pq.read_table(f) for f in parquet_files]
                parquet_rows = sum(len(t) for t in tables)
            else:
                parquet_rows = 0

            # Results
            print(f"\n=== OLD Implementation Results (No Locks, Small ThreadPool=2) ===")
            print(f"Expected documents: {num_docs_to_write}")
            print(f"Checkpoint JSONL: {jsonl_lines} lines")
            print(f"Parquet output: {parquet_rows} rows")
            loss_pct = (1 - parquet_rows/num_docs_to_write) * 100 if num_docs_to_write > 0 else 0
            print(f"Data loss: {num_docs_to_write - parquet_rows} documents ({loss_pct:.1f}%)")

            # Assert: Checkpoint should be correct
            assert jsonl_lines == num_docs_to_write, \
                f"Checkpoint should have {num_docs_to_write} lines"

            # We expect data loss in old implementation
            if parquet_rows < num_docs_to_write:
                print(f"✓ Bug successfully reproduced: {loss_pct:.1f}% data loss demonstrated")
                pytest.skip(f"OLD implementation correctly shows data loss: {num_docs_to_write - parquet_rows} docs lost")
            else:
                pytest.fail(f"Expected data loss but got 100% integrity - bug not reproduced")

    finally:
        # Restore original executor
        loop.set_default_executor(None)
        small_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_QUEUE_implementation_prevents_data_loss():
    """
    Proves Queue-based implementation has ZERO data loss.

    This test uses the NEW Queue-based CheckpointManager.
    Expected result: PASS (100% data integrity)
    """
    from datatrove.pipeline.inference.run_inference import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        output_dir = Path(tmpdir) / "output"

        num_docs_to_write = 100
        records_per_chunk = 200

        checkpoint_mgr = CheckpointManager(
            checkpoints_local_dir=str(checkpoint_dir),
            records_per_chunk=records_per_chunk
        )

        # Start Queue-based writer
        await checkpoint_mgr.start_writer()

        output_writer = ParquetWriter(
            output_folder=str(output_dir),
            output_filename="test.parquet",
            batch_size=25,
            schema=None
        )

        test_docs = [
            Document(text=f"Test document {i}", id=f"doc_{i:03d}", metadata={"index": i})
            for i in range(num_docs_to_write)
        ]

        # Simulate concurrent writes
        with output_writer:
            async def write_document_concurrent(doc, task_id):
                await checkpoint_mgr.write_document(
                    doc, rank=0, chunk_index=0,
                    output_writer_context=output_writer
                )
                await asyncio.sleep(0.0001 * (task_id % 10))

            tasks = [write_document_concurrent(doc, i) for i, doc in enumerate(test_docs)]
            await asyncio.gather(*tasks)

            # Stop writer BEFORE closing output_writer
            await checkpoint_mgr.stop_writer()

        # Verify checkpoint JSONL
        checkpoint_file = checkpoint_dir / "00000" / "chunk_00000.jsonl"
        jsonl_lines = sum(1 for _ in open(checkpoint_file))

        # Verify Parquet output
        parquet_files = list(output_dir.glob("*.parquet"))
        if parquet_files:
            tables = [pq.read_table(f) for f in parquet_files]
            parquet_rows = sum(len(t) for t in tables)
        else:
            parquet_rows = 0

        # Results
        print(f"\n=== QUEUE Implementation Results ===")
        print(f"Expected documents: {num_docs_to_write}")
        print(f"Checkpoint JSONL: {jsonl_lines} lines")
        print(f"Parquet output: {parquet_rows} rows")
        print(f"Data loss: {num_docs_to_write - parquet_rows} documents (0.0%)")

        # Assert: Checkpoint should be correct
        assert jsonl_lines == num_docs_to_write, \
            f"Checkpoint should have {num_docs_to_write} lines"

        # Assert: Parquet should match JSONL (MUST PASS with Queue implementation)
        assert parquet_rows == num_docs_to_write, \
            f"Data loss detected: {num_docs_to_write} JSONL → {parquet_rows} Parquet"

        # Verify IDs match
        import orjson
        jsonl_ids = set()
        with open(checkpoint_file, 'rb') as f:
            for line in f:
                doc_dict = orjson.loads(line)
                jsonl_ids.add(doc_dict['id'])

        parquet_ids = set()
        for table in tables:
            parquet_ids.update(table['id'].to_pylist())

        assert jsonl_ids == parquet_ids, \
            f"ID mismatch: {len(jsonl_ids)} in JSONL vs {len(parquet_ids)} in Parquet"

        print("✓ Test PASSED: 100% data integrity with Queue-based implementation")


if __name__ == "__main__":
    # Allow running tests directly
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
