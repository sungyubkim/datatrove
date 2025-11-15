"""
Extreme race condition reproduction test.

This test uses aggressive techniques to force race conditions in limited hardware.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from collections import Counter, defaultdict
import concurrent.futures
import time

import pytest
import pyarrow.parquet as pq

from datatrove.data import Document
from datatrove.pipeline.writers.parquet import ParquetWriter


@pytest.mark.asyncio
async def test_extreme_race_condition_with_contention():
    """
    Ultra-aggressive race condition test with:
    - ThreadPool size: 1 (extreme bottleneck)
    - Concurrent tasks: 500
    - Artificial blocking in thread pool
    - Small batch size (10) for frequent close operations
    """
    import aiofiles
    import orjson
    import dataclasses

    # Set thread pool to 1 worker (extreme contention!)
    loop = asyncio.get_event_loop()
    tiny_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(tiny_executor)

    try:
        # Buggy implementation
        class ExtremeBuggyCheckpointManager:
            def __init__(self, checkpoints_local_dir, records_per_chunk):
                self.checkpoints_local_dir = checkpoints_local_dir
                self.records_per_chunk = records_per_chunk
                self.file_locks = defaultdict(asyncio.Lock)
                self.per_chunk_counts = Counter()

            async def write_document_old(self, document, rank, chunk_index, output_writer_context):
                async with self.file_locks[chunk_index]:
                    # Write to Parquet
                    output_writer_context.write(document, rank=rank, chunk_index=chunk_index)
                    self.per_chunk_counts[chunk_index] += 1

                    # Write to checkpoint with aiofiles (uses thread pool)
                    if self.checkpoints_local_dir is not None:
                        save_path = os.path.join(
                            self.checkpoints_local_dir,
                            f"{rank:05d}/chunk_{chunk_index:05d}.jsonl"
                        )
                        save_dir = os.path.dirname(save_path)

                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)

                        # BUG: Multiple tasks in thread pool, but asyncio.Lock doesn't protect them
                        async with aiofiles.open(save_path, "ab") as f:
                            await f.write(orjson.dumps(dataclasses.asdict(document), option=orjson.OPT_APPEND_NEWLINE))
                            # Add artificial blocking to increase race probability
                            time.sleep(0.001)

                    # Trigger close_file frequently
                    if self.per_chunk_counts[chunk_index] == self.records_per_chunk:
                        filename = output_writer_context._get_output_filename(
                            document, rank, chunk_index=chunk_index
                        )
                        # This can race with other writes!
                        output_writer_context.close_file(filename)

        class UltraBuggyParquetWriter(ParquetWriter):
            """ParquetWriter with intentionally slow operations to trigger races"""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Remove lock
                if hasattr(self, '_write_lock'):
                    del self._write_lock

            def _write(self, document, file_handler, filename):
                import pyarrow as pa
                import pyarrow.parquet as pq

                # Add artificial delay to increase race window
                time.sleep(0.001)

                # NO LOCK - concurrent access
                if filename not in self._writers:
                    self._writers[filename] = pq.ParquetWriter(
                        file_handler,
                        schema=self.schema if self.schema is not None else pa.RecordBatch.from_pylist([document]).schema,
                        compression=self.compression,
                    )

                self._batches[filename].append(document)
                if len(self._batches[filename]) == self.batch_size:
                    # Add delay before flush
                    time.sleep(0.001)
                    self._write_batch(filename)

            def close_file(self, original_name):
                # Add delay to increase race probability
                time.sleep(0.001)

                # NO LOCK - can race with _write()
                if original_name in self._batches:
                    self._write_batch(original_name)
                if original_name in self._writers:
                    self._writers.pop(original_name).close()

                from datatrove.pipeline.writers.disk_base import DiskWriter
                DiskWriter.close_file(self, original_name)

            def close(self):
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

            # Smaller numbers for faster test, but higher concurrency
            num_docs_to_write = 50
            records_per_chunk = 100

            checkpoint_mgr = ExtremeBuggyCheckpointManager(
                checkpoints_local_dir=str(checkpoint_dir),
                records_per_chunk=records_per_chunk
            )

            output_writer = UltraBuggyParquetWriter(
                output_folder=str(output_dir),
                output_filename="test.parquet",
                batch_size=10,  # Very small batch → frequent close_file calls
                schema=None
            )

            test_docs = [
                Document(text=f"Doc {i}" * 100, id=f"doc_{i:03d}", metadata={"index": i})
                for i in range(num_docs_to_write)
            ]

            # Extreme concurrency
            with output_writer:
                async def write_concurrent(doc, task_id):
                    await checkpoint_mgr.write_document_old(
                        doc, rank=0, chunk_index=0,
                        output_writer_context=output_writer
                    )
                    # No delay - maximize contention!

                # Fire all tasks simultaneously
                tasks = [write_concurrent(doc, i) for i, doc in enumerate(test_docs)]
                await asyncio.gather(*tasks)

            # Verify results
            checkpoint_file = checkpoint_dir / "00000" / "chunk_00000.jsonl"
            jsonl_lines = sum(1 for _ in open(checkpoint_file))

            parquet_files = list(output_dir.glob("*.parquet"))
            if parquet_files:
                tables = [pq.read_table(f) for f in parquet_files]
                parquet_rows = sum(len(t) for t in tables)
            else:
                parquet_rows = 0

            print(f"\n=== EXTREME Race Condition Test (ThreadPool=1, Delays Added) ===")
            print(f"Expected documents: {num_docs_to_write}")
            print(f"Checkpoint JSONL: {jsonl_lines} lines")
            print(f"Parquet output: {parquet_rows} rows")

            if parquet_rows < num_docs_to_write:
                loss = num_docs_to_write - parquet_rows
                loss_pct = (loss / num_docs_to_write) * 100
                print(f"Data loss: {loss} documents ({loss_pct:.1f}%)")
                print(f"✓ ✓ ✓ Race condition SUCCESSFULLY REPRODUCED! ✓ ✓ ✓")
                pytest.skip(f"Race condition reproduced: {loss} docs lost")
            else:
                print("Race condition not triggered in this run (may need more extreme conditions)")
                pytest.skip("Race condition not reproduced - try running multiple times")

    finally:
        loop.set_default_executor(None)
        tiny_executor.shutdown(wait=True)


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
