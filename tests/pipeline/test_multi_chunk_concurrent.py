"""
Integration test for multi-chunk concurrent processing.

This test verifies that multiple chunks can be processed concurrently
with out-of-order completion without data loss.

Scenario:
- 5 chunks (0-4), 50 documents each = 250 total
- Documents shuffled to ensure chunks complete out of order
- High concurrency to trigger potential race conditions
"""

import asyncio
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import pyarrow.parquet as pq

from datatrove.data import Document
from datatrove.pipeline.writers.parquet import ParquetWriter


@pytest.mark.asyncio
@pytest.mark.parametrize("use_buggy_logic", [True, False])
async def test_multiple_chunks_concurrent_writes(use_buggy_logic):
    """
    Test multiple chunks being written concurrently with out-of-order completion.

    Args:
        use_buggy_logic: If True, use assigned==completed check (BUGGY)
                        If False, use len(written)==records_per_chunk (FIXED)
    """

    class TestableCheckpointManager:
        """Minimal CheckpointManager for testing."""

        def __init__(self, use_buggy_logic, records_per_chunk):
            self.use_buggy_logic = use_buggy_logic
            self.records_per_chunk = records_per_chunk

            self.chunk_assigned = defaultdict(set)
            self.chunk_completed = defaultdict(set)
            self.chunk_written = defaultdict(set)
            self.closed_chunks = set()
            self.per_chunk_counts = Counter()

            self.write_queue = None
            self.writer_task = None

        async def start_writer(self):
            self.write_queue = asyncio.Queue()
            self.writer_task = asyncio.create_task(self.checkpoint_writer_task())

        async def stop_writer(self):
            await self.write_queue.put(None)
            await self.write_queue.join()
            if self.writer_task:
                await self.writer_task

        async def checkpoint_writer_task(self):
            while True:
                item = await self.write_queue.get()

                if item is None:
                    self.write_queue.task_done()
                    break

                try:
                    doc, rank, chunk_idx, writer = item

                    writer.write(doc, rank=rank, chunk_index=chunk_idx)
                    self.chunk_written[chunk_idx].add(doc.id)
                    self.chunk_completed[chunk_idx].add(doc.id)
                    self.per_chunk_counts[chunk_idx] += 1

                    should_close = False

                    if self.use_buggy_logic:
                        assigned = self.chunk_assigned[chunk_idx]
                        completed = self.chunk_completed[chunk_idx]
                        if len(assigned) == self.records_per_chunk and assigned == completed:
                            should_close = True
                    else:
                        written = self.chunk_written[chunk_idx]
                        if len(written) == self.records_per_chunk:
                            should_close = True

                    if should_close and chunk_idx not in self.closed_chunks:
                        filename = writer._get_output_filename(doc, rank, chunk_index=chunk_idx)
                        writer.close_file(filename)
                        self.closed_chunks.add(chunk_idx)

                except Exception as e:
                    print(f"Error in checkpoint writer: {e}")
                    raise
                finally:
                    self.write_queue.task_done()

        def assign_document(self, doc_id, chunk_idx):
            self.chunk_assigned[chunk_idx].add(doc_id)

        async def write_document(self, doc, rank, chunk_idx, writer):
            await self.write_queue.put((doc, rank, chunk_idx, writer))

    # Test setup
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        records_per_chunk = 50
        num_chunks = 5
        total_docs = records_per_chunk * num_chunks  # 250

        manager = TestableCheckpointManager(
            use_buggy_logic=use_buggy_logic,
            records_per_chunk=records_per_chunk
        )

        writer = ParquetWriter(
            output_folder=str(output_dir),
            output_filename="${rank}_chunk_${chunk_index}.parquet",
            batch_size=50,  # Match records_per_chunk
            schema=None
        )

        # Generate documents with chunk assignment
        docs_by_chunk = []
        for chunk_idx in range(num_chunks):
            for i in range(records_per_chunk):
                doc_id = chunk_idx * records_per_chunk + i
                doc = Document(
                    text=f"Chunk {chunk_idx} Doc {i}" * 10,
                    id=f"doc_{doc_id:04d}",
                    metadata={"chunk": chunk_idx, "index": i}
                )
                docs_by_chunk.append((doc, chunk_idx))

        # Shuffle to ensure out-of-order processing
        # This is key: chunks will complete in random order
        random.shuffle(docs_by_chunk)

        await manager.start_writer()

        with writer:
            # Pre-assign all documents
            for doc, chunk_idx in docs_by_chunk:
                manager.assign_document(doc.id, chunk_idx)

            # Process with random delays
            async def process_doc(doc, chunk_idx, delay):
                await asyncio.sleep(delay)
                await manager.write_document(doc, rank=0, chunk_idx=chunk_idx, writer=writer)

            # Vary delays to create different completion patterns
            # Some chunks will finish before others
            tasks = [
                asyncio.create_task(
                    process_doc(doc, chunk_idx, random.uniform(0, 0.1))
                )
                for doc, chunk_idx in docs_by_chunk
            ]

            await asyncio.gather(*tasks)
            await manager.stop_writer()

        # Verify each chunk
        chunk_results = {}
        total_actual_rows = 0

        for chunk_idx in range(num_chunks):
            # File might have 000_ prefix, so use glob
            parquet_files = list(output_dir.glob(f"*chunk_{chunk_idx}.parquet"))

            if not parquet_files:
                chunk_results[chunk_idx] = 0
            else:
                table = pq.read_table(parquet_files[0])
                actual = len(table)
                chunk_results[chunk_idx] = actual
                total_actual_rows += actual

        # Print results
        print(f"\n=== Multi-Chunk Test ({'BUGGY' if use_buggy_logic else 'FIXED'} logic) ===")
        print(f"Total expected: {total_docs} ({num_chunks} chunks × {records_per_chunk} docs)")
        print(f"Total actual: {total_actual_rows}")
        print("\nPer-chunk results:")
        for chunk_idx in range(num_chunks):
            actual = chunk_results[chunk_idx]
            status = "✓" if actual == records_per_chunk else "✗"
            print(f"  Chunk {chunk_idx}: {actual}/{records_per_chunk} rows {status}")

        # Check results
        if use_buggy_logic:
            # With buggy logic, expect at least one chunk to have data loss
            chunks_with_loss = [
                chunk_idx for chunk_idx, actual in chunk_results.items()
                if actual < records_per_chunk
            ]

            if not chunks_with_loss:
                pytest.skip(
                    f"Bug not reproduced! All {num_chunks} chunks complete. "
                    f"Try running multiple times or increasing concurrency."
                )
            else:
                total_loss = total_docs - total_actual_rows
                pytest.xfail(
                    f"EXPECTED FAILURE (buggy logic): "
                    f"{len(chunks_with_loss)} chunks with data loss "
                    f"(total {total_loss} docs lost). "
                    f"Affected chunks: {chunks_with_loss}"
                )
        else:
            # With fixed logic, all chunks should be complete
            for chunk_idx in range(num_chunks):
                actual = chunk_results[chunk_idx]
                assert actual == records_per_chunk, (
                    f"Chunk {chunk_idx} failed: expected {records_per_chunk}, got {actual}"
                )

            assert total_actual_rows == total_docs, (
                f"Total mismatch: expected {total_docs}, got {total_actual_rows}"
            )

            print("✓ SUCCESS: All chunks saved correctly with fixed logic")


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
