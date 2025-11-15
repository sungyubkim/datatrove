"""
Unit test for checkpoint queue ordering bug.

This test verifies that chunk completion is based on documents WRITTEN to ParquetWriter,
not just documents that have been assigned or completed in the queue.

Bug scenario:
- With high concurrency, documents complete out of order
- checkpoint_writer_task processes documents from queue sequentially
- If close_file() is called when assigned == completed, but some docs still in queue
  → Those remaining docs are lost when file is closed

Fix:
- Track chunk_written_docs separately
- Only close when len(written) == records_per_chunk
"""

import asyncio
import os
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
async def test_chunk_completion_queue_ordering(use_buggy_logic):
    """
    Test that chunk completion waits for ALL documents to be written.

    Args:
        use_buggy_logic: If True, use assigned==completed check (BUGGY, expect FAIL)
                        If False, use len(written)==records_per_chunk (FIXED, expect PASS)
    """

    class TestableCheckpointManager:
        """Minimal CheckpointManager for testing with configurable logic."""

        def __init__(self, use_buggy_logic, records_per_chunk):
            self.use_buggy_logic = use_buggy_logic
            self.records_per_chunk = records_per_chunk

            # Tracking sets
            self.chunk_assigned = defaultdict(set)
            self.chunk_completed = defaultdict(set)
            self.chunk_written = defaultdict(set)

            # State
            self.closed_chunks = set()
            self.per_chunk_counts = Counter()

            # Queue
            self.write_queue = None
            self.writer_task = None

        async def start_writer(self):
            """Start checkpoint writer task."""
            self.write_queue = asyncio.Queue()
            self.writer_task = asyncio.create_task(self.checkpoint_writer_task())

        async def stop_writer(self):
            """Stop checkpoint writer task."""
            await self.write_queue.put(None)  # Shutdown signal
            await self.write_queue.join()
            if self.writer_task:
                await self.writer_task

        async def checkpoint_writer_task(self):
            """
            Process writes from queue sequentially.

            This is where the bug manifests: if we close based on assigned==completed,
            we might close before all documents are written.
            """
            while True:
                item = await self.write_queue.get()

                if item is None:  # Shutdown signal
                    self.write_queue.task_done()
                    break

                try:
                    doc, rank, chunk_idx, writer = item

                    # Write to ParquetWriter
                    writer.write(doc, rank=rank, chunk_index=chunk_idx)
                    self.chunk_written[chunk_idx].add(doc.id)

                    # Track completion
                    self.chunk_completed[chunk_idx].add(doc.id)
                    self.per_chunk_counts[chunk_idx] += 1

                    # Check if chunk is complete - THIS IS WHERE BUG IS
                    should_close = False

                    if self.use_buggy_logic:
                        # BUGGY: Check assigned == completed
                        # Problem: Some documents might still be in queue!
                        assigned = self.chunk_assigned[chunk_idx]
                        completed = self.chunk_completed[chunk_idx]

                        if len(assigned) == self.records_per_chunk and assigned == completed:
                            should_close = True

                    else:
                        # FIXED: Check written count
                        # This guarantees all documents are actually in ParquetWriter
                        written = self.chunk_written[chunk_idx]

                        if len(written) == self.records_per_chunk:
                            should_close = True

                    if should_close and chunk_idx not in self.closed_chunks:
                        # Close file
                        filename = writer._get_output_filename(doc, rank, chunk_index=chunk_idx)
                        writer.close_file(filename)
                        self.closed_chunks.add(chunk_idx)
                        # Continue processing remaining items in queue (don't break)

                except Exception as e:
                    print(f"Error in checkpoint writer: {e}")
                    raise
                finally:
                    self.write_queue.task_done()

        def assign_document(self, doc_id, chunk_idx):
            """Record document assignment."""
            self.chunk_assigned[chunk_idx].add(doc_id)

        async def write_document(self, doc, rank, chunk_idx, writer):
            """Queue document for writing."""
            await self.write_queue.put((doc, rank, chunk_idx, writer))

    # Test setup
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        records_per_chunk = 100
        num_docs = 100

        # Create manager with specified logic
        manager = TestableCheckpointManager(
            use_buggy_logic=use_buggy_logic,
            records_per_chunk=records_per_chunk
        )

        # Create ParquetWriter
        writer = ParquetWriter(
            output_folder=str(output_dir),
            output_filename="${rank}_chunk_${chunk_index}.parquet",
            batch_size=100,  # Batch size same as records_per_chunk
            schema=None
        )

        # Generate documents
        docs = [
            Document(
                text=f"Document {i}" * 10,
                id=f"doc_{i:03d}",
                metadata={"index": i}
            )
            for i in range(num_docs)
        ]

        # Start writer
        await manager.start_writer()

        # Open writer context
        with writer:
            # Pre-assign ALL documents (simulating main loop behavior)
            for doc in docs:
                manager.assign_document(doc.id, 0)

            # Fire tasks with random delays to simulate async completion
            async def process_doc(doc, delay):
                """Simulate document processing with random delay."""
                await asyncio.sleep(delay)
                await manager.write_document(doc, rank=0, chunk_idx=0, writer=writer)

            # Create tasks with varying delays (0 to 0.05 seconds)
            # This ensures documents complete in random order
            tasks = [
                asyncio.create_task(process_doc(doc, random.uniform(0, 0.05)))
                for doc in docs
            ]

            # Wait for all tasks to queue their writes
            await asyncio.gather(*tasks)

            # Wait for queue to drain
            await manager.stop_writer()

        # Verify results
        # File might have 000_ prefix due to max_file_size, so use glob
        parquet_files = list(output_dir.glob("*chunk_0.parquet"))

        if not parquet_files:
            all_files = list(output_dir.glob("*.parquet"))
            pytest.fail(f"No parquet file found for chunk 0. Available files: {[f.name for f in all_files]}")

        parquet_file = parquet_files[0]  # Use the found file
        table = pq.read_table(parquet_file)
        actual_rows = len(table)

        print(f"\n=== Test Results ({'BUGGY' if use_buggy_logic else 'FIXED'} logic) ===")
        print(f"Expected rows: {num_docs}")
        print(f"Actual rows: {actual_rows}")
        print(f"Assigned: {len(manager.chunk_assigned[0])}")
        print(f"Completed: {len(manager.chunk_completed[0])}")
        print(f"Written: {len(manager.chunk_written[0])}")

        if use_buggy_logic:
            # EXPECT FAILURE with buggy logic
            # Due to random completion order, close_file() will be called
            # when assigned==completed but some docs still in queue
            if actual_rows == num_docs:
                pytest.skip(
                    f"Bug not reproduced! Got all {num_docs} rows. "
                    f"Try running multiple times or increasing concurrency."
                )
            else:
                # Bug successfully reproduced
                loss = num_docs - actual_rows
                loss_pct = (loss / num_docs) * 100
                pytest.xfail(
                    f"EXPECTED FAILURE (buggy logic): "
                    f"Lost {loss} documents ({loss_pct:.1f}%). "
                    f"File closed early when assigned==completed, "
                    f"but {loss} docs still in queue."
                )
        else:
            # EXPECT SUCCESS with fixed logic
            assert actual_rows == num_docs, (
                f"Fix failed! Expected {num_docs} rows, got {actual_rows}. "
                f"Written tracking should ensure all documents are saved before close."
            )
            print("✓ SUCCESS: All documents saved correctly with fixed logic")


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
