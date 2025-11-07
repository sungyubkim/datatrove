"""
Test to verify what users can monitor during execution with Solution 1.
"""

import tempfile
from pathlib import Path
from datatrove.data import Document
from datatrove.pipeline.writers import ParquetWriter


def check_parquet_readable(file_path):
    """Check if a Parquet file can be read."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(file_path)
        return True, f"{len(table)} rows"
    except Exception as e:
        return False, str(e)


def simulate_execution_with_solution1():
    """
    Simulate Solution 1 behavior:
    - Chunks are closed when records_per_chunk is reached
    - Current chunk remains open
    """

    records_per_chunk = 5
    total_docs = 22  # 4 complete chunks + 2 in progress

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir)

        writer = ParquetWriter(
            output_folder=str(output_path),
            output_filename="chunk_${chunk_index}.parquet",
            compression="snappy",
        )

        # Simulated close_file method (Solution 1)
        def close_file_properly(filename):
            """Properly close a file with flush and cleanup."""
            if filename in writer._batches:
                writer._write_batch(filename)
            if filename in writer._writers:
                writer._writers.pop(filename).close()
            writer.output_mg.pop(filename).close()

        with writer:
            chunk_counts = {}

            for i in range(total_docs):
                chunk_index = i // records_per_chunk
                chunk_counts[chunk_index] = chunk_counts.get(chunk_index, 0) + 1

                doc = Document(text=f"Test {i}", id=f"doc_{i}", metadata={})
                writer.write(doc, rank=0, chunk_index=chunk_index)

                # When chunk is complete, close it (Solution 1)
                if chunk_counts[chunk_index] == records_per_chunk:
                    filename = f"chunk_{chunk_index}.parquet"
                    close_file_properly(filename)
                    print(f"\n✅ Chunk {chunk_index} completed and closed")

                # Simulate user checking files at this point
                if i in [7, 14, 21]:  # Check at various points
                    print(f"\n📊 Progress: {i+1}/{total_docs} documents processed")
                    print("   Files readable by user:")

                    for pf in sorted(output_path.glob("*.parquet")):
                        readable, info = check_parquet_readable(pf)
                        status = "✓ READABLE" if readable else "✗ UNREADABLE"
                        print(f"      {status}: {pf.name} - {info}")

        # After completion
        print(f"\n\n📊 After pipeline completion:")
        print("   All files:")
        for pf in sorted(output_path.glob("*.parquet")):
            readable, info = check_parquet_readable(pf)
            status = "✓ READABLE" if readable else "✗ UNREADABLE"
            print(f"      {status}: {pf.name} - {info}")


if __name__ == "__main__":
    simulate_execution_with_solution1()
