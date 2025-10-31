"""
Preprocess Code-Contests-Plus dataset to VERL format for reward scoring.

This script converts the ByteDance-Seed/Code-Contests-Plus (5x variant) dataset
into VERL format compatible with sandbox_fusion scoring.

By default, samples without test cases are filtered out (~2% of the dataset).
This is necessary for sandbox_fusion scoring to work properly.

Example usage:
    # Process 10 samples for testing
    python examples/preprocess_codecontests_plus.py --limit 10 --output output/codecontests_verl_test

    # Process full dataset (with filtering)
    python examples/preprocess_codecontests_plus.py --output output/codecontests_verl

    # Process without filtering empty test cases
    python examples/preprocess_codecontests_plus.py --output output/codecontests_verl --no-filter-empty
"""

import argparse
import glob
import os
from typing import Generator

import polars as pl

from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import ParquetWriter, JsonlWriter


class CodeContestsPlusReader(PipelineStep):
    """
    Custom reader for Code-Contests-Plus that reads local Parquet files with polars.

    This avoids PyArrow nested data conversion errors by reading directly from local files.
    """

    def __init__(self, data_dir="/Volumes/T7 Shield/huggingface_cache/huggingface/hub/datasets--ByteDance-Seed--Code-Contests-Plus/snapshots/7fe2c1821a7e36765e3308841658b0531ca281cf/ccplus_5x", limit=None, filter_empty_test_cases=True):
        super().__init__()
        self.data_dir = data_dir
        self.limit = limit
        self.filter_empty_test_cases = filter_empty_test_cases

    def run(self, data: any = None, rank: int = 0, world_size: int = 1) -> Generator[Document, None, None]:
        """Load and convert Code-Contests-Plus data to VERL format from local Parquet files."""

        print(f"Loading Code-Contests-Plus dataset from: {self.data_dir}")

        # Find all parquet files
        parquet_files = sorted(glob.glob(os.path.join(self.data_dir, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")

        print(f"Found {len(parquet_files)} parquet files")

        # Process each row
        total_count = 0
        processed_count = 0
        filtered_count = 0

        for parquet_file in parquet_files:
            # Shard files across workers
            file_idx = parquet_files.index(parquet_file)
            if file_idx % world_size != rank:
                continue

            print(f"Processing file {file_idx + 1}/{len(parquet_files)}: {os.path.basename(parquet_file)}")

            # Read parquet file with polars
            df = pl.read_parquet(parquet_file)

            # Convert to dictionaries and process
            for row_dict in df.to_dicts():
                total_count += 1

                # Filter out samples without test cases if enabled
                test_cases = row_dict.get("test_cases", [])
                if self.filter_empty_test_cases and (test_cases is None or len(test_cases) == 0):
                    filtered_count += 1
                    if filtered_count <= 5:  # Show first 5 filtered samples
                        print(f"  ⚠ Filtered sample without test cases: {row_dict.get('id')} ({row_dict.get('source')})")
                    continue

                # Convert to VERL format
                verl_data = self._convert_to_verl(row_dict)

                # Create a Document
                doc = Document(
                    text="",  # Empty text as VERL format uses structured data
                    id=f"codecontests_{row_dict.get('id', total_count)}",
                    metadata=verl_data
                )

                yield doc

                processed_count += 1
                if self.limit and processed_count >= self.limit:
                    print(f"Reached limit of {self.limit} samples (rank {rank})")
                    print(f"Statistics: {total_count} total, {processed_count} processed, {filtered_count} filtered")
                    return

        print(f"Completed processing (rank {rank}):")
        print(f"  Total samples read: {total_count}")
        print(f"  Samples processed: {processed_count}")
        print(f"  Samples filtered (no test cases): {filtered_count} ({filtered_count/total_count*100:.2f}%)")

    def _convert_to_verl(self, data: dict) -> dict:
        """
        Convert Code-Contests-Plus 5x format to VERL format.

        Input format (Code-Contests-Plus 5x):
        {
            "source": "Codeforces",
            "id": "847_J",
            "title": "Problem Title",
            "description": "Full problem description...",
            "test_cases": [
                {"input": "test input 1", "output": "expected output 1"},
                {"input": "test input 2", "output": "expected output 2"},
                ...
            ],
            "time_limit": 2000,
            "memory_limit": 256,
            "correct_submissions": [...],
            "incorrect_submissions": [...],
            ...
        }

        Output format (VERL):
        {
            "data_source": "codecontests",
            "ability": "code",
            "prompt": [{"role": "user", "content": "..."}],
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "inputs": ["input1", "input2", ...],
                    "outputs": ["output1", "output2", ...]
                }
            },
            "extra_info": {
                "problem_id": "847_J",
                "source_platform": "Codeforces",
                ...
            }
        }
        """
        # Extract test cases and convert to sandbox_fusion format
        test_cases_list = data.get("test_cases", [])

        # Convert from [{"input": ..., "output": ...}, ...] to {"inputs": [...], "outputs": [...]}
        inputs = [tc.get("input", "") for tc in test_cases_list]
        outputs = [tc.get("output", "") for tc in test_cases_list]

        # Build the prompt content
        prompt_content = f"""Solve the following competitive programming problem from {data['source']}:

Title: {data['title']}

Problem:
{data['description']}

Constraints:
- Time limit: {data['time_limit']}ms
- Memory limit: {data['memory_limit']}MB

Read the input from stdin and write the output to stdout.

Provide your solution in a code block. Examples:

```python
# Python solution
```

```cpp
// C++ solution
```

```java
// Java solution
```"""

        # Create VERL-formatted document
        return {
            "data_source": "codecontests",
            "ability": "code",
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "inputs": inputs,
                    "outputs": outputs
                }
            },
            "extra_info": {
                "problem_id": data.get("id", ""),
                "source_platform": data.get("source", ""),
                "title": data.get("title", ""),
                "time_limit_ms": data.get("time_limit", 0),
                "memory_limit_mb": data.get("memory_limit", 0),
                "num_test_cases": len(test_cases_list),
                "num_correct_submissions": len(data.get("correct_submissions", [])) if data.get("correct_submissions") else 0,
                "num_incorrect_submissions": len(data.get("incorrect_submissions", [])) if data.get("incorrect_submissions") else 0,
            }
        }


class VERLJsonlWriter(JsonlWriter):
    """
    Custom Jsonl writer that extracts VERL data from Document metadata.
    """

    def _get_output_dict(self, document: Document) -> dict:
        """Override to return the VERL format data from metadata instead of text."""
        return document.metadata


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Code-Contests-Plus to VERL format"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/codecontests_verl",
        help="Output directory for preprocessed data"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Volumes/T7 Shield/huggingface_cache/huggingface/hub/datasets--ByteDance-Seed--Code-Contests-Plus/snapshots/7fe2c1821a7e36765e3308841658b0531ca281cf/ccplus_5x",
        help="Path to local Code-Contests-Plus 5x parquet files"
    )
    parser.add_argument(
        "--no-filter-empty",
        action="store_true",
        help="Disable filtering of samples without test cases"
    )

    args = parser.parse_args()

    print("="*60)
    print("Code-Contests-Plus → VERL Preprocessing")
    print("="*60)
    print(f"Data dir: {args.data_dir}")
    print(f"Output: {args.output}")
    print(f"Limit: {args.limit if args.limit else 'None (full dataset)'}")
    print(f"Workers: {args.workers}")
    print(f"Filter empty test cases: {not args.no_filter_empty}")
    print("="*60)

    # Build the pipeline
    pipeline = [
        CodeContestsPlusReader(
            data_dir=args.data_dir,
            limit=args.limit,
            filter_empty_test_cases=not args.no_filter_empty,
        ),
        VERLJsonlWriter(
            output_folder=args.output,
            output_filename="${rank}.jsonl.gz",
            compression="gzip",
        ),
    ]

    # Execute the pipeline
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        workers=args.workers,
        tasks=args.workers,
        logging_dir=f"{args.output}_logs",
    )

    executor.run()

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print(f"Output: {args.output}")
    print(f"Logs: {args.output}_logs")
    print("="*60)


if __name__ == "__main__":
    main()
