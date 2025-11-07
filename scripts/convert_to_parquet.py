"""Convert JSONL dataset to Parquet format for HuggingFace Hub.

This script converts the RLVR-to-IFBench transformed JSONL dataset
to Parquet format, which is the recommended format for HuggingFace datasets.

Usage:
    python scripts/convert_to_parquet.py \
        --input output/ifbench-rlvr-verl/train.jsonl \
        --output output/ifbench-rlvr-verl/train.parquet
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def jsonl_to_parquet(input_path: str, output_path: str):
    """Convert JSONL file to Parquet format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output Parquet file

    Returns:
        Number of rows converted
    """
    print(f"Reading JSONL from: {input_path}")

    # Read all examples from JSONL
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

    print(f"Loaded {len(examples)} examples")

    # Extract fields
    data_sources = []
    prompts = []
    abilities = []
    reward_models = []
    extra_infos = []
    datasets = []

    for example in examples:
        data_sources.append(example["data_source"])
        prompts.append(example["prompt"])
        abilities.append(example["ability"])
        reward_models.append(example["reward_model"])
        extra_infos.append(example["extra_info"])
        datasets.append(example["dataset"])

    # Define schema matching HuggingFace ifbench-verl format
    schema = pa.schema(
        [
            ("data_source", pa.string()),
            (
                "prompt",
                pa.list_(
                    pa.struct(
                        [
                            ("role", pa.string()),
                            ("content", pa.string()),
                        ]
                    )
                ),
            ),
            ("ability", pa.string()),
            (
                "reward_model",
                pa.struct(
                    [
                        ("style", pa.string()),
                        ("ground_truth", pa.string()),
                    ]
                ),
            ),
            (
                "extra_info",
                pa.struct(
                    [
                        ("index", pa.int64()),
                    ]
                ),
            ),
            ("dataset", pa.string()),
        ]
    )

    # Create PyArrow table
    print("Creating PyArrow table...")
    table = pa.Table.from_pydict(
        {
            "data_source": data_sources,
            "prompt": prompts,
            "ability": abilities,
            "reward_model": reward_models,
            "extra_info": extra_infos,
            "dataset": datasets,
        },
        schema=schema,
    )

    # Write to Parquet
    print(f"Writing Parquet to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pq.write_table(
        table,
        output_path,
        compression="snappy",  # Good compression ratio and speed
    )

    # Verify
    print("\nVerifying Parquet file...")
    parquet_table = pq.read_table(output_path)
    print(f"  Rows: {parquet_table.num_rows}")
    print(f"  Columns: {parquet_table.num_columns}")
    print(f"  Schema: {parquet_table.schema}")

    # Calculate file sizes
    input_size = Path(input_path).stat().st_size / (1024 * 1024)  # MB
    output_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"\nFile sizes:")
    print(f"  Input (JSONL): {input_size:.2f} MB")
    print(f"  Output (Parquet): {output_size:.2f} MB")
    print(f"  Compression ratio: {input_size/output_size:.2f}x")

    return len(examples)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Convert JSONL to Parquet format")
    parser.add_argument(
        "--input",
        type=str,
        default="output/ifbench-rlvr-verl/train.jsonl",
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/ifbench-rlvr-verl/train.parquet",
        help="Output Parquet file path",
    )

    args = parser.parse_args()

    # Convert
    num_rows = jsonl_to_parquet(args.input, args.output)

    print(f"\n✓ Successfully converted {num_rows} rows to Parquet format")


if __name__ == "__main__":
    main()
