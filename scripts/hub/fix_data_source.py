#!/usr/bin/env python3
"""Fix data_source field to match Hub convention (DAPO-Math-17K)."""

import pyarrow.parquet as pq
import pandas as pd


def fix_data_source(input_file: str, output_file: str):
    """Fix data_source from 'dapo-math-17k' to 'DAPO-Math-17K'."""
    print(f"Reading: {input_file}")
    table = pq.read_table(input_file)
    df = table.to_pandas()

    print(f"Original data_source values: {df['data_source'].unique()}")

    # Fix data_source
    df['data_source'] = df['data_source'].str.replace('dapo-math-17k', 'DAPO-Math-17K', regex=False)

    print(f"Updated data_source values: {df['data_source'].unique()}")

    # Convert back to Arrow table
    import pyarrow as pa
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Write to output
    print(f"Writing: {output_file}")
    pq.write_table(table, output_file, compression='snappy')

    print("✅ data_source fixed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    fix_data_source(args.input, args.output)
