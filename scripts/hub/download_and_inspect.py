"""
Download a small parquet file and inspect its schema directly.
"""

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# Download one parquet file
print("Downloading parquet file from Hub...")
file_path = hf_hub_download(
    repo_id="sungyub/deepmath-103k-verl",
    filename="data/train-00000.parquet",
    repo_type="dataset",
)

print(f"Downloaded to: {file_path}")

# Read the parquet file
print("\nReading parquet file...")
parquet_file = pq.ParquetFile(file_path)

print("\n" + "=" * 70)
print("PARQUET SCHEMA:")
print("=" * 70)
print(parquet_file.schema)

print("\n" + "=" * 70)
print("DETAILED FIELD INFO:")
print("=" * 70)

for i, field in enumerate(parquet_file.schema):
    print(f"\n{i+1}. {field.name}")
    print(f"   Type: {field.type}")
    print(f"   Nullable: {field.nullable}")

    # Special handling for prompt field
    if field.name == "prompt":
        print(f"\n   PROMPT FIELD DETAILS:")
        print(f"   - Type string: {str(field.type)}")
        print(f"   - Type ID: {field.type.id}")
        if hasattr(field.type, 'value_type'):
            print(f"   - Value type: {field.type.value_type}")
        if hasattr(field.type, 'value_field'):
            print(f"   - Value field: {field.type.value_field}")

# Read first row to see actual data
print("\n" + "=" * 70)
print("FIRST ROW DATA:")
print("=" * 70)

table = parquet_file.read_row_group(0, columns=None)
first_row = table.to_pydict()

for key in first_row:
    value = first_row[key][0] if first_row[key] else None
    print(f"\n{key}: {value}")
    if key == "prompt":
        print(f"  Type: {type(value)}")
        if isinstance(value, list) and len(value) > 0:
            print(f"  First element: {value[0]}")
            print(f"  First element type: {type(value[0])}")

print("\n" + "=" * 70)
print("HUGGINGFACE YAML FORMAT RECOMMENDATION:")
print("=" * 70)

print("""
For the 'prompt' field which is:
  list<struct<content: string, role: string>>

The correct HuggingFace YAML format is:

  - name: prompt
    list:
      - name: content
        dtype: string
      - name: role
        dtype: string

OR more explicitly:

  - name: prompt
    dtype:
      - name: content
        dtype: string
      - name: role
        dtype: string
""")
