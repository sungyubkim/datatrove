"""
Extract the correct schema from parquet and generate proper YAML.
"""

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# Download file
print("Downloading parquet file...")
file_path = hf_hub_download(
    repo_id="sungyub/deepmath-103k-verl",
    filename="data/train-00000.parquet",
    repo_type="dataset",
)

# Read parquet
pf = pq.ParquetFile(file_path)
schema = pf.schema_arrow

print("=" * 70)
print("ARROW SCHEMA:")
print("=" * 70)
print(schema)

print("\n" + "=" * 70)
print("FIELDS BREAKDOWN:")
print("=" * 70)

for field in schema:
    print(f"\n{field.name}:")
    print(f"  Arrow Type: {field.type}")
    print(f"  Type ID: {field.type.id}")

print("\n" + "=" * 70)
print("READING ACTUAL DATA (first row):")
print("=" * 70)

table = pf.read_row_groups([0], columns=None)
first_row = table.to_pydict()

for key in first_row:
    value = first_row[key][0] if first_row[key] else None
    print(f"\n{key}:")
    print(f"  Value: {value}")
    print(f"  Python type: {type(value)}")

print("\n" + "=" * 70)
print("RECOMMENDED YAML (based on actual schema):")
print("=" * 70)

print("""
dataset_info:
  features:""")

for field in schema:
    name = field.name
    arrow_type = field.type

    print(f"  - name: {name}")

    # Handle different types
    if str(arrow_type) == "string":
        print(f"    dtype: string")
    elif str(arrow_type) == "int64":
        print(f"    dtype: int64")
    elif "list" in str(arrow_type).lower():
        # It's a list type
        if "struct" in str(arrow_type).lower():
            # List of struct
            print(f"    sequence:")
            # Try to extract struct fields
            value_type = arrow_type.value_type
            if hasattr(value_type, '__iter__') and not isinstance(value_type, str):
                for sub_field in value_type:
                    print(f"      - name: {sub_field.name}")
                    print(f"        dtype: {str(sub_field.type).replace('<', '').replace('>', '')}")
            else:
                print(f"      # TODO: Define struct fields")
        else:
            print(f"    sequence: {str(arrow_type.value_type)}")
    elif "struct" in str(arrow_type).lower():
        # It's a struct
        print(f"    struct:")
        for sub_field in arrow_type:
            print(f"      - name: {sub_field.name}")
            sub_type = str(sub_field.type).replace('<', '').replace('>', '')
            print(f"        dtype: {sub_type}")
    else:
        print(f"    dtype: {str(arrow_type)}")
