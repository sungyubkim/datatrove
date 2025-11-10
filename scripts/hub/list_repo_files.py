"""
List files in the dataset repository.
"""

from huggingface_hub import HfApi

api = HfApi()

repo_id = "sungyub/deepmath-103k-verl"

print(f"Listing files in {repo_id}...")
print("=" * 70)

files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

parquet_files = [f for f in files if f.endswith('.parquet')]

print(f"\nFound {len(parquet_files)} parquet files:")
for f in parquet_files[:10]:  # Show first 10
    print(f"  - {f}")

if len(parquet_files) > 10:
    print(f"  ... and {len(parquet_files) - 10} more")

print(f"\nAll files:")
for f in files:
    print(f"  - {f}")
