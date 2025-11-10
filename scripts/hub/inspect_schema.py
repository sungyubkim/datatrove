"""
Inspect actual schema from HuggingFace dataset.
"""

from datasets import load_dataset
import pyarrow as pa

# Load one of the datasets
print("Loading dataset from HuggingFace Hub...")
dataset = load_dataset("sungyub/deepmath-103k-verl", split="train", streaming=True)

# Get first example
print("\nFetching first example...")
first_example = next(iter(dataset))

print("\n" + "=" * 70)
print("FIRST EXAMPLE STRUCTURE:")
print("=" * 70)

for key, value in first_example.items():
    print(f"\n{key}:")
    print(f"  Type: {type(value)}")
    print(f"  Value: {value}")
    if isinstance(value, list) and len(value) > 0:
        print(f"  First element type: {type(value[0])}")
        if isinstance(value[0], dict):
            print(f"  First element: {value[0]}")

print("\n" + "=" * 70)
print("DATASET FEATURES (SCHEMA):")
print("=" * 70)

# Get the dataset features
features = dataset.features
print(features)

print("\n" + "=" * 70)
print("ARROW SCHEMA:")
print("=" * 70)

# Convert to arrow schema
arrow_schema = features.arrow_schema
print(arrow_schema)

print("\n" + "=" * 70)
print("DETAILED PROMPT FIELD:")
print("=" * 70)

prompt_feature = features['prompt']
print(f"Prompt feature type: {type(prompt_feature)}")
print(f"Prompt feature: {prompt_feature}")

# Check the actual arrow type
prompt_arrow_type = arrow_schema.field('prompt').type
print(f"\nPrompt Arrow type: {prompt_arrow_type}")
print(f"Prompt Arrow type string: {str(prompt_arrow_type)}")
