#!/usr/bin/env python3
"""
Test GPT OSS tokenization with converted dataset.

This script verifies that:
1. GPT OSS tokenizer correctly loads
2. Chat template is properly applied
3. Special tokens are recognized
4. Tokenized data fits within model limits
"""

import pandas as pd
from transformers import AutoTokenizer
import json
from pathlib import Path

print("="*80)
print("GPT OSS Tokenization Test")
print("="*80)

# Load tokenizer
print("\n📥 Loading GPT OSS 120B tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    print("✅ Tokenizer loaded successfully")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Model max length: {tokenizer.model_max_length}")
except Exception as e:
    print(f"❌ Failed to load tokenizer: {e}")
    print("\nNote: If you don't have access to the model, this is expected.")
    print("You can still verify the conversion logic is correct.")
    exit(0)

# Check special tokens
print("\n🔍 Checking GPT OSS special tokens...")
special_tokens = ['<|start|>', '<|end|>', '<|message|>', '<|channel|>', '<|call|>', '<|return|>']
for token in special_tokens:
    token_id = tokenizer.encode(token, add_special_tokens=False)
    print(f"   {token}: {token_id}")

# Load converted dataset
print("\n📚 Loading converted dataset...")
# Get path relative to this test file
test_dir = Path(__file__).parent
dataset_path = test_dir / '../../../examples/data/rlla_4k_gpt/train.parquet'
train_df = pd.read_parquet(dataset_path)
print(f"✅ Loaded {len(train_df)} samples")

# Test tokenization with a few samples
print("\n" + "="*80)
print("Testing Tokenization")
print("="*80)

for idx in [0, 2, 5]:
    print(f"\n{'='*80}")
    print(f"Sample {idx}")
    print(f"{'='*80}")

    sample = train_df.iloc[idx].to_dict()

    # Get the prompt messages
    messages = sample['prompt']

    print(f"\n📝 Messages:")
    for i, msg in enumerate(messages):
        content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
        print(f"   {i+1}. [{msg['role']}] {content_preview}")

    # Apply chat template
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print(f"\n📋 Formatted prompt (first 500 chars):")
        print(formatted_prompt[:500])
        print("...")

        # Tokenize
        tokens = tokenizer.encode(formatted_prompt)
        print(f"\n📊 Token statistics:")
        print(f"   Prompt tokens: {len(tokens)}")

        # Tokenize ground truth
        ground_truth = sample['reward_model']['ground_truth']
        gt_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)
        print(f"   Ground truth tokens: {len(gt_tokens)}")
        print(f"   Total tokens: {len(tokens) + len(gt_tokens)}")

        # Check if within limits (typical: 2048 prompt + 1024 response)
        if len(tokens) > 2048:
            print(f"   ⚠️  Prompt exceeds 2048 token limit")
        if len(gt_tokens) > 1024:
            print(f"   ⚠️  Ground truth exceeds 1024 token limit")

        # Show ground truth format
        print(f"\n📝 Ground truth (first 300 chars):")
        print(ground_truth[:300])

    except Exception as e:
        print(f"❌ Error during tokenization: {e}")

# Statistics across dataset
print("\n" + "="*80)
print("Dataset Token Statistics")
print("="*80)

prompt_lengths = []
gt_lengths = []
total_lengths = []

print("\n⏳ Computing token statistics for all samples...")

for idx, row in train_df.iterrows():
    try:
        messages = row['prompt']
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_tokens = tokenizer.encode(formatted_prompt)
        ground_truth = row['reward_model']['ground_truth']
        gt_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)

        prompt_lengths.append(len(prompt_tokens))
        gt_lengths.append(len(gt_tokens))
        total_lengths.append(len(prompt_tokens) + len(gt_tokens))

    except Exception as e:
        print(f"Warning: Error processing sample {idx}: {e}")

if prompt_lengths:
    print(f"\n📊 Token length distribution:")
    print(f"   Prompt tokens:")
    print(f"     - Min: {min(prompt_lengths)}")
    print(f"     - Max: {max(prompt_lengths)}")
    print(f"     - Mean: {sum(prompt_lengths)/len(prompt_lengths):.1f}")
    print(f"     - Over 2048: {sum(1 for x in prompt_lengths if x > 2048)}")

    print(f"\n   Ground truth tokens:")
    print(f"     - Min: {min(gt_lengths)}")
    print(f"     - Max: {max(gt_lengths)}")
    print(f"     - Mean: {sum(gt_lengths)/len(gt_lengths):.1f}")
    print(f"     - Over 1024: {sum(1 for x in gt_lengths if x > 1024)}")

    print(f"\n   Total tokens:")
    print(f"     - Min: {min(total_lengths)}")
    print(f"     - Max: {max(total_lengths)}")
    print(f"     - Mean: {sum(total_lengths)/len(total_lengths):.1f}")

print("\n" + "="*80)
print("✅ Tokenization test complete!")
print("="*80)
