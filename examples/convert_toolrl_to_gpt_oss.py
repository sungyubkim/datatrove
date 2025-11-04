#!/usr/bin/env python3
"""
Convert ToolRL dataset from XML tag format to GPT OSS 120B native format.

This script converts:
- <think>...</think> → <|start|>assistant<|channel|>analysis<|message|>...<|end|>
- <tool_call>...</tool_call> → <|start|>assistant to=functions.{name}<|channel|>commentary json<|message|>{params}<|call|>
- <response>...</response> → <|start|>assistant<|channel|>final<|message|>...<|return|>

Usage:
    python convert_toolrl_to_gpt_oss.py --input-dir /path/to/rlla_4k --output-dir /path/to/rlla_4k_gpt

    Or using default paths (relative to script location):
    python convert_toolrl_to_gpt_oss.py
"""

import os
import re
import json
import argparse
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def convert_system_prompt(system_content: str) -> str:
    """Convert system prompt from ToolRL XML format to GPT OSS format."""

    # Replace output format instructions
    old_format_pattern = r'\*\*Output Format\*\*\s*```plaintext\s*<think>.*?</response>\s*```'

    new_format_instructions = """**Output Format**
When responding, use the following channel structure:

1. **Analysis Channel** (for thinking/reasoning):
   <|start|>assistant<|channel|>analysis<|message|>Your reasoning and thought process here<|end|>

2. **Tool Call** (when using tools):
   <|start|>assistant to=functions.ToolName<|channel|>commentary json<|message|>{"parameter": "value"}<|call|>

3. **Final Response** (when responding to user):
   <|start|>assistant<|channel|>final<|message|>Your response to the user here<|return|>

Example with tool call:
<|start|>assistant<|channel|>analysis<|message|>I need to retrieve the ESG score for Microsoft<|end|>
<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb": "MSFT"}<|call|>

Example with direct response:
<|start|>assistant<|channel|>analysis<|message|>I should directly respond to clarify missing parameters<|end|>
<|start|>assistant<|channel|>final<|message|>I need the following information: user_id, pin, and event_type<|return|>"""

    # Replace the old format section with new format
    converted = re.sub(old_format_pattern, new_format_instructions, system_content, flags=re.DOTALL)

    # If pattern not found, try a simpler replacement
    if converted == system_content:
        # Try replacing just the example part
        converted = re.sub(
            r'```plaintext\s*<think>.*?</response>\s*```',
            new_format_instructions.split('Example with tool call:')[1].strip(),
            system_content,
            flags=re.DOTALL
        )

    # Replace any remaining references to XML tags with GPT OSS format
    replacements = [
        (r'`<think>`', '`<|channel|>analysis`'),
        (r'`<tool_call>`', '`to=functions.{name}` with `<|call|>`'),
        (r'`<response>`', '`<|channel|>final`'),
        (r'<think> field', 'analysis channel'),
        (r'<tool_call> field', 'tool call with <|call| token'),
        (r'<response> field', 'final channel'),
    ]

    for old, new in replacements:
        converted = re.sub(old, new, converted, flags=re.IGNORECASE)

    return converted


def parse_ground_truth(ground_truth: str) -> str:
    """Convert ground truth from ToolRL XML format to GPT OSS format."""

    result_parts = []

    # Extract <think> content
    think_match = re.search(r'<think>(.*?)</think>', ground_truth, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        result_parts.append(
            f"<|start|>assistant<|channel|>analysis<|message|>{think_content}<|end|>"
        )

    # Extract <tool_call> content
    tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', ground_truth, re.DOTALL)
    if tool_call_match:
        tool_call_content = tool_call_match.group(1).strip()

        # Parse each JSON line (one tool call per line)
        for line in tool_call_content.split('\n'):
            line = line.strip()
            if not line:
                continue

            try:
                tool_data = json.loads(line)
                tool_name = tool_data.get('name', '')
                parameters = tool_data.get('parameters', {})

                # Convert parameters dict to JSON string (compact format)
                params_json = json.dumps(parameters, ensure_ascii=False, separators=(',', ':'))

                # Create GPT OSS tool call message
                tool_call_msg = (
                    f"<|start|>assistant to=functions.{tool_name}"
                    f"<|channel|>commentary json<|message|>{params_json}<|call|>"
                )
                result_parts.append(tool_call_msg)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse tool call JSON: {line}")
                print(f"Error: {e}")
                # Keep original line if parsing fails
                result_parts.append(f"<|start|>assistant<|channel|>commentary<|message|>{line}<|call|>")

    # Extract <response> content
    response_match = re.search(r'<response>(.*?)</response>', ground_truth, re.DOTALL)
    if response_match:
        response_content = response_match.group(1).strip()
        result_parts.append(
            f"<|start|>assistant<|channel|>final<|message|>{response_content}<|return|>"
        )

    # Join all parts with newline
    return '\n'.join(result_parts)


def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single data sample from ToolRL to GPT OSS format."""

    # Deep copy to avoid modifying original
    converted = {
        'data_source': 'rlla_gpt',  # Changed from 'rlla' to 'rlla_gpt' for GPT OSS format
        'prompt': [],
        'ability': sample['ability'],
        'reward_model': {},
        'extra_info': {
            'index': sample.get('extra_info', {}).get('index', 0)
        }
    }

    # Convert prompt messages
    for msg in sample['prompt']:
        if msg['role'] == 'system':
            # Convert system prompt
            converted_content = convert_system_prompt(msg['content'])
            converted['prompt'].append({
                'role': 'system',
                'content': converted_content
            })
        else:
            # User and other messages remain unchanged
            converted['prompt'].append(msg.copy())

    # Convert reward model ground truth
    if 'reward_model' in sample and 'ground_truth' in sample['reward_model']:
        ground_truth = sample['reward_model']['ground_truth']
        converted_ground_truth = parse_ground_truth(ground_truth)

        converted['reward_model'] = {
            'style': sample['reward_model'].get('style', 'rule'),
            'ground_truth': converted_ground_truth
        }

    return converted


def convert_parquet(input_path: str, output_path: str, verbose: bool = True):
    """Convert a parquet file from ToolRL to GPT OSS format."""

    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)

    print(f"Found {len(df)} samples")

    # Convert each sample
    converted_samples = []
    errors = 0

    for idx, row in df.iterrows():
        try:
            sample = row.to_dict()
            converted = convert_sample(sample)
            converted_samples.append(converted)

            if verbose and idx < 3:
                print(f"\n{'='*80}")
                print(f"Sample {idx} conversion:")
                print(f"{'='*80}")
                print(f"Original ground_truth:")
                print(sample['reward_model']['ground_truth'][:500])
                print(f"\nConverted ground_truth:")
                print(converted['reward_model']['ground_truth'][:500])

        except Exception as e:
            print(f"Error converting sample {idx}: {e}")
            errors += 1
            # Keep original sample if conversion fails
            converted_samples.append(row.to_dict())

    print(f"\nConversion complete: {len(converted_samples)} samples, {errors} errors")

    # Create new DataFrame
    converted_df = pd.DataFrame(converted_samples)

    # Save to parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    converted_df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return converted_df


def main():
    """Main conversion function."""

    parser = argparse.ArgumentParser(
        description='Convert ToolRL dataset from XML tag format to GPT OSS 120B native format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert using default paths (relative to ToolRL project)
  python convert_toolrl_to_gpt_oss.py

  # Convert using custom paths
  python convert_toolrl_to_gpt_oss.py \\
    --input-dir /path/to/rlla_4k \\
    --output-dir /path/to/rlla_4k_gpt

  # Quiet mode (less verbose)
  python convert_toolrl_to_gpt_oss.py --quiet
        """
    )

    # Get default paths relative to script location
    script_dir = Path(__file__).parent
    default_input = script_dir / '../ToolRL/dataset/rlla_4k'
    default_output = script_dir / 'data/rlla_4k_gpt'

    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(default_input),
        help=f'Input directory containing ToolRL dataset (default: {default_input})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(default_output),
        help=f'Output directory for GPT OSS format dataset (default: {default_output})'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce verbosity (do not show sample conversions)'
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    verbose = not args.quiet

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert train and test sets
    print("="*80)
    print("Converting ToolRL dataset to GPT OSS 120B format")
    print("="*80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Train set
    print("\n📚 Converting training set...")
    train_df = convert_parquet(
        os.path.join(input_dir, 'train.parquet'),
        os.path.join(output_dir, 'train.parquet'),
        verbose=verbose
    )

    # Test set
    print("\n📚 Converting test set...")
    test_df = convert_parquet(
        os.path.join(input_dir, 'test.parquet'),
        os.path.join(output_dir, 'test.parquet'),
        verbose=verbose
    )

    # Print statistics
    print("\n" + "="*80)
    print("Conversion Statistics")
    print("="*80)
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Total samples: {len(train_df) + len(test_df)}")

    # Analyze converted samples
    print("\n" + "="*80)
    print("Sample Analysis")
    print("="*80)

    def analyze_sample(df, name):
        analysis_count = 0
        tool_call_count = 0
        final_count = 0

        for idx, row in df.iterrows():
            gt = row['reward_model']['ground_truth']
            if '<|channel|>analysis' in gt:
                analysis_count += 1
            if 'to=functions.' in gt and '<|call|>' in gt:
                tool_call_count += 1
            if '<|channel|>final' in gt:
                final_count += 1

        print(f"\n{name}:")
        print(f"  - Samples with analysis channel: {analysis_count} ({analysis_count/len(df)*100:.1f}%)")
        print(f"  - Samples with tool calls: {tool_call_count} ({tool_call_count/len(df)*100:.1f}%)")
        print(f"  - Samples with final response: {final_count} ({final_count/len(df)*100:.1f}%)")

    analyze_sample(train_df, "Training set")
    analyze_sample(test_df, "Test set")

    print("\n✅ Conversion complete!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
