# ToolRL Dataset - GPT OSS 120B Format

This directory contains the ToolRL dataset converted to GPT OSS 120B native format.

## Overview

The original ToolRL dataset used XML-style tags (`<think>`, `<tool_call>`, `<response>`) which are not natively supported by GPT OSS 120B's tokenizer. This converted dataset uses GPT OSS's native special tokens and channel system for optimal compatibility and training efficiency.

## Dataset Statistics

- **Training samples**: 3,920
- **Test samples**: 80
- **Total samples**: 4,000
- **Data source**: `rlla_gpt` (triggers GPT OSS reward function)

## Format Conversion

### Original Format (ToolRL)

```xml
<think>I should use the appropriate tool</think>
<tool_call>
{"name": "esg", "parameters": {"symb": "MSFT"}}
</tool_call>
```

### Converted Format (GPT OSS)

```
<|start|>assistant<|channel|>analysis<|message|>I should use the appropriate tool<|end|>
<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb":"MSFT"}<|call|>
```

## Channel System

GPT OSS 120B uses three primary channels for structured outputs:

### 1. Analysis Channel (Thinking/Reasoning)

**Purpose**: Model's internal reasoning and thought process

**Format**:
```
<|start|>assistant<|channel|>analysis<|message|>{thinking_content}<|end|>
```

**Example**:
```
<|start|>assistant<|channel|>analysis<|message|>I need to retrieve the ESG score for Microsoft using the esg tool<|end|>
```

### 2. Commentary JSON Channel (Tool Calls)

**Purpose**: Function/tool invocation with parameters

**Format**:
```
<|start|>assistant to=functions.{tool_name}<|channel|>commentary json<|message|>{parameters_json}<|call|>
```

**Example**:
```
<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb":"MSFT"}<|call|>
```

**Key points**:
- One tool call per message block
- Multiple tool calls = multiple message blocks
- Parameters in compact JSON format (no spaces)
- Ends with `<|call|>` token (not `<|end|>`)

### 3. Final Channel (User Responses)

**Purpose**: Final text response to the user

**Format**:
```
<|start|>assistant<|channel|>final<|message|>{response_content}<|return|>
```

**Example**:
```
<|start|>assistant<|channel|>final<|message|>I need the following information: user_id, pin, and event_type<|return|>
```

**Note**: Ends with `<|return|>` during training, `<|end|>` during inference

## Output Patterns

### Pattern 1: Analysis + Tool Call (87.9%)

Most common pattern for tool-using responses.

```
<|start|>assistant<|channel|>analysis<|message|>I should use the esg tool<|end|>
<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb":"MSFT"}<|call|>
```

### Pattern 2: Analysis + Final Response (12.1%)

For responses that don't require tool calls.

```
<|start|>assistant<|channel|>analysis<|message|>I should respond directly<|end|>
<|start|>assistant<|channel|>final<|message|>Please provide the user_id and pin<|return|>
```

### Pattern 3: Analysis + Multiple Tool Calls

Each tool call is a separate message block.

```
<|start|>assistant<|channel|>analysis<|message|>I need to call two tools<|end|>
<|start|>assistant to=functions.tool1<|channel|>commentary json<|message|>{"p1":"v1"}<|call|>
<|start|>assistant to=functions.tool2<|channel|>commentary json<|message|>{"p2":"v2"}<|call|>
```

## Special Tokens

GPT OSS 120B uses these special tokens:

| Token | ID | Purpose |
|-------|-----|---------|
| `<|start|>` | 200006 | Message beginning marker |
| `<|end|>` | 200007 | Message ending marker |
| `<|message|>` | 200008 | Content separator |
| `<|channel|>` | 200005 | Channel specification |
| `<|call|>` | 200012 | Tool invocation marker |
| `<|return|>` | 200002 | Generation terminator (training) |

## Data Schema

Each sample contains:

```python
{
    "data_source": "rlla_gpt",  # Triggers GPT OSS reward function
    "prompt": [
        {
            "role": "system",
            "content": "..."  # Updated with GPT OSS format instructions
        },
        {
            "role": "user",
            "content": "..."  # Unchanged
        }
    ],
    "ability": "math",  # Task category
    "reward_model": {
        "style": "rule",
        "ground_truth": "..."  # Converted to GPT OSS format
    },
    "extra_info": {
        "index": 0,
        "input": "...",
        "instruction": "...",
        "output": "...",
        "split": "train"
    }
}
```

## System Prompt Changes

The system prompt has been updated to teach GPT OSS format:

### Old (ToolRL):
```
**Output Format**
<think> Your thoughts and reasoning </think>
<tool_call>
{"name": "Tool name", "parameters": {"param": "value"}}
</tool_call>
<response> AI's final response </response>
```

### New (GPT OSS):
```
**Output Format**
<|start|>assistant<|channel|>analysis<|message|>Your reasoning here<|end|>
<|start|>assistant to=functions.{tool_name}<|channel|>commentary json<|message|>{"parameter": "value"}<|call|>
<|start|>assistant<|channel|>final<|message|>Your response here<|return|>
```

## Reward Function

The dataset uses a custom reward function (`rlla_gpt_oss`) that evaluates:

### 1. Format Reward (0.0 to 1.0)

Validates:
- Correct token sequence (`<|start|>`, `<|channel|>`, `<|message|>`, etc.)
- Proper channel usage (analysis, commentary json, final)
- Appropriate terminators (`<|end|>`, `<|call|>`, `<|return|>`)

### 2. Correctness Reward (-3.0 to 3.0)

For tool calls:
- Tool name matching (frequency-based)
- Parameter key matching
- Parameter value matching
- Partial credit for partial matches

### 3. Length Reward (0.0 to 1.0, optional)

Measures reasoning length in analysis channel.
- Target: 384-512 words
- Enable with `WITHLENGTH=1`

### Total Score

```python
score = format_score + correctness_score + length_score
```

**Perfect match**: 4.0 (format=1.0, correctness=3.0, length=0.0 by default)

## Training

### Quick Start

```bash
bash train_grpo_gpt_oss.sh
```

### Configuration

Edit `train_grpo_gpt_oss.sh`:

```bash
export BASE_MODEL="openai/gpt-oss-120b"
export DATA_DIR="./dataset/rlla_4k_gpt"
export EXPERIMENT_NAME="grpo-gpt-oss-120b"
export N_GPUS=8
export ROLLOUT_TP_SIZE=2
```

### Reward Variants

Enable optional reward features:

```bash
export WITHLENGTH=1          # Add length reward
export CORRECTMAX1=1         # Scale correctness to ±1.0
export SCHEDULEREWARD=1      # Dynamic reward scaling
export SCHEDULELENGTH=1      # Dynamic length targets
export REFINEDREWARD=1       # Strict matching
export COARSEREWARD=1        # Binary rewards only
export INTERMEDIATEREWARD=1  # Simplified scoring
```

## Files

```
dataset/rlla_4k_gpt/
├── README.md                           # This file
├── train.parquet                       # Training data (3,920 samples)
└── test.parquet                        # Test data (80 samples)

dataset/rlla_4k_raw/
└── convert_to_gpt_oss.py              # Conversion script

verl/utils/reward_score/
└── rlla_gpt_oss.py                    # GPT OSS reward function

verl/trainer/
└── main_ppo.py                        # Updated to support rlla_gpt

train_grpo_gpt_oss.sh                  # Training script
test_gpt_oss_reward_standalone.py      # Reward function tests
```

## Testing

### Test Reward Function

```bash
python3 test_gpt_oss_reward_standalone.py
```

This validates:
- ✅ Format reward parsing
- ✅ Tool call extraction
- ✅ Correctness scoring
- ✅ Edge cases (wrong tool, missing analysis, etc.)

### Test Tokenization (requires transformers)

```bash
python3 test_gpt_oss_tokenization.py
```

This checks:
- ✅ Tokenizer loading
- ✅ Chat template application
- ✅ Special token recognition
- ✅ Token length statistics

## Conversion Process

To regenerate the dataset:

```bash
python3 dataset/rlla_4k_raw/convert_to_gpt_oss.py
```

**What it does**:
1. Reads `dataset/rlla_4k/train.parquet` and `test.parquet`
2. Converts system prompts (format instructions)
3. Parses and converts ground truth:
   - `<think>` → analysis channel
   - `<tool_call>` JSON lines → individual commentary json messages
   - `<response>` → final channel
4. Updates `data_source` to `rlla_gpt`
5. Saves to `dataset/rlla_4k_gpt/`

## Key Differences from ToolRL

| Aspect | ToolRL | GPT OSS |
|--------|--------|---------|
| **Think** | `<think>...</think>` | `<|channel|>analysis<|message|>...<|end|>` |
| **Tool call** | `<tool_call>\n{json}\n</tool_call>` | `to=functions.{name}<|channel|>commentary json<|message|>{json}<|call|>` |
| **Response** | `<response>...</response>` | `<|channel|>final<|message|>...<|return|>` |
| **Multiple tools** | Multiple JSON lines in one block | Separate message blocks |
| **Tokenization** | Generic text tokens | Dedicated special tokens |
| **Data source** | `rlla` | `rlla_gpt` |
| **Reward function** | `verl.utils.reward_score.rlla` | `verl.utils.reward_score.rlla_gpt_oss` |

## Performance Expectations

### Token Efficiency

GPT OSS special tokens are single tokens, compared to XML tags which require multiple tokens:
- `<think>` → ~3 tokens in generic tokenizer
- `<|channel|>analysis` → 2 special tokens in GPT OSS
- Estimated savings: ~10-15% fewer tokens per sample

### Training Compatibility

- ✅ Compatible with GRPO (Group Relative Policy Optimization)
- ✅ Compatible with PPO (Proximal Policy Optimization)
- ✅ Works with vLLM rollout
- ✅ Supports tensor parallelism
- ✅ Supports FSDP (Fully Sharded Data Parallel)

## Troubleshooting

### Issue: "No module named 'rlla_gpt_oss'"

**Solution**: Make sure you've imported the module in `verl/trainer/main_ppo.py`:

```python
from verl.utils.reward_score import gsm8k, math, multiply, countdown, rlla, rlla_gpt_oss
```

### Issue: Wrong reward function selected

**Solution**: Check that `data_source` in the parquet files is `"rlla_gpt"` (not `"rlla"`):

```python
import pandas as pd
df = pd.read_parquet('dataset/rlla_4k_gpt/train.parquet')
print(df['data_source'].unique())  # Should show ['rlla_gpt']
```

### Issue: Format score always 0

**Solution**: Verify the response contains proper GPT OSS tokens:
- Must have `<|start|>assistant`
- Must have `<|channel|>`
- Must have proper terminators (`<|end|>`, `<|call|>`, or `<|return|>`)

### Issue: Tokenizer not found

**Solution**: Install and authenticate HuggingFace:

```bash
pip install transformers
huggingface-cli login
```

## Citation

If you use this converted dataset, please cite:

```bibtex
@article{toolrl2024,
  title={ToolRL: Reward is All Tool Learning Needs},
  author={...},
  year={2024}
}

@misc{gpt-oss-120b,
  title={GPT OSS 120B},
  author={OpenAI},
  howpublished={\url{https://huggingface.co/openai/gpt-oss-120b}},
  year={2024}
}
```

## License

Same as the original ToolRL dataset (Apache 2.0).

## Contact

For issues related to:
- **Dataset conversion**: Check `dataset/rlla_4k_raw/convert_to_gpt_oss.py`
- **Reward function**: Check `verl/utils/reward_score/rlla_gpt_oss.py`
- **Training**: Check `train_grpo_gpt_oss.sh` and training logs

---

**Generated**: 2025-01-04
**Format Version**: 1.0
**Compatible with**: GPT OSS 120B, veRL framework
