---
language:
- en
license: cc-by-4.0
size_categories:
- 1K<n<10K
task_categories:
- text-generation
- reinforcement-learning
tags:
- tool-use
- function-calling
- RLHF
- GRPO
- PPO
- GPT-OSS-120B
pretty_name: ToolRL Dataset - GPT OSS 120B Format
---

# ToolRL Dataset - GPT OSS 120B Format

A preprocessed tool-learning dataset in GPT OSS 120B native format for reinforcement learning training with GRPO/PPO algorithms.

## Dataset Description

This dataset contains 4,000 tool-use samples (3,920 training / 80 test) converted from the ToolRL dataset to GPT OSS 120B's native format. The conversion replaces XML-style tags with GPT OSS's special tokens and channel system, resulting in ~10-15% token efficiency improvement.

### Dataset Summary

- **Total Samples**: 4,000 (3,920 train / 80 test)
- **Format**: GPT OSS 120B native (using special tokens and channels)
- **Purpose**: Reinforcement Learning for tool-use with GRPO/PPO
- **Token Efficiency**: ~10-15% fewer tokens compared to XML format
- **Compatibility**: veRL framework, vLLM rollout
- **Data Source Identifier**: `rlla_gpt`

### Source Data

This dataset is derived from three high-quality tool-learning datasets:

- **2,000 samples** from [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) (Apache 2.0)
- **1,000 samples** from [xLAM Function Calling](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (CC BY 4.0)
- **1,000 samples** from [Hammer](https://github.com/MadeAgents/Hammer) (CC BY 4.0)

### Key Features

- ✅ **Native GPT OSS Format**: Uses GPT OSS 120B's special tokens (200002-200012)
- ✅ **Channel System**: Structured outputs with analysis, commentary json, and final channels
- ✅ **Token Efficient**: Dedicated special tokens instead of multi-token XML tags
- ✅ **RL Ready**: Includes reward function (`rlla_gpt_oss`) for GRPO/PPO training
- ✅ **Comprehensive**: System prompts updated with GPT OSS format instructions

## Dataset Structure

### Data Fields

Each sample contains:

- **data_source** (string): Always `"rlla_gpt"` (triggers GPT OSS reward function)
- **prompt** (list of dicts): List of messages with `role` and `content`
  - System message with GPT OSS format instructions
  - User message with tool-use query
- **ability** (string): Task category (e.g., "math", "finance")
- **reward_model** (dict): Contains reward configuration
  - `style` (string): Reward style (e.g., "rule")
  - `ground_truth` (string): Expected model output in GPT OSS format
- **extra_info** (dict): Additional metadata
  - `index`, `input`, `instruction`, `output`, `split`

### Data Splits

| Split | Samples |
|-------|---------|
| train | 3,920   |
| test  | 80      |

### GPT OSS Format Specification

The dataset uses GPT OSS 120B's channel system with three primary channels:

#### 1. Analysis Channel (Thinking/Reasoning)

Format:
```
<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>
```

Example:
```
<|start|>assistant<|channel|>analysis<|message|>I need to retrieve the ESG score for Microsoft using the esg tool<|end|>
```

#### 2. Commentary JSON Channel (Tool Calls)

Format:
```
<|start|>assistant to=functions.{tool_name}<|channel|>commentary json<|message|>{params_json}<|call|>
```

Example:
```
<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb":"MSFT"}<|call|>
```

#### 3. Final Channel (User Responses)

Format:
```
<|start|>assistant<|channel|>final<|message|>{response}<|return|>
```

Example:
```
<|start|>assistant<|channel|>final<|message|>I need the following information: user_id and pin<|return|>
```

### Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<\|start\|>` | 200006 | Message beginning marker |
| `<\|end\|>` | 200007 | Message ending marker |
| `<\|message\|>` | 200008 | Content separator |
| `<\|channel\|>` | 200005 | Channel specification |
| `<\|call\|>` | 200012 | Tool invocation marker |
| `<\|return\|>` | 200002 | Generation terminator (training) |

### Output Patterns

- **87.9%**: Analysis + Tool Call
- **12.1%**: Analysis + Final Response
- **0%**: Analysis + Tool Call + Final Response

## Format Conversion

### From ToolRL (XML) to GPT OSS

| Original (ToolRL) | Converted (GPT OSS) |
|-------------------|---------------------|
| `<think>...</think>` | `<\|start\|>assistant<\|channel\|>analysis<\|message\|>...<\|end\|>` |
| `<tool_call>{json}</tool_call>` | `<\|start\|>assistant to=functions.{name}<\|channel\|>commentary json<\|message\|>{params}<\|call\|>` |
| `<response>...</response>` | `<\|start\|>assistant<\|channel\|>final<\|message\|>...<\|return\|>` |

### Example Conversion

**Before (ToolRL)**:
```xml
<think>I should use the appropriate tool</think>
<tool_call>
{"name": "esg", "parameters": {"symb": "MSFT"}}
</tool_call>
```

**After (GPT OSS)**:
```
<|start|>assistant<|channel|>analysis<|message|>I should use the appropriate tool<|end|>
<|start|>assistant to=functions.esg<|channel|>commentary json<|message|>{"symb":"MSFT"}<|call|>
```

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("sungyub/toolrl-4k-verl")

# Access splits
train_data = dataset['train']
test_data = dataset['test']

# View a sample
print(train_data[0])
```

### Training with veRL

```python
from verl.trainer.ppo import RayPPOTrainer
from verl.utils.reward_score import rlla_gpt_oss

# Dataset is compatible with veRL framework
# The data_source "rlla_gpt" automatically selects the correct reward function

# Example training configuration
config = {
    'data': {
        'train_files': 'path/to/train.parquet',
        'val_files': 'path/to/test.parquet',
        'train_batch_size': 512,
        'max_prompt_length': 2048,
        'max_response_length': 1024,
    },
    'actor_rollout_ref': {
        'model': {
            'path': 'openai/gpt-oss-120b'
        }
    }
}
```

### Using with Transformers

```python
from transformers import AutoTokenizer
import pandas as pd

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")

# Load dataset
train_df = pd.read_parquet("hf://datasets/sungyub/toolrl-4k-verl/train.parquet")

# Get a sample
sample = train_df.iloc[0]
messages = sample['prompt']

# Apply chat template
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(formatted_prompt)
```

## Token Statistics

| Metric | Min | Mean | Max | Over Limit |
|--------|-----|------|-----|------------|
| **Prompt Tokens** | 584 | 953.8 | 3,991 | 27/3,920 (0.7%) |
| **Ground Truth Tokens** | 28 | 83.2 | 685 | 0/3,920 (0%) |
| **Total Tokens** | 623 | 1,037.0 | 4,201 | - |

**Note**: 27 samples (0.7%) exceed the 2048 prompt token limit and may require truncation during training.

## Data Sources & Attribution

This dataset is derived from the following sources:

### ToolACE

- **Dataset**: [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)
- **License**: Apache 2.0
- **Samples**: 2,000
- **Description**: Synthetic tool-learning data

### xLAM Function Calling

- **Dataset**: [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- **License**: CC BY 4.0
- **Samples**: 1,000
- **Description**: APIGen-generated function-calling data
- **Citation Required**: Please cite the APIGen paper when using this data (see below)

### Hammer (Masked Data)

- **Project**: [MadeAgents/Hammer](https://github.com/MadeAgents/Hammer)
- **License**: CC BY 4.0
- **Samples**: 1,000
- **Description**: Masked tool-use data

## Training Integration

### Compatible Frameworks

- ✅ **veRL**: Full support with custom reward function
- ✅ **vLLM**: Rollout generation with tensor parallelism
- ✅ **FSDP**: Fully Sharded Data Parallel training
- ✅ **Megatron**: Megatron-style parallelism

### Reward Function

The dataset includes a custom reward function (`rlla_gpt_oss`) that evaluates:

1. **Format Reward** (0.0 to 1.0)
   - Validates GPT OSS token structure
   - Checks channel usage (analysis, commentary json, final)
   - Verifies proper terminators

2. **Correctness Reward** (-3.0 to 3.0)
   - Tool name matching
   - Parameter key matching
   - Parameter value matching
   - Partial credit for partial matches

3. **Length Reward** (0.0 to 1.0, optional)
   - Measures reasoning length in analysis channel
   - Enable with `WITHLENGTH=1`

**Total Score Range**: -3.0 to 4.0 (5.0 with length reward)

### Training Example

See the [ToolRL repository](https://github.com/qiancheng0/ToolRL) for complete training scripts.

```bash
# Example training command
export BASE_MODEL="openai/gpt-oss-120b"
export DATA_DIR="./dataset/rlla_4k_gpt"
export EXPERIMENT_NAME="grpo-gpt-oss-120b"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    actor_rollout_ref.model.path=$BASE_MODEL
```

## Limitations

- **Language**: English only
- **Domain**: Limited to tool-use and function-calling tasks
- **Prompt Length**: 27 samples (0.7%) exceed 2048 tokens
- **GPT OSS Specific**: Optimized for GPT OSS 120B tokenizer
- **Synthetic Data**: Some samples are synthetically generated

## Ethical Considerations

- This dataset is designed for tool-use research and should be used responsibly
- The dataset may reflect biases present in the source datasets
- Users should ensure appropriate use cases and avoid potential misuse
- Follow the CC BY 4.0 license requirements for attribution

## Citation

If you use this dataset, please cite:

```bibtex
@article{toolrl2024,
  title={ToolRL: Reward is All Tool Learning Needs},
  author={[Authors]},
  year={2024}
}

@article{liu2024apigen,
  title={APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets},
  author={Liu, Yuxiang and Jiang, Yue and Zhang, Tianyi and Yang, Zhen and Xie, Guizhen and Zhang, Wenqi and Sun, Yunfan and Chen, Yi-Ren and Zhang, Tianjun and Rosner, Ben and Lu, Qi and Zhao, Siwei and Wei, Jiayi and De Filippo Zhouxiang, Lingxuan and Zhang, Bin Benjamin and Ying, Lei and Jiao, Jiawei and Wang, Yuming and Lu, Yumeng and Li, Xingyao and Liu, Tianqi and Luk, Karson and others},
  journal={arXiv preprint arXiv:2406.18518},
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

This dataset is licensed under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

### Source Dataset Licenses

- **ToolACE**: Apache 2.0 - https://huggingface.co/datasets/Team-ACE/ToolACE
- **xLAM Function Calling**: CC BY 4.0 - https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- **Hammer**: CC BY 4.0 - https://github.com/MadeAgents/Hammer

When using this dataset, you must:
- ✅ Provide attribution to all original sources
- ✅ Cite the APIGen paper (xLAM requirement)
- ✅ Include a copy of the CC BY 4.0 license
- ✅ Indicate if modifications were made

For full license terms, see: https://creativecommons.org/licenses/by/4.0/

## Contact & Support

- **Repository**: https://github.com/qiancheng0/ToolRL
- **Issues**: Please report issues on the GitHub repository
- **Questions**: For questions about usage, refer to the repository documentation

## Version

- **Format Version**: 1.0
- **Dataset Version**: 2025-01-04
- **Compatible Models**: GPT OSS 120B and derivatives
