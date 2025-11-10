---
license:
- apache-2.0
- mit
- unknown
language:
- en
tags:
- math
- reasoning
- verl
- reinforcement-learning
- math-reasoning
- unified-collection
size_categories:
- 1M<n<10M
task_categories:
- text-generation
- question-answering
dataset_info:
  features:
  - name: data_source
    dtype: string
  - name: prompt
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: ability
    dtype: string
  - name: reward_model
    struct:
    - name: style
      dtype: string
    - name: ground_truth
      dtype: string
  - name: extra_info
    struct:
    - name: index
      dtype: int64
    - name: original_dataset
      dtype: string
    - name: split
      dtype: string
  splits:
  - name: train
    num_bytes: 1780000000
    num_examples: 2269166
  download_size: 1780000000
  dataset_size: 1780000000
---

# Math-VERL Unified Collection

<div align="center">

![Datasets](https://img.shields.io/badge/Datasets-9_sources-purple)
![Samples](https://img.shields.io/badge/Samples-2.27M_deduplicated-blue)
![Size](https://img.shields.io/badge/Size-{{ total_size_human }}-green)
![Format](https://img.shields.io/badge/Format-VERL-orange)
![Version](https://img.shields.io/badge/Version-v3.0-red)

</div>

## 📊 Dataset Summary

A unified collection of **9 high-quality mathematical reasoning datasets** totaling **2,269,166 deduplicated problems**, all converted to VERL format for reinforcement learning applications. This collection combines diverse mathematical content from competition-level problems to advanced reasoning tasks.

**Key Features:**
- **2.27M deduplicated samples** from 9 curated sources
- **Inter-dataset deduplication** applied (v3.0) - 12.7% cross-dataset duplicates removed
- **Unified VERL format** for consistent reward modeling across all datasets
- **Diverse difficulty levels** from elementary to IMO-level competition problems
- **Multiple mathematical domains**: algebra, geometry, number theory, calculus, combinatorics
- **Priority-based deduplication** preserving smaller, high-quality datasets

---

## 📚 Source Datasets

This unified collection aggregates the following 9 datasets:

{{ dataset_table }}

**Total after deduplication**: 2,269,166 samples (~{{ total_size_human }})

### Deduplication Priority

Smaller, more curated datasets are prioritized during inter-dataset deduplication (v3.0):
```
Priority: dapo → deepscaler → orz → deepmath → skywork → openr1 → big-math → eurus → mathx
```

---

## 🔗 Original Sources

Each dataset in this collection has been processed from the following sources:

{{ source_links }}

---

## 🔄 Preprocessing Pipeline

### Dataset-Specific Cleaning

Each source dataset underwent tailored preprocessing based on its characteristics:

{{ cleaning_summary }}

### Global Deduplication

**Phase 1: Intra-Dataset Deduplication**
- **Method**: SHA-256 hash-based with text normalization
- **Result**: ~27.9M → 2.46M samples (91.2% reduction)
- Applied individually to each dataset before merging

**Phase 2: Inter-Dataset Deduplication** (v3.0)
- **Method**: Priority-based cross-dataset deduplication
- **Result**: 2.46M → 2.27M samples (12.7% reduction)
- **Strategy**: Smaller datasets preserved; duplicates removed from larger datasets
- **Rationale**: Curated small datasets often have higher quality per sample

---

## 💡 Preprocessing Examples

Representative preprocessing examples from different cleaning presets:

### ORZ-Math Preset (Maximum Cleaning)

**Before:**
```
24th Eötvös 1917 Problem 2 A square is divided into $n^2$ smaller squares.
```

**After:**
```
A square is divided into $n^2$ smaller squares.
```

**Changes**: Removed contest metadata, problem numbers, preserved LaTeX

### OpenR1-Math Preset (Comprehensive Cleaning)

**Before:**
```
## Problem Statement

Calculate the limit: $\lim_{n \rightarrow \infty} \frac{(n+1)^{4}-(n-1)^{4}}{(n+1)^{3}+(n-1)^{3}}$
```

**After:**
```
Calculate the limit: $\lim_{n \rightarrow \infty} \frac{(n+1)^{4}-(n-1)^{4}}{(n+1)^{3}+(n-1)^{3}}$
```

**Changes**: Removed markdown headers, preserved mathematical content

---

## 📐 VERL Schema

All datasets follow the standardized VERL (Verification and Reinforcement Learning) format:

```json
{
  "data_source": "openai/gsm8k",
  "prompt": [
    {
      "content": "Calculate the sum of all odd numbers from 1 to 99.",
      "role": "user"
    }
  ],
  "ability": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "\\boxed{2500}",
    "hash": "sha256:abc123..."
  },
  "extra_info": {
    "split": "train"
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `data_source` | `string` | Original dataset identifier |
| `prompt` | `list[dict]` | User query in chat format |
| `ability` | `string` | Task type (always `"math"`) |
| `reward_model.style` | `string` | Reward computation method (`"rule"`) |
| `reward_model.ground_truth` | `string` | Expected answer (often `\boxed{}` format) |
| `reward_model.hash` | `string` | SHA-256 hash for deduplication |
| `extra_info.split` | `string` | Original split (`"train"`, `"test"`) |

---

## 📈 Dataset Statistics

### Split Distribution

{{ split_statistics }}

### Mathematical Domain Coverage

The unified collection covers diverse mathematical topics:

- **Elementary Mathematics**: Arithmetic, basic algebra, fractions
- **Competition Mathematics**: AIME, AMC, IMO-level problems
- **Advanced Topics**: Calculus, analysis, abstract algebra
- **Number Theory**: Prime numbers, modular arithmetic, Diophantine equations
- **Geometry**: Euclidean geometry, coordinate geometry, trigonometry
- **Combinatorics**: Counting, probability, graph theory

### Quality Metrics

- **Average problem length**: ~{{ avg_length }} characters
- **Ground truth format**: Primarily `\boxed{}` notation
- **Verification method**: Rule-based exact match
- **Duplicate rate** (inter-dataset): 12.7%

---

## 🚀 Usage

### Loading the Full Collection

```python
from datasets import load_dataset

# Load entire unified collection
dataset = load_dataset("sungyub/math-verl-unified")

# Access all samples
for example in dataset['train']:
    print(example['prompt'][0]['content'])
    print(example['reward_model']['ground_truth'])
```

### Loading Specific Source Datasets

```python
# Filter by data_source
dataset = load_dataset("sungyub/math-verl-unified", streaming=True)

# Filter for specific source (e.g., OpenAI GSM8K)
gsm8k_samples = [
    ex for ex in dataset['train']
    if 'gsm8k' in ex['data_source'].lower()
]
```

### Using with VERL

```python
from datatrove.utils.reward_score import compute_score

# Compute reward for generated solution
score = compute_score(
    data_source=example['data_source'],
    solution_str=generated_solution,
    ground_truth=example['reward_model']['ground_truth'],
    format_type="auto"  # Auto-detect XML or GPT OSS format
)
```

### Streaming for Large-Scale Training

```python
# Recommended for training large models
dataset = load_dataset("sungyub/math-verl-unified", streaming=True)

for batch in dataset['train'].batch(32):
    # Process batches without loading entire dataset
    train_step(batch)
```

---

## 📚 Citation

### This Unified Collection

```bibtex
@dataset{sungyub_math_verl_unified_2025,
  author = {Sungyub Kim},
  title = {Math-VERL Unified Collection: A Curated Collection of 2.27M Mathematical Reasoning Problems},
  year = {2025},
  publisher = {Hugging Face},
  version = {3.0},
  howpublished = {\url{https://huggingface.co/datasets/sungyub/math-verl-unified}}
}
```

### Original Source Datasets

Please cite the original datasets when using this collection:

{{ source_citations }}

---

## ⚖️ License

**Mixed Licenses** - Each source dataset retains its original license:

{{ license_table }}

**Usage Terms**:
- Respect individual dataset licenses when using subsets
- Attribution required for all source datasets
- Commercial use permitted where allowed by source licenses

---

## 🙏 Acknowledgments

This unified collection was made possible by:

**Original Dataset Authors**:
{{ acknowledgments }}

**Processing Tools**:
- [DataTrove](https://github.com/huggingface/datatrove) - Data processing pipeline
- [VERL](https://github.com/volcengine/verl) - Reward modeling framework

**Special Thanks**:
- All dataset creators for open-sourcing their work
- Hugging Face for hosting infrastructure
- The open-source ML community

---

## 📝 Version History

### v3.0.0 (Current - Inter-Dataset Deduplication)
- **Breaking Change**: Inter-dataset deduplication applied
- Reduced from 2.46M to 2.27M samples (12.7% cross-dataset duplicates removed)
- Priority-based deduplication strategy
- Improved quality through duplicate removal
- **Total**: 2,269,166 unique samples

### v2.0.0 (Intra-Dataset Deduplication)
- Intra-dataset deduplication applied to all 9 datasets
- Reduced from ~27.9M to 2.46M samples (91.2% reduction)
- SHA-256 hash-based with text normalization
- All datasets converted to VERL format

### v1.0.0 (Initial Release)
- Initial collection of 9 mathematical reasoning datasets
- Individual dataset cleaning presets applied
- VERL format conversion
- Basic validation and quality checks

---

## 🔗 Related Resources

### Individual Datasets
{{ related_links }}

### Documentation
- **DataTrove**: [https://github.com/huggingface/datatrove](https://github.com/huggingface/datatrove)
- **VERL Framework**: [https://github.com/volcengine/verl](https://github.com/volcengine/verl)
- **Reward Scoring Guide**: See DataTrove documentation

### Other Math Collections
- [OpenMathInstruct](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)
- [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)
- [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)

---

<div align="center">

**Questions or issues?** Open an issue on the [DataTrove GitHub repository](https://github.com/huggingface/datatrove/issues)

**Want to contribute?** We welcome additional high-quality math datasets!

</div>
