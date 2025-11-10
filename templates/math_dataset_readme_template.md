---
license: {{ license }}
language:
- en
tags:
- math
- reasoning
- verl
- reinforcement-learning
- math-reasoning
size_categories:
- {{ size_category }}
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
    num_bytes: {{ num_bytes }}
    num_examples: {{ num_examples }}
  download_size: {{ download_size }}
  dataset_size: {{ dataset_size }}
---

# {{ dataset_name }}

<div align="center">

![Dataset](https://img.shields.io/badge/Dataset-{{ num_examples }}_samples-blue)
![Size](https://img.shields.io/badge/Size-{{ dataset_size_human | replace(' ', '_') }}-green)
![Format](https://img.shields.io/badge/Format-VERL-orange)
![License](https://img.shields.io/badge/License-{{ license | replace(' ', '_') }}-red)

</div>

## 📊 Dataset Summary

{{ dataset_summary }}

**Key Features:**
{{ key_features }}

---

## 🔗 Source Dataset

### Original Repository
- **Repository**: [{{ source_repo_name }}]({{ source_repo_url }})
- **License**: {{ source_license }}
{% if source_paper_title %}
- **Paper**: [{{ source_paper_title }}]({{ source_paper_url }})
{% endif %}
- **Authors**: {{ source_authors }}

### Dataset Description
{{ source_description }}

---

## 🔄 Preprocessing Pipeline

This dataset has been preprocessed and converted to the VERL (Verification and Reinforcement Learning) format for use in mathematical reasoning tasks with reward modeling.

### Cleaning Methodology

{% if cleaning_preset %}
**Cleaning Preset**: `{{ cleaning_preset }}`

The following artifact patterns were removed using the `MathDatasetCleaner` formatter:

{{ cleaning_patterns }}

**Cleaning Statistics:**
- **Original samples**: {{ original_samples | format_number }}
- **After cleaning**: {{ cleaned_samples | format_number }}
- **Removed samples**: {{ removed_samples | format_number }} ({{ removal_rate }}%)
- **Artifacts removed**: {{ artifacts_removed | format_number }}
{% else %}
**Standard Processing:**
- URL filtering (samples containing URLs removed)
- Format normalization to VERL schema
- Basic text cleaning and validation
{% endif %}

### Deduplication

{% if dedup_stats %}
**Intra-dataset Deduplication:**
- **Method**: SHA-256 hash-based with text normalization
- **Before deduplication**: {{ before_dedup | format_number }} samples
- **After deduplication**: {{ after_dedup | format_number }} samples
- **Reduction**: {{ dedup_reduction }}%

{% if inter_dedup %}
**Inter-dataset Deduplication** (v3.0):
- **Priority level**: {{ dedup_priority }}
- **Cross-dataset duplicates removed**: {{ inter_dedup_removed | format_number }}
{% endif %}
{% endif %}

---

## 💡 Preprocessing Examples

### Example 1: {{ example1_title }}

**Before Cleaning:**
```
{{ example1_before }}
```

**After Cleaning:**
```
{{ example1_after }}
```

**Changes Applied:**
{{ example1_changes }}

{% if example2_before %}
### Example 2: {{ example2_title }}

**Before Cleaning:**
```
{{ example2_before }}
```

**After Cleaning:**
```
{{ example2_after }}
```

**Changes Applied:**
{{ example2_changes }}
{% endif %}

---

## 📐 VERL Schema

This dataset follows the standardized VERL (Verification and Reinforcement Learning) format:

```json
{
  "data_source": "{{ data_source_example }}",
  "prompt": [
    {
      "content": "{{ prompt_example }}",
      "role": "user"
    }
  ],
  "ability": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "{{ ground_truth_example }}",
    "hash": "{{ hash_example }}"
  },
  "extra_info": {
    "split": "train"
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `data_source` | `string` | Original dataset identifier (e.g., `openai/gsm8k`, `numina_aime`) |
| `prompt` | `list[dict]` | User query in chat format with role and content |
| `ability` | `string` | Task type (always `"math"` for this dataset) |
| `reward_model.style` | `string` | Reward computation method (`"rule"` for rule-based verification) |
| `reward_model.ground_truth` | `string` | Expected answer for verification (often in `\boxed{}` format) |
| `reward_model.hash` | `string` | SHA-256 hash of prompt content for deduplication |
| `extra_info.split` | `string` | Original split identifier (`"train"`, `"test"`, etc.) |

---

## 📈 Dataset Statistics

{{ dataset_statistics }}

---

## 🚀 Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("{{ hub_dataset_path }}")

# Load with streaming (recommended for large datasets)
dataset = load_dataset("{{ hub_dataset_path }}", streaming=True)

# Preview first few examples
for example in dataset['train'].take(5):
    print(example['prompt'][0]['content'])  # User question
    print(example['reward_model']['ground_truth'])  # Answer
    print("---")
```

### Using with VERL

```python
from datatrove.utils.reward_score import compute_score

# Compute reward score for a generated solution
score = compute_score(
    data_source=example['data_source'],
    solution_str=generated_solution,
    ground_truth=example['reward_model']['ground_truth'],
    format_type="auto"  # Auto-detect XML or GPT OSS format
)

print(f"Reward score: {score}")
```

### Integration with DataTrove

```python
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.executor import LocalPipelineExecutor

pipeline = [
    ParquetReader("{{ hub_dataset_path }}", text_key="prompt"),
    LambdaFilter(lambda doc: len(doc.text) > 100),  # Filter short problems
    # Add more processing steps...
]

executor = LocalPipelineExecutor(pipeline=pipeline, tasks=4)
executor.run()
```

---

## 📚 Citation

### Original Dataset

```bibtex
{{ source_citation }}
```

### This Processed Version

```bibtex
@dataset{sungyub_math_verl_{{ dataset_id }},
  author = {Sungyub Kim},
  title = {{ dataset_name }},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/{{ hub_dataset_path }}}}
}
```

---

## ⚖️ License

- **This processed dataset**: {{ license }}
- **Original dataset**: {{ source_license }}

{{ license_notes }}

---

## 🙏 Acknowledgments

This dataset was processed using the [DataTrove](https://github.com/huggingface/datatrove) library.

**Credits:**
- Original dataset authors: {{ source_authors }}
- Processing and VERL conversion: Sungyub Kim
- MathDatasetCleaner implementation: DataTrove contributors

**Special thanks to:**
{{ special_thanks }}

---

## 📝 Version History

{{ version_history }}

---

## 🔗 Related Resources

- **Unified Collection**: [sungyub/math-verl-unified](https://huggingface.co/datasets/sungyub/math-verl-unified) - All 9 math datasets with inter-dataset deduplication
- **DataTrove Documentation**: [https://github.com/huggingface/datatrove](https://github.com/huggingface/datatrove)
- **VERL Format Specification**: See VERL Schema section above

---

<div align="center">

**Questions or issues?** Open an issue on the [DataTrove GitHub repository](https://github.com/huggingface/datatrove/issues)

</div>
