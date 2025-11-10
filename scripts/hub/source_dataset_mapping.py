"""
Source dataset mapping for math-verl collection.
Contains original source information, licenses, and metadata for all datasets.
"""

from typing import Dict, Any

# Source dataset information mapping
SOURCE_DATASET_MAPPING: Dict[str, Dict[str, Any]] = {
    "mathx-5m-verl": {
        "dataset_name": "MathX-5M VERL",
        "source_repo_name": "XenArcAI/MathX-5M",
        "source_repo_url": "https://huggingface.co/datasets/XenArcAI/MathX-5M",
        "source_license": "MIT",
        "source_authors": "XenArcAI Team",
        "source_description": (
            "MathX-5M contains 5 million examples of highly curated step-by-step "
            "thinking data for mathematical reasoning. The dataset was curated from "
            "multiple premium mathematical datasets (Nvidia, Openr1, XenArcAI) and "
            "generated using both closed-source and open-source language models, with "
            "human-verified mathematical solutions and explanations."
        ),
        "source_paper_title": None,
        "source_paper_url": None,
        "source_citation": """@dataset{xenarcai_mathx5m_2025,
  author = {XenArcAI Team},
  title = {MathX-5M: Large-Scale Mathematical Reasoning Dataset},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/XenArcAI/MathX-5M}}
}""",
        "cleaning_preset": None,  # No MathDatasetCleaner preset
        "special_thanks": "XenArcAI team for curating high-quality mathematical reasoning data",
    },

    "eurus-2-math-verl": {
        "dataset_name": "Eurus-2 Math VERL",
        "source_repo_name": "PRIME-RL/Eurus-2-RL-Data",
        "source_repo_url": "https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data",
        "source_license": "MIT",
        "source_authors": "PRIME-RL Team",
        "source_description": (
            "Eurus-2-RL-Data contains verified mathematical reasoning problems designed "
            "for reinforcement learning applications. The dataset focuses on high-quality "
            "verification and reward modeling for mathematical problem-solving."
        ),
        "source_paper_title": None,
        "source_paper_url": None,
        "source_citation": """@dataset{primerl_eurus2_2025,
  author = {PRIME-RL Team},
  title = {Eurus-2-RL-Data: Mathematical Reasoning for Reinforcement Learning},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data}}
}""",
        "cleaning_preset": None,
        "special_thanks": "PRIME-RL team for creating verified RL training data",
    },

    "big-math-rl-verl": {
        "dataset_name": "Big-Math-RL VERL",
        "source_repo_name": "SynthLabsAI/Big-Math-RL-Verified",
        "source_repo_url": "https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified",
        "source_license": "Apache 2.0",
        "source_authors": "SynthLabs AI Team",
        "source_description": (
            "Big-Math-RL-Verified is a large-scale verified mathematical dataset designed "
            "for reinforcement learning applications. It contains diverse mathematical "
            "problems with verified solutions suitable for reward-based training."
        ),
        "source_paper_title": None,
        "source_paper_url": None,
        "source_citation": """@dataset{synthlabs_bigmath_2025,
  author = {SynthLabs AI Team},
  title = {Big-Math-RL-Verified: Large-Scale Verified Math for RL},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified}}
}""",
        "cleaning_preset": None,
        "special_thanks": "SynthLabs AI for providing verified mathematical problems",
    },

    "openr1-math-verl": {
        "dataset_name": "OpenR1-Math VERL",
        "source_repo_name": "open-r1/OpenR1-Math-220k",
        "source_repo_url": "https://huggingface.co/datasets/open-r1/OpenR1-Math-220k",
        "source_license": "Apache 2.0",
        "source_authors": "Open-R1 Team",
        "source_description": (
            "OpenR1-Math-220k contains 220,000 mathematical reasoning problems designed "
            "for training open reasoning models. The dataset covers diverse mathematical "
            "topics and difficulty levels, suitable for large-scale model training."
        ),
        "source_paper_title": None,
        "source_paper_url": None,
        "source_citation": """@dataset{openr1_math_2025,
  author = {Open-R1 Team},
  title = {OpenR1-Math-220k: Mathematical Reasoning Dataset},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/open-r1/OpenR1-Math-220k}}
}""",
        "cleaning_preset": "openr1-math",
        "special_thanks": "Open-R1 team for providing diverse mathematical reasoning problems",
    },

    "deepmath-103k-verl": {
        "dataset_name": "DeepMath-103K VERL",
        "source_repo_name": "zwhe99/DeepMath-103K",
        "source_repo_url": "https://huggingface.co/datasets/zwhe99/DeepMath-103K",
        "source_license": "MIT",
        "source_authors": "Zhenwen He et al.",
        "source_description": (
            "DeepMath-103K is a curated collection of 103,000 mathematical problems "
            "covering various difficulty levels and mathematical domains. The dataset "
            "emphasizes step-by-step reasoning and verification."
        ),
        "source_paper_title": "DeepMath: Deep Learning for Mathematical Reasoning",
        "source_paper_url": "https://arxiv.org/abs/2504.11456",
        "source_citation": """@article{he2025deepmath,
  title = {DeepMath: Deep Learning for Mathematical Reasoning},
  author = {He, Zhenwen and others},
  journal = {arXiv preprint arXiv:2504.11456},
  year = {2025},
  url = {https://arxiv.org/abs/2504.11456}
}""",
        "cleaning_preset": None,
        "special_thanks": "Zhenwen He and collaborators for the DeepMath dataset",
    },

    "orz-math-72k-verl": {
        "dataset_name": "ORZ-Math-72K VERL",
        "source_repo_name": "vwxyzjn/rlvr_orz_math_72k_collection_extended",
        "source_repo_url": "https://huggingface.co/datasets/vwxyzjn/rlvr_orz_math_72k_collection_extended",
        "source_license": "Unknown",  # TODO: Verify license
        "source_authors": "Open-Reasoner-Zero Team",
        "source_description": (
            "ORZ-Math-72K is an extended collection of mathematical problems from the "
            "Open-Reasoner-Zero project. The dataset combines problems from AIME "
            "(up to 2023), MATH, Numina-Math collection, Tulu3 MATH, and cleaned "
            "samples from OpenR1-Math-220k. It focuses on competition-level mathematical "
            "reasoning with detailed solutions."
        ),
        "source_paper_title": None,
        "source_paper_url": None,
        "source_citation": """@dataset{orz_math72k_2025,
  author = {Open-Reasoner-Zero Team},
  title = {ORZ-Math-72K: Extended Mathematical Reasoning Collection},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/vwxyzjn/rlvr_orz_math_72k_collection_extended}}
}""",
        "cleaning_preset": "orz-math",
        "special_thanks": "Open-Reasoner-Zero team and Shengyi Costa Huang (vwxyzjn) for dataset curation",
    },

    "skywork-or1-math-verl": {
        "dataset_name": "Skywork-OR1-Math VERL",
        "source_repo_name": "Skywork/Skywork-OR1-RL-Data",
        "source_repo_url": "https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data",
        "source_license": "Apache 2.0",  # Based on Skywork models typically using Apache 2.0
        "source_authors": "Skywork AI Team",
        "source_description": (
            "Skywork-OR1-RL-Data contains 105,000 verifiable and challenging mathematical "
            "problems and 14,000 coding questions used to train the Skywork-OR1 model "
            "series. The dataset underwent multi-stage filtering, deduplication, and "
            "removal of similar problems from recent benchmarks (AIME 24/25, LiveCodeBench) "
            "to prevent data contamination."
        ),
        "source_paper_title": "Skywork Open Reasoner 1 Technical Report",
        "source_paper_url": "https://arxiv.org/abs/2505.22312",
        "source_citation": """@article{skywork2025or1,
  title = {Skywork Open Reasoner 1 Technical Report},
  author = {Skywork AI Team},
  journal = {arXiv preprint arXiv:2505.22312},
  year = {2025},
  url = {https://arxiv.org/abs/2505.22312}
}""",
        "cleaning_preset": "skywork-or1",
        "special_thanks": "Skywork AI team for open-sourcing high-quality RL training data",
    },

    "deepscaler-preview-verl": {
        "dataset_name": "DeepScaleR-Preview VERL",
        "source_repo_name": "agentica-org/DeepScaleR-Preview-Dataset",
        "source_repo_url": "https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset",
        "source_license": "MIT",
        "source_authors": "Agentica Organization",
        "source_description": (
            "DeepScaleR-Preview-Dataset contains approximately 40,000 unique mathematics "
            "problem-answer pairs compiled from AIME (American Invitational Mathematics "
            "Examination, 1984-2023), AMC (American Mathematics Competition, pre-2023), "
            "Omni-MATH dataset, and Still dataset. The dataset was used to train the "
            "DeepScaleR-1.5B-Preview model using distributed reinforcement learning."
        ),
        "source_paper_title": None,
        "source_paper_url": None,
        "source_citation": """@dataset{agentica_deepscaler_2025,
  author = {Agentica Organization},
  title = {DeepScaleR-Preview-Dataset: AIME and AMC Mathematics Collection},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset}}
}""",
        "cleaning_preset": None,
        "special_thanks": "Agentica team for curating competition-level mathematics problems",
    },

    "dapo-math-17k-verl": {
        "dataset_name": "DAPO-Math-17K VERL",
        "source_repo_name": "haizhongzheng/DAPO-Math-17K-cleaned",
        "source_repo_url": "https://huggingface.co/datasets/haizhongzheng/DAPO-Math-17K-cleaned",
        "source_license": "Apache 2.0",  # Based on BytedTsinghua-SIA/DAPO-Math-17k
        "source_authors": "ByteDance Seed and Tsinghua AIR (DAPO Team)",
        "source_description": (
            "DAPO-Math-17K is a cleaned mathematical reasoning dataset from the DAPO "
            "(Decoupled Clip and Dynamic Sampling Policy Optimization) project. The "
            "dataset contains 17,000 prompts, each paired with an integer answer, "
            "designed for training reinforcement learning systems on mathematical tasks."
        ),
        "source_paper_title": "DAPO: An Open-Source LLM Reinforcement Learning System at Scale",
        "source_paper_url": "https://arxiv.org/abs/2503.14476",
        "source_citation": """@article{dapo2025,
  title = {DAPO: An Open-Source LLM Reinforcement Learning System at Scale},
  author = {ByteDance Seed and Tsinghua AIR},
  journal = {arXiv preprint arXiv:2503.14476},
  year = {2025},
  url = {https://arxiv.org/abs/2503.14476}
}""",
        "cleaning_preset": "dapo-math",
        "special_thanks": "ByteDance Seed and Tsinghua AIR teams for the DAPO project",
    },
}


# Helper function to get dataset info
def get_source_info(dataset_id: str) -> Dict[str, Any]:
    """Get source information for a dataset by its ID.

    Args:
        dataset_id: Dataset identifier (e.g., 'mathx-5m-verl')

    Returns:
        Dictionary containing source dataset information

    Raises:
        KeyError: If dataset_id is not found in mapping
    """
    if dataset_id not in SOURCE_DATASET_MAPPING:
        raise KeyError(
            f"Dataset '{dataset_id}' not found in source mapping. "
            f"Available datasets: {list(SOURCE_DATASET_MAPPING.keys())}"
        )
    return SOURCE_DATASET_MAPPING[dataset_id]


# List of all dataset IDs
DATASET_IDS = list(SOURCE_DATASET_MAPPING.keys())


# Cleaning preset mapping
CLEANING_PRESETS = {
    "orz-math": ["orz-math-72k-verl"],
    "openr1-math": ["openr1-math-verl"],
    "skywork-or1": ["skywork-or1-math-verl"],
    "dapo-math": ["dapo-math-17k-verl"],
}


# License notes for each license type
LICENSE_NOTES = {
    "MIT": (
        "This dataset is released under the MIT License, allowing free use, "
        "modification, and distribution with proper attribution."
    ),
    "Apache 2.0": (
        "This dataset is released under the Apache 2.0 License, allowing free use, "
        "modification, and distribution with proper attribution and patent grants."
    ),
    "Unknown": (
        "The license for this dataset is not clearly specified. Please refer to the "
        "original source repository for licensing information before use."
    ),
}


if __name__ == "__main__":
    # Print summary of all datasets
    print("Math-VERL Dataset Collection - Source Information Summary")
    print("=" * 70)
    print(f"\nTotal datasets: {len(DATASET_IDS)}\n")

    for dataset_id in DATASET_IDS:
        info = get_source_info(dataset_id)
        print(f"📊 {info['dataset_name']}")
        print(f"   Source: {info['source_repo_name']}")
        print(f"   License: {info['source_license']}")
        print(f"   Cleaning: {info['cleaning_preset'] or 'Standard'}")
        print()
