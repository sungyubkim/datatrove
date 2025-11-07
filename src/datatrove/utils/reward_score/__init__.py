def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source in [
        "openai/gsm8k",
        "lighteval/MATH",
        "DigitalLearningGmbH/MATH-lighteval",
        "HuggingFaceH4/MATH-500",
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
        "math_dapo",
        "math",
        "math_dapo_reasoning",
        # Additional math datasets
        "Big-Math-RL-Verified",
        "DAPO-Math-17K",
        "DeepScaleR-Preview",
        "MathX-5M",
        "OpenR1-Math-220k",
        "orz-math-72k",
        "train-math-deepscaler",
        "train-math-numinamath1.5_amc_aime",
        "train-math-numinamath1.5_aops_forum",
        "train-math-numinamath1.5_cn_contest",
        "train-math-numinamath1.5_olympiads",
        "train-math-numinamath1.5_olympiads_ref",
        "train-math-still3",
    ]:
        from . import math

        res = math.compute_score(solution_str, ground_truth, **kwargs)
    elif data_source in [
        "rlla",
        "toolrl",
        "tool_learning",
        "toolace",
        "hammer",
        "xlam",
        "sungyub/toolrl-verl",
        "rlla_gpt",  # GPT-OSS format ToolRL
    ]:
        from . import toolrl

        res = toolrl.compute_score(
            solution_str,
            ground_truth,
            step=kwargs.get("step", 0),
            model_type=kwargs.get("model_type", "auto"),
            enable_length_reward=kwargs.get("enable_length_reward", False),
            format_type=kwargs.get("format_type", "auto"),
            **{k: v for k, v in kwargs.items() if k not in ["step", "model_type", "enable_length_reward", "format_type"]}
        )
    elif data_source in [
        "codecontests",
        "apps",
        "codeforces",
        "taco",
        # Additional code execution datasets
        "code-contests-plus",
        "kodcode-leetcode",
        "oss",  # AceCode
        "rstar-coder",
        "train-code-leetcode-Easy",
        "train-code-leetcode-Medium",
        "train-code-leetcode-Hard",
        "test-code-leetcode-Medium",
        "train-code-taco-easy",
        "train-code-taco-medium",
        "train-code-taco-hard",
        "train-code-taco-medium_hard",
        "train-code-taco-very_hard",
        "train-code-taco-unknown_difficulty",
    ]:
        # Code execution scoring requires external sandbox service
        if sandbox_fusion_url is None:
            raise ValueError(
                f"Code execution scoring for {data_source} requires a sandbox_fusion_url. "
                "Please set SANDBOX_FUSION_URL in your configuration to point to a running "
                "sandbox fusion server (e.g., 'http://your-sandbox-server.com:5000'). "
                "See VERL documentation for sandbox setup: "
                "https://github.com/volcengine/verl/tree/main/verl/utils/reward_score/sandbox_fusion"
            )

        from . import sandbox_fusion

        # Pass the URL directly, ground_truth likely contains test cases here
        res = sandbox_fusion.compute_score(
            sandbox_fusion_url,
            concurrent_semaphore,
            memory_limit_mb,
            solution_str,
            ground_truth,
            continuous=True,
        )
    elif data_source in ["allenai/IF_multi_constraints_upto5", "ifeval", "sungyub/ifbench-verl", "sungyub/ifeval-rlvr-verl"]:
        # Instruction Following evaluation
        from . import ifeval

        res = ifeval.compute_score(
            solution_str,
            ground_truth,
            **kwargs
        )
    elif data_source in ["codev", "sungyub/codev-r1-verl"]:
        # CodeV Verilog code generation with equivalence checking
        if sandbox_fusion_url is None:
            raise ValueError(
                f"CodeV scoring for {data_source} requires a sandbox_fusion_url for Verilog simulation. "
                "Please set SANDBOX_FUSION_URL in your configuration to point to a running "
                "sandbox fusion server (e.g., 'http://localhost:8080/run_code'). "
                "See documentation: https://github.com/bytedance/SandboxFusion"
            )

        from . import codev

        res = codev.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore,
            **kwargs
        )
    elif data_source in ['hitab', 'multihier', 'finqa']:
        # Table reasoning with boxed answer format (Guru datasets)
        from . import table_boxed

        res = table_boxed.compute_score(
            model_output=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif data_source in ['WTQ', 'HiTab']:
        # Table QA: WikiTableQuestions, HiTab (JSON list answers)
        from . import tqa

        res = tqa.compute_score(
            predict_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif data_source in ['TabFact']:
        # Table Fact Verification (binary: entailed/refuted)
        from . import tfv

        res = tfv.compute_score(
            predict_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif data_source in ['FeTaQA']:
        # Free-form Table QA with BLEU/ROUGE scoring
        from . import ff_tqa

        res = ff_tqa.compute_score(
            predict_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif "long_toc_choices" in data_source:
        # Long-context multiple choice QA (A-D)
        from . import long

        res = long.compute_score(
            predict_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif "docmath" in data_source:
        # Document math problems with numeric answers
        from . import docmath

        res = docmath.compute_score(
            predict_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif "multihoprag" in data_source or "musique" in data_source:
        # Document QA with free text answers (EM/F1 scoring)
        from . import docqa

        res = docqa.compute_score(
            predict_str=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    elif (
        data_source in [
            "ordering_puzzle",
            "zebra_puzzle",
            "graph_logical",
            "arcagi1",
            "arcagi2",
            "barc",
        ]
        or "puzzle" in data_source
        or "arcagi" in data_source
        or "barc" in data_source
    ):
        # Logic domain scoring: ordering puzzles, zebra puzzles, graph problems, ARC-AGI
        from . import logic

        res = logic.compute_score(
            model_output=solution_str,
            ground_truth=ground_truth,
            data_source=data_source,
            **kwargs
        )
    else:
        raise NotImplementedError(
            f"Reward function is not implemented for {data_source=}"
        )

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return {
            "score": float(res),
            "reward_fmt": 1.0,
            "reward_think": 1.0,
        }
    else:
        return {
            "score": float(res[0]),
            "reward_fmt": 1.0,
            "reward_think": 1.0,
        }


__all__ = ["compute_score"]


# Backward compatibility alias
default_compute_score = compute_score
