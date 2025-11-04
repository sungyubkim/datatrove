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
    ]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
        "rlla",
        "toolrl",
        "tool_learning",
        "toolace",
        "hammer",
        "xlam",
        "sungyub/toolrl-verl",
    ]:
        from . import toolrl

        res = toolrl.compute_score(
            solution_str,
            ground_truth,
            step=kwargs.get("step", 0),
            model_type=kwargs.get("model_type", "auto"),
            enable_length_reward=kwargs.get("enable_length_reward", False),
        )
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
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
    elif data_source in ["allenai/IF_multi_constraints_upto5", "ifeval", "sungyub/ifbench-verl"]:
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


__all__ = ["default_compute_score"]
