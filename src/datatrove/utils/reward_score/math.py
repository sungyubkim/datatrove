from math_verify import parse, verify

from .utils import parse_think, parse_answer


def compute_score(
    model_output: str, ground_truth: str, timeout_score: float = 0
) -> bool:
    reward_think = 0.0
    reward_fmt = 0.0
    reward_correct = 0.0

    pred, pass_think_parsed = parse_think(model_output)
    if pass_think_parsed:
        reward_think = 1.0
        pred_parsed, pred_type = parse_answer(pred)
        gt_parsed, gt_type = parse_answer(ground_truth)
        if pred_type == gt_type:
            reward_fmt = 1.0
            is_correct = verify(
                parse(f"\\boxed{{{pred_parsed}}}"),
                parse(f"\\boxed{{{gt_parsed}}}"),
                strict=False,
                float_rounding=2,
            )
            reward_correct = 1.0 if is_correct else 0.0

    return {
        "score": reward_correct,
        "reward_think": reward_think,
        "reward_fmt": reward_fmt,
    }
