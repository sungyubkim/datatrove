# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CodeV scorer for Verilog code generation.
Performs equivalence checking between generated code and golden reference
using Sandbox Fusion for Verilog simulation.

Supports both XML (<think>, <answer>) and GPT OSS (<|channel|>analysis, <|channel|>final) formats.
"""

import json
import logging
import pickle
import re
import threading
from itertools import combinations, product
from typing import Optional

from .codev_eval_toolkit import eda_tools, extract_verilog
from .sandbox_fusion.utils import call_sandbox_api
from .format_handlers import detect_format, get_format_handler

logger = logging.getLogger(__name__)


def check_format(output, format_type="auto"):
    """
    Check if the output has proper format with thinking and answer sections.

    Supports:
    - XML format: <think>...</think> and <answer>...</answer> tags
    - GPT OSS format: <|channel|>analysis and <|channel|>final blocks

    Args:
        output: String containing the model output
        format_type: Response format ("xml", "gpt_oss", or "auto" for auto-detection)

    Returns:
        bool: True if format is valid, False otherwise

    Examples:
        XML format:
        >>> check_format("<think>reasoning</think><answer>```verilog\\nmodule...\\n```</answer>")
        True

        GPT OSS format:
        >>> check_format("<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>\\n<|start|>assistant<|channel|>final<|message|>```verilog\\nmodule...\\n```<|return|>")
        True
    """
    if format_type == "auto":
        format_type = detect_format(output)

    if format_type == "gpt_oss":
        # For GPT OSS, check that we have analysis and final channels
        handler = get_format_handler(format_type)
        has_analysis = handler.has_analysis(output)
        has_final = handler.has_final_response(output)

        # Must have at least final channel (analysis is optional)
        return has_final
    else:
        # XML format: <think> tags are required, <answer> tags are optional (Qwen3 compatibility)
        # Pattern 1 (traditional): <think>...</think>\n<answer>...</answer>
        # Pattern 2 (Qwen3): <think>...</think>\n[plain text with code]

        # Must have exactly one pair of <think> tags
        if output.count("<think>") != 1 or output.count("</think>") != 1:
            return False

        # Check if has <answer> tags (traditional format)
        if output.count("<answer>") == 1 and output.count("</answer>") == 1:
            # Validate tag order: <think>, </think>, <answer>, </answer>
            tags = ["<think>", "</think>", "<answer>", "</answer>"]
            positions = [output.find(tag) for tag in tags]
            return positions[0] < positions[1] < positions[2] < positions[3]

        # Or allow plain text after </think> (Qwen3 format)
        # Just verify <think> tags are properly positioned
        return (output.find("<think>") < output.find("</think>") and
                output.strip().startswith("<think>"))


def assemble_verilog_code(golden_code, dut_code, testbench_code):
    """
    Assemble golden code, DUT code, and testbench into a single Verilog file.

    Args:
        golden_code: Golden reference Verilog code with _gold suffix
        dut_code: Device Under Test Verilog code with _gate suffix
        testbench_code: Testbench module code

    Returns:
        str: Complete Verilog program
    """
    return f"""
// ========== GOLDEN REFERENCE CODE ==========
{golden_code}

// ========== DEVICE UNDER TEST CODE ==========
{dut_code}

// ========== TESTBENCH CODE ==========
{testbench_code}
"""


def verify_verilog_via_sandbox(
    golden_code,
    dut_code,
    golden_top,
    gate_top,
    port_info,
    sandbox_fusion_url,
    concurrent_semaphore=None,
    compile_timeout=30,
    run_timeout=60,
    random_seq_steps=1000,
    random_seq_num=100
):
    """
    Verify Verilog code equivalence using Sandbox Fusion.

    Args:
        golden_code: Golden reference Verilog code
        dut_code: Device Under Test Verilog code
        golden_top: Top module name for golden code
        gate_top: Top module name for DUT code
        port_info: Tuple of (input_port_width, output_port_width, clock_port_polarity, reset_port_polarity_sync)
        sandbox_fusion_url: URL of Sandbox Fusion service
        concurrent_semaphore: Optional semaphore for concurrency control
        compile_timeout: Compilation timeout in seconds
        run_timeout: Run timeout in seconds
        random_seq_steps: Number of random test steps per sequence
        random_seq_num: Number of random test sequences

    Returns:
        dict: Verification result with 'correct' boolean and optional error info
    """
    try:
        # Initialize EDA tools
        v = eda_tools(
            golden_suffix="_gold",
            gate_suffix="_gate",
            random_seq_steps=random_seq_steps,
            random_seq_num=random_seq_num,
            quiet=True
        )

        # Process Verilog code (add suffixes to module names)
        renamed_golden_code = v.process_verilog(golden_code, "_gold")
        renamed_gate_code = v.process_verilog(dut_code, "_gate")

        # Extract port information
        input_port_width, output_port_width, clock_port_polarity, reset_port_polarity_sync = port_info

        # Generate testbench
        testbench_code = v.generate_testbench(
            input_port_width=input_port_width,
            output_port_width=output_port_width,
            clock_port_polarity=clock_port_polarity,
            reset_port_polarity_sync=reset_port_polarity_sync,
            golden_top=golden_top,
            gate_top=gate_top
        )

        # Assemble complete Verilog program
        full_verilog = assemble_verilog_code(renamed_golden_code, renamed_gate_code, testbench_code)

        # Call Sandbox Fusion API
        logger.info("Calling Sandbox Fusion for Verilog simulation")
        api_response, error_msg = call_sandbox_api(
            sandbox_fusion_url=sandbox_fusion_url,
            code=full_verilog,
            stdin="",  # Testbench self-generates inputs
            compile_timeout=compile_timeout,
            run_timeout=run_timeout,
            memory_limit_mb=2048,
            language="verilog"
        )

        if error_msg:
            logger.error(f"Sandbox API error: {error_msg}")
            return {"correct": False, "api_error": error_msg}

        if not api_response:
            logger.error("No API response received")
            return {"correct": False, "api_error": "No response from Sandbox"}

        # Check response status
        api_status = api_response.get("status")
        if api_status != "Success":
            logger.warning(f"API returned status: {api_status}")
            compile_result = api_response.get("compile_result", {})
            run_result = api_response.get("run_result", {})

            error_info = {
                "correct": False,
                "api_status": api_status,
                "compile_stderr": compile_result.get("stderr") if compile_result else None,
                "run_stderr": run_result.get("stderr") if run_result else None
            }
            return error_info

        # Check run result
        run_result = api_response.get("run_result", {})
        stdout = run_result.get("stdout", "")
        stderr = run_result.get("stderr", "")

        # Parse error rate from stdout
        error_rate_pattern = r"Error rate:\s*(\d+\.\d+)"
        error_rate_match = re.search(error_rate_pattern, stdout)
        error_rate = float(error_rate_match.group(1)) if error_rate_match else 1.0

        # Check if all tests passed
        if "All tests passed." in stdout:
            logger.info("Verification passed: All tests passed")
            return {
                "correct": True,
                "error_rate": error_rate,
                "stdout": stdout[:500],  # Truncate for logging
                "stderr": stderr[:500] if stderr else None
            }
        else:
            logger.info(f"Functional mismatch: {error_rate*100:.1f}% of test cases failed (code compiled and ran successfully)")
            return {
                "correct": False,
                "error_rate": error_rate,
                "stdout": stdout[:500],
                "stderr": stderr[:500] if stderr else None
            }

    except Exception as e:
        logger.error(f"Exception during Verilog verification: {e}", exc_info=True)
        return {"correct": False, "exception": str(e)}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: bytes,
    extra_info: dict = None,
    sandbox_fusion_url: str = "http://localhost:8080/run_code",
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    format_type: str = "auto",
    **kwargs
):
    """
    Compute score for CodeV Verilog generation task.

    Supports both XML (<think>, <answer>) and GPT OSS (<|channel|>analysis, <|channel|>final) formats.

    Args:
        data_source: Data source identifier (should be 'codev')
        solution_str: Generated solution string (may contain thinking and answer sections)
        ground_truth: Pickled ground truth data containing golden Verilog code and port info
        extra_info: Optional extra information
        sandbox_fusion_url: URL of Sandbox Fusion service
        concurrent_semaphore: Optional semaphore for concurrency control
        format_type: Response format ("xml", "gpt_oss", or "auto" for auto-detection)
        **kwargs: Additional arguments

    Returns:
        dict: Score dictionary with 'score', 'reward_fmt', 'reward_think' keys
    """
    try:
        # Parse ground truth from JSON or pickle (backward compatibility)
        if isinstance(ground_truth, bytes):
            # Legacy format: pickled bytes
            gts = pickle.loads(ground_truth)
        elif isinstance(ground_truth, str):
            # New format: JSON string
            gts = json.loads(ground_truth)
            # Convert lists back to sets for port info (JSON doesn't support sets or tuples)
            for variant in gts.values():
                if isinstance(variant, dict):
                    for key in ['input_port_width', 'output_port_width',
                               'clock_port_polarity', 'reset_port_polarity_sync']:
                        if key in variant and isinstance(variant[key], list):
                            # Convert inner lists to tuples, then create set
                            # Original: {('port', width), ...} → JSON: [['port', width], ...] → Back: {('port', width), ...}
                            variant[key] = set(
                                tuple(item) if isinstance(item, list) else item
                                for item in variant[key]
                            )
        else:
            raise ValueError(f"Unexpected ground_truth type: {type(ground_truth)}")

        # Extract assistant response from chat template wrapper (format-aware)
        # This handles Llama, Qwen, GPT OSS, and raw formats
        if format_type == "auto":
            format_type = detect_format(solution_str)

        handler = get_format_handler(format_type)
        solution_str = handler.extract_assistant_response(solution_str, model_type="auto")

        # Check format (must have proper tags/channels)
        if not check_format(solution_str, format_type=format_type):
            logger.warning("Invalid format: missing or incorrect tags/channels")
            return {
                "score": 0.0,
                "reward_fmt": 0.0,
                "reward_think": 0.0,
                "error": "Invalid format"
            }

        # Extract Verilog code from answer block
        extracted_answer = extract_verilog(solution_str)
        if not extracted_answer:
            logger.warning("No Verilog code extracted from answer")
            return {
                "score": 0.0,
                "reward_fmt": 1.0,  # Format is correct, but no code
                "reward_think": 1.0,
                "error": "No Verilog code found"
            }

        # Test against all ground truth variants
        # Some problems may have multiple acceptable solutions
        rewards = []
        verification_results = []

        for variant_key, gt_variant in gts.items():
            logger.info(f"Testing against ground truth variant: {variant_key}")

            # Extract golden code and port information
            golden_code = gt_variant.get('code')
            if not golden_code:
                logger.error(f"No golden code in variant {variant_key}")
                continue

            port_info = (
                gt_variant.get('input_port_width', set()),
                gt_variant.get('output_port_width', set()),
                gt_variant.get('clock_port_polarity', set()),
                gt_variant.get('reset_port_polarity_sync', set())
            )

            # Parse golden code
            try:
                v = eda_tools(quiet=True)
                golden_top = v.auto_top(golden_code)
            except Exception as e:
                logger.error(f"Failed to parse GOLDEN Verilog code: {e}")
                logger.error(f"Golden code (first 300 chars): {golden_code[:300]!r}...")

                verification_results.append({
                    "correct": False,
                    "parse_error": f"Golden code parsing failed: {str(e)}",
                    "golden_code_preview": golden_code[:200] if golden_code else None,
                })
                rewards.append(0.0)
                continue

            # Parse generated code
            try:
                gate_top = v.auto_top(extracted_answer)
            except Exception as e:
                logger.error(f"Failed to parse GENERATED Verilog code: {e}")
                logger.error(f"Generated code (first 300 chars): {extracted_answer[:300] if extracted_answer else None!r}...")
                logger.error(f"Golden code for comparison (first 300 chars): {golden_code[:300]!r}...")

                verification_results.append({
                    "correct": False,
                    "parse_error": f"Generated code parsing failed: {str(e)}",
                    "extracted_answer_preview": extracted_answer[:200] if extracted_answer else None,
                    "golden_code_preview": golden_code[:200] if golden_code else None,
                })
                rewards.append(0.0)
                continue

            # Verify via Sandbox Fusion
            result = verify_verilog_via_sandbox(
                golden_code=golden_code,
                dut_code=extracted_answer,
                golden_top=golden_top,
                gate_top=gate_top,
                port_info=port_info,
                sandbox_fusion_url=sandbox_fusion_url,
                concurrent_semaphore=concurrent_semaphore
            )

            verification_results.append(result)
            reward = 1.0 if result.get("correct", False) else 0.0
            rewards.append(reward)

            # Early exit if we found a correct match
            if reward == 1.0:
                logger.info(f"Found correct match with variant {variant_key}")
                break

        # Compute final score (max across all variants)
        final_score = max(rewards) if rewards else 0.0

        # Check if we should raise an exception for actual errors (not functional mismatch)
        if final_score == 0.0 and verification_results:
            # Distinguish between functional mismatch (normal) and actual errors
            has_functional_mismatch = False
            error_msgs = []

            for result in verification_results:
                if not result.get("correct", False):
                    # Functional mismatch: code ran successfully but answer is wrong
                    # (has error_rate but no api_error, exception, or parse_error)
                    if ("error_rate" in result and
                        "exception" not in result and
                        "api_error" not in result and
                        "parse_error" not in result):
                        has_functional_mismatch = True
                        break

                    # Collect actual error messages
                    if "api_error" in result:
                        error_msgs.append(result["api_error"])
                    elif "exception" in result:
                        error_msgs.append(result["exception"])
                    elif "parse_error" in result:
                        error_msgs.append(result["parse_error"])
                    elif result.get("api_status") and result["api_status"] != "Success":
                        error_msgs.append(f"API status: {result['api_status']}")

            # Raise exception if all variants failed with actual errors (not just wrong answers)
            if not has_functional_mismatch and error_msgs:
                raise RuntimeError(f"Verilog verification failed: {error_msgs[0]}")

        return {
            "score": final_score,
            "reward_fmt": 1.0,  # Format is correct if we got here
            "reward_think": 1.0,  # Thinking structure is present
            "num_variants_tested": len(rewards),
            "num_variants_passed": sum(rewards),
            "verification_results": verification_results[:3]  # Include first 3 for debugging
        }

    except Exception as e:
        logger.error(f"Error in compute_score: {e}", exc_info=True)
        return {
            "score": 0.0,
            "reward_fmt": 0.0,
            "reward_think": 0.0,
            "error": f"Exception: {str(e)}"
        }
