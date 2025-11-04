import regex as re
from typing import Tuple, Optional

from .format_handlers import detect_format, get_format_handler


FORMATS_REGEX = [
    [r"\\boxed\{", "}"],
]


def parse_think(text: str, format_type: str = "auto") -> Tuple[str, bool]:
    """
    Remove thinking/reasoning section from text.

    This function is format-aware and supports both XML (<think>) and GPT OSS (<|channel|>analysis) formats.

    Args:
        text: Full response text
        format_type: Format type ("xml", "gpt_oss", or "auto" for auto-detection)

    Returns:
        Tuple of (text_without_thinking, success)
        - text_without_thinking: Text with thinking section removed
        - success: True if thinking section was properly formatted (or absent)

    Examples:
        >>> parse_think("<think>reasoning</think>\\nThe answer is 42")
        ('The answer is 42', True)

        >>> parse_think("<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>\\n<|start|>assistant<|channel|>final<|message|>42<|return|>")
        ('<|start|>assistant<|channel|>final<|message|>42<|return|>', True)
    """
    if format_type == "auto":
        format_type = detect_format(text)

    handler = get_format_handler(format_type)
    thinking_content, success = handler.extract_thinking(text)

    if thinking_content is not None:
        # Remove thinking section from text
        text_without_thinking = handler.remove_thinking(text)
        return text_without_thinking, success
    else:
        # No thinking section found - return original text
        return text, success


def parse_answer(text: str, format_type: str = "auto") -> Tuple[str, int]:
    """
    Extract final answer from text.

    This function is format-aware and supports:
    - XML format: Looks for \\boxed{answer} or <response>/<answer> tags
    - GPT OSS format: Looks for <|channel|>final content

    Args:
        text: Full response text
        format_type: Format type ("xml", "gpt_oss", or "auto" for auto-detection)

    Returns:
        Tuple of (answer, format_index)
        - answer: Extracted answer or original text if no format matched
        - format_index: Index of matched format (-1 if no format matched, 0 for \\boxed, 1 for final channel)

    Examples:
        >>> parse_answer("The answer is \\\\boxed{42}")
        ('42', 0)

        >>> parse_answer("<|start|>assistant<|channel|>final<|message|>The answer is 42<|return|>")
        ('The answer is 42', 1)
    """
    if not isinstance(text, str):
        return text, -1

    # Detect format if auto
    if format_type == "auto":
        format_type = detect_format(text)

    # For GPT OSS format, try to extract from final channel first
    if format_type == "gpt_oss":
        handler = get_format_handler(format_type)
        final_response = handler.extract_final_response(text)
        if final_response:
            # Still try to extract \boxed{} from final response if present
            for i, pattern in enumerate(FORMATS_REGEX):
                match = extract_answer_recursive(final_response, pattern[0], pattern[1])
                if match:
                    return match, i
            # Return final channel content
            return final_response, 1

    # Try standard answer extraction patterns (e.g., \boxed{})
    last_match = None
    last_index = -1

    for i, pattern in enumerate(FORMATS_REGEX):
        match = extract_answer_recursive(text, pattern[0], pattern[1])
        if match:
            last_match = match
            last_index = i

    if last_match:
        return last_match, last_index

    # For XML format, try to extract from <response> or <answer> tags
    if format_type == "xml":
        handler = get_format_handler(format_type)
        final_response = handler.extract_final_response(text)
        if final_response:
            return final_response, 1

    return text, -1


def extract_answer_recursive(text: str, start_pattern: str, end_pattern: str) -> str:

    def find_matching_brace(s: str, start_idx: int) -> int:
        count = 0
        for i in range(start_idx, len(s)):
            if s[i] in ["(", "{", "["]:
                count += 1
            elif s[i] in [")", "}", "]"]:
                count -= 1
                if count == 0:
                    return i
        return -1

    matches = list(re.finditer(start_pattern, text))
    if not matches:
        return None

    results = []
    for match in matches:
        start_paren = match.end() - 1
        end_paren = find_matching_brace(text, start_paren)

        if end_paren != -1:
            after_paren = text[end_paren:]
            if after_paren.startswith(end_pattern):
                extracted = text[start_paren + 1 : end_paren]
                results.append(extracted)

    return results[-1] if results else None
