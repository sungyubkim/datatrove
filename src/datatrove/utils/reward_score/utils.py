import regex as re

FORMATS_REGEX = [
    [r"\\boxed\{", "}"],
]


def parse_think(text: str) -> str:
    tag_count_start = text.count("<think>")
    if tag_count_start > 0:
        tag_count_end = text.count("</think>")
        if tag_count_start == tag_count_end == 1:
            text = text.split("</think>")[-1].strip()
            return text, True
        else:
            return text, False
    else:
        return text, True


def parse_answer(text: str) -> str:
    if not isinstance(text, str):
        return text, -1

    last_match = None
    last_index = -1

    for i, pattern in enumerate(FORMATS_REGEX):
        match = extract_answer_recursive(text, pattern[0], pattern[1])
        if match:
            last_match = match
            last_index = i

    if last_match:
        return last_match, last_index
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
