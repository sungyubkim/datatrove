"""Domain-specific instruction format parsers.

This module provides specialized parsers for different code generation domains:
- Verilog HDL: Parses signal interface tables and extracts implicit constraints
- Python Code: Handles minimal format with function signatures
- Competitive Programming: Parses explicit section headers (CodeContests format)

Each parser extracts:
- Problem description
- Constraints (explicit or inferred from tables/descriptions)
- Input/Output formats
- Examples
- Domain-specific information (signal tables, function signatures, etc.)
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List


class InstructionParser(ABC):
    """Base class for domain-specific instruction parsers."""

    @abstractmethod
    def parse(self, text: str) -> Dict[str, str]:
        """Parse instruction into standardized sections.

        Args:
            text: The instruction text to parse

        Returns:
            Dictionary with keys: problem, constraints, input_format,
            output_format, examples, domain_specific
        """
        pass

    def generate_implementation_hints(self) -> str:
        """Generate domain-specific implementation hints for sandbox grading.

        Returns:
            Implementation requirements text optimized for sandbox_fusion grading
        """
        return ""  # Default: no hints


class VerilogInstructionParser(InstructionParser):
    """Parser for Verilog HDL instructions (codev-r1-verl, hardware design format).

    Verilog instructions typically contain:
    - Problem description with behavioral specifications
    - Signal interface table (NOT "Input Format" section)
    - Implicit constraints in description and table

    This parser extracts:
    - Input/Output signals from table
    - Bit width constraints
    - Timing constraints (delays, clock periods)
    - Behavioral constraints (reset types, edge-triggered, synchronous/asynchronous)
    - Memory/capacity constraints
    """

    def parse(self, text: str) -> Dict[str, str]:
        """Parse Verilog instruction with signal interface table."""
        sections = {
            "problem": "",
            "constraints": "",
            "input_format": "",
            "output_format": "",
            "examples": "",
            "domain_specific": "",
        }

        # Step 1: Detect and extract signal table
        # Pattern: | Signal Name | Direction | Width | Description |
        table_pattern = re.compile(
            r'(\|[^\n]*Signal[^\n]*\|[^\n]*\|[^\n]*\|[^\n]*\|.*?)(?:\n\n|\Z)',
            re.DOTALL | re.IGNORECASE
        )

        table_match = table_pattern.search(text)

        if table_match:
            # Extract problem (everything before table)
            table_start = table_match.start()
            sections["problem"] = text[:table_start].strip()

            # Extract table text
            table_text = table_match.group(1)
            sections["domain_specific"] = f"\n\n{table_text.strip()}"

            # Parse signal table
            input_signals, output_signals, signal_constraints = self._parse_signal_table(table_text)

            # Format input/output sections
            if input_signals:
                sections["input_format"] = (
                    "**Input Signals:**\n" +
                    "\n".join(f"- {s}" for s in input_signals)
                )

            if output_signals:
                sections["output_format"] = (
                    "**Output Signals:**\n" +
                    "\n".join(f"- {s}" for s in output_signals)
                )

            # Extract additional constraints from problem description
            problem_constraints = self._extract_constraints_from_description(sections["problem"])

            # Combine all constraints
            all_constraints = signal_constraints + problem_constraints
            if all_constraints:
                sections["constraints"] = "\n".join(f"- {c}" for c in all_constraints)
        else:
            # No table found, use full text as problem
            sections["problem"] = text.strip()

            # Try to extract constraints anyway
            problem_constraints = self._extract_constraints_from_description(text)
            if problem_constraints:
                sections["constraints"] = "\n".join(f"- {c}" for c in problem_constraints)

        return sections

    def _parse_signal_table(self, table_text: str) -> tuple[List[str], List[str], List[str]]:
        """Parse signal interface table into input/output signals and constraints.

        Args:
            table_text: The markdown table text

        Returns:
            Tuple of (input_signals, output_signals, constraints)
        """
        input_signals = []
        output_signals = []
        constraints = []

        # Split into rows
        rows = [row.strip() for row in table_text.split('\n') if row.strip()]

        for row in rows:
            # Skip if not enough columns
            if row.count('|') < 4:
                continue

            # Parse columns
            parts = [p.strip() for p in row.split('|')]
            parts = [p for p in parts if p]  # Remove empty strings

            if len(parts) < 3:
                continue

            # Skip header/separator rows
            signal_name = parts[0]
            if not signal_name or signal_name in ('Signal Name', 'Signal', '---', ''):
                continue
            if signal_name.startswith('---') or signal_name.startswith('==='):
                continue

            # Extract signal info
            direction = parts[1] if len(parts) > 1 else ""
            width = parts[2] if len(parts) > 2 else ""
            description = parts[3] if len(parts) > 3 else ""

            # Format signal entry
            signal_entry = f"`{signal_name}` ({width} bits): {description}" if description else f"`{signal_name}` ({width} bits)"

            # Categorize by direction
            if 'input' in direction.lower():
                input_signals.append(signal_entry)
                # Extract bit width constraint
                if width and any(char.isdigit() for char in width):
                    constraints.append(f"{signal_name} width: {width} bits")
            elif 'output' in direction.lower() or 'out' in direction.lower():
                output_signals.append(signal_entry)
                # Extract bit width constraint
                if width and any(char.isdigit() for char in width):
                    constraints.append(f"{signal_name} width: {width} bits")

            # Extract timing constraints from description
            timing_patterns = [
                (r'delayed?\s+by\s+(\d+)\s*(ps|ns|picoseconds|nanoseconds)', 'Timing delay on {}: {}'),
                (r'(\d+)\s*(ps|ns|picoseconds|nanoseconds)\s+delay', 'Timing delay on {}: {}'),
            ]

            for pattern, template in timing_patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    delay_value = f"{match.group(1)} {match.group(2)}"
                    constraints.append(template.format(signal_name, delay_value))

        return input_signals, output_signals, constraints

    def _extract_constraints_from_description(self, text: str) -> List[str]:
        """Extract implicit constraints from problem description.

        Looks for:
        - Memory/capacity specifications
        - Timing requirements
        - Behavioral keywords (reset types, synchronous/asynchronous, edge-triggered)
        - Bit width specifications
        - Frequency/clock specifications

        Args:
            text: The problem description text

        Returns:
            List of constraint strings
        """
        constraints = []

        # Memory/capacity constraints
        capacity_patterns = [
            r'capacity\s+of\s+(\d+)\s*(\w+)',
            r'(\d+)[-\s](\w+)\s+(?:memory|storage|capacity)',
            r'memory\s+size[:\s]+(\d+)\s*(\w+)',
        ]

        for pattern in capacity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    constraints.append(f"Memory capacity: {match.group(1)} {match.group(2)}")
                break

        # Timing constraints
        timing_patterns = [
            (r'(?:delayed?|delay)\s+(?:of|by)?\s*(\d+)\s*(ps|ns|picoseconds|nanoseconds)',
             'Timing delay: {} {}'),
            (r'clock\s+period[:\s]+(\d+)\s*(ps|ns|ms|us)',
             'Clock period: {} {}'),
            (r'frequency[:\s]+(\d+)\s*(MHz|GHz|KHz|Hz)',
             'Frequency: {} {}'),
        ]

        for pattern, template in timing_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                constraint_text = template.format(match.group(1), match.group(2))
                if constraint_text not in constraints:
                    constraints.append(constraint_text)

        # Behavioral constraints
        behavioral_keywords = [
            (r'(?:on\s+)?(?:the\s+)?rising\s+edge', 'Rising edge triggered'),
            (r'(?:on\s+)?(?:the\s+)?falling\s+edge', 'Falling edge triggered'),
            (r'active[-\s]high\s+reset', 'Active-high reset'),
            (r'active[-\s]low\s+reset', 'Active-low reset'),
            (r'synchronous\s+(?:reset|operation)', 'Synchronous operation'),
            (r'asynchronous\s+(?:reset|operation)', 'Asynchronous operation'),
            (r'posedge\s+\w+', 'Positive edge sensitive'),
            (r'negedge\s+\w+', 'Negative edge sensitive'),
        ]

        for pattern, constraint_text in behavioral_keywords:
            if re.search(pattern, text, re.IGNORECASE):
                if constraint_text not in constraints:
                    constraints.append(constraint_text)

        # Bit width specifications (not already in table)
        bit_width_matches = re.finditer(r'(\d+)[-\s]bit(?:\s+(\w+))?', text, re.IGNORECASE)
        for match in bit_width_matches:
            width = match.group(1)
            component = match.group(2) if match.group(2) else "value"
            constraint_text = f"{component.capitalize()} width: {width} bits"
            if constraint_text not in constraints:
                constraints.append(constraint_text)

        return constraints

    def generate_implementation_hints(self) -> str:
        """Generate Verilog-specific implementation hints."""
        return """**Code Format:**
- Wrap your Verilog code in markdown code blocks with ```verilog
- Module name should match the problem specification
- Include all required input/output ports

**Module Structure:**
- Define module with exact port names and widths from Signal Interface table
- Implement behavioral logic matching the problem description
- Use proper always blocks for sequential logic (e.g., `always @(posedge clk)`)
- Use assign statements for combinational logic

**Common Pitfalls:**
- Ensure port directions match (input vs output)
- Check signal widths match specifications
- Verify reset polarity (active-high vs active-low)
- Test edge-triggered vs level-sensitive behavior"""


class PythonCodeInstructionParser(InstructionParser):
    """Parser for Python code instructions (kodcode, LeetCode, HumanEval format).

    Python code instructions are typically minimal:
    - Problem description only
    - Sometimes includes function signature
    - Rarely has explicit constraints or IO format sections

    This parser:
    - Uses full text as problem description
    - Extracts function signature if present
    - Looks for example-based constraints (1 ≤ n ≤ 10^5)
    """

    def parse(self, text: str) -> Dict[str, str]:
        """Parse Python code instruction (usually minimal format)."""
        sections = {
            "problem": text.strip(),
            "constraints": "",
            "input_format": "",
            "output_format": "",
            "examples": "",
            "domain_specific": "",
        }

        # Try to extract function signature
        func_sig_pattern = r'def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*\w+)?:'
        func_sig_match = re.search(func_sig_pattern, text)

        if func_sig_match:
            # Extract the full function signature with context
            start = func_sig_match.start()
            end = func_sig_match.end()

            # Try to get the full signature including docstring if present
            func_text = text[start:end]

            sections["domain_specific"] = (
                "**Function Signature:**\n"
                f"```python\n{func_text}\n    ...\n```"
            )

        # Try to extract constraints (competitive programming style)
        constraint_patterns = [
            r'(\d+\s*[≤<=]\s*\w+\s*[≤<=]\s*\d+)',  # 1 ≤ n ≤ 10^5
            r'(\d+\s*[≤<=]\s*\w+\.\w+\s*[≤<=]\s*\d+)',  # 1 ≤ arr.length ≤ 1000
        ]

        found_constraints = []
        for pattern in constraint_patterns:
            for match in re.finditer(pattern, text):
                constraint = match.group(1)
                if constraint not in found_constraints:
                    found_constraints.append(constraint)

        if found_constraints:
            sections["constraints"] = "\n".join(f"- {c}" for c in found_constraints)

        # Try to extract examples
        example_pattern = re.compile(
            r'(?:Example|Sample|Input|Output)s?:?\s*(.*?)(?:\n\n|Example \d|\Z)',
            re.IGNORECASE | re.DOTALL
        )

        example_matches = example_pattern.finditer(text)
        examples_text = []

        for match in example_matches:
            example_content = match.group(1).strip()
            if example_content and len(example_content) > 10:  # Filter out short matches
                examples_text.append(example_content)

        if examples_text:
            sections["examples"] = "\n\n".join(examples_text)

        return sections

    def generate_implementation_hints(self) -> str:
        """Generate Python-specific implementation hints."""
        return """**Code Format:**
- Wrap your Python code in markdown code blocks with ```python
- Match the exact function signature if provided
- Return the result (don't print unless explicitly required)

**Implementation Guidelines:**
- Use appropriate data structures (list, dict, set, deque, heap)
- Import standard library modules as needed (collections, heapq, bisect, etc.)
- Follow time/space complexity constraints if specified
- Handle edge cases (empty input, single element, duplicates)

**Common Pitfalls:**
- Don't modify function signature unless problem allows it
- Avoid unnecessary print statements (return instead)
- Check if problem requires in-place modification vs new data structure
- Verify return type matches expected output format"""


class CompetitiveProgrammingParser(InstructionParser):
    """Parser for competitive programming instructions (CodeContests, TACO, APPS format).

    Competitive programming problems have explicit sections:
    - Problem/Description
    - Input Format
    - Output Format
    - Constraints
    - Examples/Sample Test Cases

    This parser looks for explicit section headers and extracts content.
    """

    def parse(self, text: str) -> Dict[str, str]:
        """Parse competitive programming instruction with explicit sections."""
        sections = {
            "problem": "",
            "constraints": "",
            "input_format": "",
            "output_format": "",
            "examples": "",
            "domain_specific": "",
        }

        # Define section headers for each category
        section_patterns = {
            'problem': [
                r'Problem\s*(?:Statement|Description)?:?',
                r'Description:?',
                r'Task:?',
            ],
            'constraints': [
                r'Constraints?:',
                r'Limitations?:',
                r'Restrictions?:',
                r'Note:',
            ],
            'input_format': [
                r'Input\s*(?:Format|Description)?:?',
                r'Input:',
            ],
            'output_format': [
                r'Output\s*(?:Format|Description)?:?',
                r'Output:',
            ],
            'examples': [
                r'Examples?:?',
                r'Sample\s+(?:Input|Output|Test\s+Cases?):?',
            ],
        }

        # Build a combined pattern to find all section headers
        all_patterns = []
        for patterns in section_patterns.values():
            all_patterns.extend(patterns)

        combined_pattern = '|'.join(f'({p})' for p in all_patterns)
        header_regex = re.compile(f'(?:^|\n)({combined_pattern})', re.MULTILINE | re.IGNORECASE)

        # Find all header positions
        headers = [(m.start(), m.group(1), m.end()) for m in header_regex.finditer(text)]

        if not headers:
            # No explicit sections found, use full text as problem
            sections["problem"] = text.strip()
            return sections

        # Extract content between headers
        for i, (start, header_text, header_end) in enumerate(headers):
            # Determine which section this header belongs to
            section_key = None
            for key, patterns in section_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, header_text, re.IGNORECASE):
                        section_key = key
                        break
                if section_key:
                    break

            if not section_key:
                continue

            # Extract content until next header
            if i + 1 < len(headers):
                next_start = headers[i + 1][0]
                content = text[header_end:next_start].strip()
            else:
                content = text[header_end:].strip()

            # Accumulate content (some sections may have multiple headers)
            if sections[section_key]:
                sections[section_key] += "\n\n" + content
            else:
                sections[section_key] = content

        # If no problem section found, use content before first header
        if not sections["problem"] and headers:
            first_header_start = headers[0][0]
            sections["problem"] = text[:first_header_start].strip()

        return sections

    def generate_implementation_hints(self) -> str:
        """Generate competitive programming-specific implementation hints."""
        return """**Code Format:**
- Wrap your code in markdown code blocks (```python, ```cpp, ```java, etc.)
- Read from stdin and write to stdout (unless using function signature)
- Handle multiple test cases if specified

**Input/Output Guidelines:**
- Parse input exactly as described in Input Format section
- Output must match Output Format precisely (spacing, newlines, formatting)
- Use appropriate data types (int, long, double) based on constraints
- Handle large inputs efficiently (avoid TLE - Time Limit Exceeded)

**Common Pitfalls:**
- Don't add extra whitespace or debug output
- Check integer overflow for large constraints (use long/int64)
- Optimize algorithm based on time complexity constraints
- Test with boundary cases (min/max values, edge cases from constraints)"""


def get_parser_for_domain(domain: str) -> InstructionParser:
    """Get the appropriate parser for a domain.

    Args:
        domain: Domain name ("verilog", "python", "competitive", "auto", etc.)

    Returns:
        Appropriate InstructionParser instance
    """
    domain_lower = domain.lower()

    if domain_lower in ("verilog", "hdl", "verilog-hdl"):
        return VerilogInstructionParser()
    elif domain_lower in ("python", "python-code"):
        return PythonCodeInstructionParser()
    elif domain_lower in ("competitive", "codecontests", "programming"):
        return CompetitiveProgrammingParser()
    else:
        # Default to competitive parser (handles explicit sections well)
        # For truly generic cases, could create GenericInstructionParser
        return CompetitiveProgrammingParser()
