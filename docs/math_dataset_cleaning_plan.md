# Math Dataset Cleaning Plan

## Executive Summary

This document outlines the comprehensive plan to clean 4 VERL math datasets hosted on HuggingFace Hub. The cleaning will be performed sequentially with user review at each stage.

**Last Updated**: 2025-11-07
**Status**: Implementation In Progress

---

## Target Datasets

| Dataset | Size | Current Quality | Target Quality | Priority |
|---------|------|-----------------|----------------|----------|
| sungyub/orz-math-72k-verl | 121K | 63% clean | 85% clean | 1 (First) |
| sungyub/openr1-math-verl | 106K | 89% clean | 97% clean | 2 |
| sungyub/skywork-or1-math-verl | 209K | 87% clean | 96% clean | 3 |
| sungyub/dapo-math-17k-verl | 17K | 99% clean | 99.5% clean | 4 (Last) |

---

## Analysis Summary

### Common Issues Across Datasets

1. **Problem Numbering Prefixes** (2-15% affected)
   - Patterns: "Problem 6.", "Question 230,", "8.3", "G1.4"
   - Action: Remove with regex

2. **Contest Metadata** (1-8% affected)
   - Patterns: "2004 AIME Problem 3", "20th APMC 1997 Problem 3"
   - Action: Remove from text

3. **Image References** (1-5% affected)
   - Patterns: "[asy]", "Figure 1", "[*.gif]", markdown images
   - Action: Detect and track in statistics (data preserved)

4. **Point Allocations** (1-4% affected)
   - Patterns: "(8 points)", "[15 points]"
   - Action: Remove with regex

5. **Markdown Headers** (1-5% in OpenR1)
   - Patterns: "## Problem Statement", "## Zadatak"
   - Action: Remove with regex

### Dataset-Specific Issues

#### ORZ-Math (Most Issues - 37% affected)
- **Vague ground truths** (19%): Single letters 'A', 'B', 'C', 'D' → **Keep as-is**
- **Non-English text** (8%): Chinese characters → **Keep as-is**
- **LaTeX issues** (5%): Inconsistent formatting
- **Incomplete problems** (5-7%): Missing context

#### OpenR1-Math
- Highest prefix removal rate (11% problem numbers)
- Markdown headers (5%)
- Most improvement potential

#### Skywork-OR1
- Relatively clean (only 13% affected)
- Mainly figure references (3%)

#### DAPO-Math
- Almost perfect (99% clean)
- Minimal cleaning needed

---

## Implementation Architecture

### Phase 1: MathDatasetCleaner Formatter

**File**: `src/datatrove/pipeline/formatters/math_cleaner.py`

**Class Design**:
```python
class MathDatasetCleaner(BaseFormatter):
    """Unified formatter for cleaning VERL math datasets.

    Conservative approach: Only remove clear artifacts, preserve content.
    NO modifications to extra_info metadata.
    """

    def __init__(
        self,
        remove_problem_numbers: bool = True,
        remove_point_allocations: bool = True,
        remove_contest_metadata: bool = True,
        remove_markdown_headers: bool = True,
        remove_image_references: bool = False,  # Only detect, don't remove
        normalize_whitespace: bool = True,
        log_cleaning_stats: bool = True,
    ):
        ...
```

**Preset Configurations**:

```python
CLEANING_PRESETS = {
    "orz-math": {
        "remove_problem_numbers": True,      # 10-15%
        "remove_point_allocations": True,    # 2-3%
        "remove_contest_metadata": True,     # 5-8%
        "remove_markdown_headers": False,
        "remove_image_references": False,    # Detect only, 1-2%
        "normalize_whitespace": True,
    },
    "openr1-math": {
        "remove_problem_numbers": True,      # 11%
        "remove_point_allocations": True,    # 4%
        "remove_contest_metadata": True,     # 5%
        "remove_markdown_headers": True,     # 5%
        "remove_image_references": False,    # Detect only, 5%
        "normalize_whitespace": True,
    },
    "skywork-or1": {
        "remove_problem_numbers": True,      # 2%
        "remove_point_allocations": False,
        "remove_contest_metadata": False,
        "remove_markdown_headers": False,
        "remove_image_references": False,    # Detect only, 3%
        "normalize_whitespace": True,
    },
    "dapo-math": {
        "remove_problem_numbers": True,      # <1%
        "remove_point_allocations": False,
        "remove_contest_metadata": False,
        "remove_markdown_headers": False,
        "remove_image_references": False,
        "normalize_whitespace": True,
    }
}
```

**Regex Patterns**:

```python
# Problem numbering patterns (from all 4 datasets)
PROBLEM_NUMBER_PATTERNS = [
    r"^Problem\s+\d+[\.:]\s*",               # "Problem 6."
    r"^Question\s+\d+[\.:,]\s*",             # "Question 230,"
    r"^Exercise\s+\d+[\.:]\s*",              # "Exercise 12:"
    r"^\d+\.\d+[\.:]\s*",                    # "8.3:", "7.1:"
    r"^[A-Z]\d+\.\d+[\.:]\s*",               # "G1.4:", "I2.1:"
    r"^[A-Z]\d+\s*\([A-Z]+\)\s*",            # "A2 (RUS)"
    r"^Example\s+\d+[\.:]\s*",               # "Example 31:"
]

# Contest metadata patterns
CONTEST_METADATA_PATTERNS = [
    r"^\d+(?:st|nd|rd|th)\s+[A-Z]+\s+\d{4}\s+Problem\s+\d+",
    # "20th APMC 1997 Problem 3"

    r"\(?(?:19|20)\d{2}[,\s]+[^)]{0,80}(?:Competition|Entrance Examination)[^)]*\)?",
    # "(2004 College Entrance Examination...)"

    r"(?:19|20)\d{2},?\s+(?:AIME|IMO|AMC|USAMO|BMO)\s*(?:Problem)?\s*\d+",
    # "2004 AIME Problem 3", "1997 IMO 3"
]

# Point allocations
POINT_ALLOCATION_PATTERNS = [
    r"\(\d+\s*points?\)",                    # "(8 points)", "(1 point)"
    r"\[\d+\s*points?\]",                    # "[15 points]"
]

# Markdown headers
MARKDOWN_HEADER_PATTERNS = [
    r"^##\s+Problem Statement\s*\n",
    r"^##\s+Zadatak.*\n",
    r"^##\s+Solution\s*\n",
]

# Image reference detection (for statistics only)
IMAGE_REFERENCE_PATTERNS = [
    r"!\[.*?\]\(https?://[^\)]+\)",          # Markdown images
    r"\[asy\].*?\[/asy\]",                   # Asymptote diagrams
    r"\[[\w\-]+\.(gif|png|jpg|jpeg)\]",      # [file.gif]
    r"(?:see|refer to|shown in)\s+(?:Figure|Diagram|figure|diagram)\s+\d+",
]
```

**Key Implementation Notes**:
1. **NO LaTeX escaping changes**: `\\frac` stays as `\\frac` (correct JSON format)
2. **NO extra_info modifications**: Do not add new fields to metadata
3. **Clean `prompt[0]['content']` only**: Preserve `ground_truth` and other fields
4. **Statistics in memory**: Track cleaning stats but don't write to dataset
5. **Schema preservation**: Output must match input schema exactly

---

### Phase 2: Sequential Cleaning Script

**File**: `scripts/clean_math_dataset_single.py`

**Script Features**:
- Download dataset from HuggingFace Hub
- Apply MathDatasetCleaner with appropriate preset
- Save cleaned data locally in parquet format
- Generate before/after comparison report
- Validate schema matches original

**Usage**:
```bash
# Clean ORZ-Math (first)
python scripts/clean_math_dataset_single.py \
    --dataset sungyub/orz-math-72k-verl \
    --preset orz-math \
    --output ./output/orz-math-cleaned/

# After user review, clean OpenR1-Math
python scripts/clean_math_dataset_single.py \
    --dataset sungyub/openr1-math-verl \
    --preset openr1-math \
    --output ./output/openr1-math-cleaned/

# Continue for Skywork-OR1 and DAPO-Math...
```

**Report Format**:
```
====================================
Cleaning Report: orz-math-72k-verl
====================================
Timestamp: 2025-11-07 10:30:00
Preset: orz-math

Original samples: 120,592
Cleaned samples: 120,592 (100% retained)

Changes Applied:
------------------------------------
Problem numbers removed:   15,234 samples (12.6%)
Point allocations removed:  2,891 samples (2.4%)
Contest metadata removed:   7,245 samples (6.0%)
Markdown headers removed:       0 samples (0.0%)
Image references detected:  1,446 samples (1.2%)

Total samples modified:    25,370 samples (21.0%)
Clean samples (no changes): 95,222 samples (79.0%)

Before/After Examples:
------------------------------------
[Example 1]
BEFORE: "8.3 In the tetrahedron $ABCD$..."
AFTER:  "In the tetrahedron $ABCD$..."

[Example 2]
BEFORE: "20th APMC 1997 Problem 3 The 97 numbers..."
AFTER:  "The 97 numbers..."

... (10-15 more examples)

Schema Validation:
------------------------------------
✓ All columns present
✓ Data types match
✓ Row count matches
✓ Parquet format valid

Output Location:
------------------------------------
./output/orz-math-cleaned/data-00000-of-00001.parquet

Next Steps:
------------------------------------
1. Review the before/after examples above
2. Inspect samples in the output directory
3. Approve to proceed with OpenR1-Math cleaning
```

---

### Phase 3: Testing

**File**: `tests/pipeline/formatters/test_math_cleaner.py`

**Test Cases** (from real dataset samples):

```python
class TestMathDatasetCleaner:
    """Tests using actual problematic samples from datasets."""

    def test_orz_problem_number_removal(self):
        """ORZ Sample 10: '8.3 In the tetrahedron...'"""
        ...

    def test_orz_contest_metadata_removal(self):
        """ORZ Sample 21: '24th Eötvös 1917 Problem 2...'"""
        ...

    def test_openr1_problem_number_removal(self):
        """OpenR1 Sample 8: 'Problem 6. (8 points) In the plane...'"""
        ...

    def test_openr1_markdown_header_removal(self):
        """OpenR1 Sample 15: '## Problem Statement\n\nCalculate...'"""
        ...

    def test_skywork_problem_number_removal(self):
        """Skywork Sample 36: 'Question 230, Let $S$...'"""
        ...

    def test_latex_preservation(self):
        """Verify LaTeX unchanged across all datasets."""
        ...

    def test_schema_preservation(self):
        """Verify output schema matches input schema."""
        ...

    def test_ground_truth_unchanged(self):
        """Verify ground_truth field is never modified."""
        ...

    def test_extra_info_unchanged(self):
        """Verify extra_info is never modified."""
        ...
```

---

## Execution Plan

### Step-by-Step Workflow

#### **Step 1: Implementation** (Current Phase)
- [ ] Create `docs/math_dataset_cleaning_plan.md` (this file)
- [ ] Implement `src/datatrove/pipeline/formatters/math_cleaner.py`
- [ ] Write comprehensive tests in `tests/pipeline/formatters/test_math_cleaner.py`
- [ ] Create `scripts/clean_math_dataset_single.py`
- [ ] Run tests: `uv run pytest -sv tests/pipeline/formatters/test_math_cleaner.py`

#### **Step 2: Clean ORZ-Math** (Highest Priority - 63% → 85%)
- [ ] Run cleaning script with preset="orz-math"
- [ ] Generate comparison report
- [ ] Review 15-20 before/after samples
- [ ] Validate schema and row count
- [ ] **USER REVIEW CHECKPOINT** ✋
- [ ] Approve to proceed OR adjust patterns and rerun

#### **Step 3: Clean OpenR1-Math** (89% → 97%)
- [ ] Run cleaning script with preset="openr1-math"
- [ ] Generate comparison report
- [ ] Review samples
- [ ] **USER REVIEW CHECKPOINT** ✋

#### **Step 4: Clean Skywork-OR1** (87% → 96%)
- [ ] Run cleaning script with preset="skywork-or1"
- [ ] Generate comparison report
- [ ] Review samples
- [ ] **USER REVIEW CHECKPOINT** ✋

#### **Step 5: Clean DAPO-Math** (99% → 99.5%)
- [ ] Run cleaning script with preset="dapo-math"
- [ ] Generate comparison report
- [ ] Review samples
- [ ] **FINAL REVIEW** ✋

#### **Step 6: Upload (Optional)**
- [ ] Create upload helper script
- [ ] Upload cleaned datasets to HuggingFace Hub
- [ ] Update dataset cards with cleaning information

---

## Design Decisions & Rationale

### What We Clean

1. **Problem numbering prefixes** ✅
   - Rationale: Metadata, not part of problem content
   - Impact: 2-15% of samples across datasets
   - Risk: Low (easily reversed, clear pattern)

2. **Contest metadata** ✅
   - Rationale: Source information, not problem content
   - Impact: 1-8% of samples
   - Risk: Low (information loss acceptable)

3. **Point allocations** ✅
   - Rationale: Grading metadata, not problem content
   - Impact: 1-4% of samples
   - Risk: Low (not needed for training)

4. **Markdown headers** ✅
   - Rationale: Formatting artifacts from original documents
   - Impact: 1-5% in OpenR1 only
   - Risk: Low (structural markup)

### What We Preserve

1. **Vague ground truths** ❌ (Do NOT clean)
   - Rationale: May be valid answers in context
   - User decision: Keep as-is, manual review later
   - Impact: 19% in ORZ-Math

2. **Non-English text** ❌ (Do NOT clean)
   - Rationale: Multilingual data has value
   - User decision: Keep as-is
   - Impact: 8% in ORZ-Math

3. **LaTeX escaping** ❌ (Do NOT modify)
   - Rationale: `\\frac` is correct JSON storage format
   - Investigation: Confirmed NOT a bug
   - Risk: High if modified (data corruption)

4. **Image references** ❌ (Do NOT remove)
   - Action: Detect and track in statistics only
   - Rationale: Removal makes problems unsolvable
   - Alternative: Flag for future filtering

5. **extra_info metadata** ❌ (Do NOT modify)
   - User requirement: No new fields added
   - All statistics kept in memory/reports only

### Conservative Approach Principles

1. **Preservation over removal**: When in doubt, keep data
2. **Reversibility**: All operations can be undone
3. **Statistics-driven**: Track everything, modify selectively
4. **User review**: Checkpoint at each dataset
5. **Schema compatibility**: Output = Input format exactly

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Regex over-matching removes valid content | Low | High | Extensive testing with real samples; conservative patterns |
| Schema incompatibility breaks downstream | Low | High | Strict validation; automated schema checks |
| LaTeX corruption | Very Low | Critical | No LaTeX modifications at all |
| Loss of useful metadata | Medium | Low | Keep contest info removal optional; track all changes |
| Ground truth modification | Very Low | Critical | Explicitly skip ground_truth field |

---

## Success Criteria

### Per-Dataset Targets

| Dataset | Current | Target | Key Metrics |
|---------|---------|--------|-------------|
| ORZ-Math | 63% clean | 85% clean | Problem numbers removed, contest metadata cleaned |
| OpenR1-Math | 89% clean | 97% clean | Highest improvement, all major issues addressed |
| Skywork-OR1 | 87% clean | 96% clean | Minimal changes, prefix removal only |
| DAPO-Math | 99% clean | 99.5% clean | Already excellent, verify quality maintained |

### Quality Indicators

- ✅ No LaTeX corruption
- ✅ Schema preserved exactly
- ✅ Ground truth unchanged
- ✅ No extra_info modifications
- ✅ Parquet format maintained
- ✅ Row counts match (no data loss)
- ✅ Before/after samples show clear improvement
- ✅ User approval at each checkpoint

---

## Timeline

- **Phase 1 (Implementation)**: 1 day
  - MathDatasetCleaner: 4-6 hours
  - Tests: 2-3 hours
  - Cleaning script: 1-2 hours

- **Phase 2 (Cleaning)**: 2-4 days
  - ORZ-Math: 0.5 day + review
  - OpenR1-Math: 0.5 day + review
  - Skywork-OR1: 0.5 day + review
  - DAPO-Math: 0.5 day + review

- **Total**: 3-5 days with user reviews

---

## Appendix: Dataset Analysis Details

### ORZ-Math Detailed Issues

**Problem Numbers (10-15%)**:
- "8.3 In the tetrahedron..."
- "7.1 Let $N$ be a regular nonagon..."
- "G1.4 When 491 is divided by..."
- "I2.1", "I2.2" (multi-part splits)

**Contest Metadata (5-8%)**:
- "24th Eötvös 1917 Problem 2 A square..."
- "20th APMC 1997 Problem 3 The 97 numbers..."
- "Example 31 (2004 College Entrance Examination...)"
- "(1968 Bulgarian Competition Problem)"

**Vague Ground Truths (19% - KEEPING)**:
- Single letters: "A", "B", "C", "D" (MCQ choices)
- Single numbers: "1", "8" (ambiguous)
- Variables: "N", "R-2r" (unclear)

**Non-English (8% - KEEPING)**:
- Chinese: "保留了源文本的换行和格式。"
- Mixed text: "____枚 chess pieces"

### OpenR1-Math Detailed Issues

**Problem Numbers (11%)**:
- "Problem 6. (8 points) In the plane..."
- Most affected dataset for this issue

**Markdown Headers (5%)**:
- "## Problem Statement\n\nCalculate..."
- "## Zadatak B-1.2."

**Contest Metadata (5%)**:
- Similar patterns to ORZ-Math

### Skywork-OR1 Detailed Issues

**Relatively Clean** (87% clean):
- "Question 230, Let $S$ be..." (2%)
- Figure references (3%)
- Very few other issues

### DAPO-Math Detailed Issues

**Almost Perfect** (99% clean):
- Minimal issues found
- Best quality reference

---

## Document Version History

- **v1.0** (2025-11-07): Initial plan created
- **v1.1** (2025-11-07): Implementation completed
  - MathDatasetCleaner formatter class implemented (380+ lines)
  - 32 comprehensive tests written and passing
  - Sequential cleaning script created
  - Ready for dataset cleaning phase

---

## Contact & Support

For questions or issues during implementation:
- Review this plan first
- Check test cases for examples
- Refer to analysis details in Appendix
- User review required at each checkpoint

---

*This plan is a living document and will be updated as implementation progresses.*
