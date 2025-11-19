#!/usr/bin/env python3
"""
Test suite for table-r1-zero scorers (tqa, tfv, ff_tqa).

Tests:
1. tqa.py - Table QA with JSON list answers (WTQ, HiTab)
2. tfv.py - Table Fact Verification (TabFact)
3. ff_tqa.py - Free-form Table QA (FeTaQA)
"""

import pytest
from datatrove.utils.reward_score import default_compute_score


class TestTQAScorer:
    """Test tqa.py scorer for WTQ and HiTab."""

    def test_correct_answer_list(self):
        """Test correct answer with list format."""
        predict_str = '<think>The answer is [1, 2, 3]</think><answer>```json\n{"answer": ["1", "2", "3"]}```</answer>'
        ground_truth = ["1", "2", "3"]
        result = default_compute_score("WTQ", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0
        assert result["format_score"] == 1.0
        assert result["score"] == 2.0

    def test_correct_answer_unordered(self):
        """Test correct answer with different order."""
        predict_str = '{"answer": ["3", "1", "2"]}'
        ground_truth = ["1", "2", "3"]
        result = default_compute_score("HiTab", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0
        assert result["format_score"] == 0.0  # No tags
        assert result["score"] == 1.0

    def test_wrong_answer(self):
        """Test wrong answer."""
        predict_str = '{"answer": ["1", "2"]}'
        ground_truth = ["1", "2", "3"]
        result = default_compute_score("WTQ", predict_str, ground_truth)

        assert result["accurate_score"] == 0.0

    def test_numeric_normalization(self):
        """Test numeric answer normalization."""
        predict_str = '{"answer": ["1,234.56"]}'
        ground_truth = ["1234.56"]
        result = default_compute_score("HiTab", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0

    def test_single_value_answer(self):
        """Test single value answer (not in list)."""
        predict_str = '{"answer": 42}'
        ground_truth = ["42"]
        result = default_compute_score("WTQ", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0

    def test_no_answer_format(self):
        """Test missing answer format."""
        predict_str = "This is some random text"
        ground_truth = ["1", "2", "3"]
        result = default_compute_score("WTQ", predict_str, ground_truth)

        assert result["score"] == 0.0
        assert result["accurate_score"] == 0.0


class TestTFVScorer:
    """Test tfv.py scorer for TabFact."""

    def test_correct_entailed(self):
        """Test correct 'entailed' prediction."""
        predict_str = '<think>Analysis...</think><answer>```json\n{"answer": "entailed"}```</answer>'
        ground_truth = "entailed"
        result = default_compute_score("TabFact", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0
        assert result["format_score"] == 1.0
        assert result["score"] == 2.0

    def test_correct_refuted(self):
        """Test correct 'refuted' prediction."""
        predict_str = '{"answer": "refuted"}'
        ground_truth = "refuted"
        result = default_compute_score("TabFact", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0
        assert result["format_score"] == 0.0  # No tags
        assert result["score"] == 1.0

    def test_wrong_answer(self):
        """Test wrong classification."""
        predict_str = '{"answer": "entailed"}'
        ground_truth = "refuted"
        result = default_compute_score("TabFact", predict_str, ground_truth)

        assert result["accurate_score"] == 0.0

    def test_ground_truth_list(self):
        """Test ground truth in list format."""
        predict_str = '{"answer": "entailed"}'
        ground_truth = ["entailed"]
        result = default_compute_score("TabFact", predict_str, ground_truth)

        assert result["accurate_score"] == 1.0

    def test_invalid_value(self):
        """Test invalid answer value."""
        predict_str = '{"answer": "unknown"}'
        ground_truth = "entailed"
        result = default_compute_score("TabFact", predict_str, ground_truth)

        assert result["score"] == 0.0

    def test_no_answer_format(self):
        """Test missing answer format."""
        predict_str = "This is entailed"
        ground_truth = "entailed"
        result = default_compute_score("TabFact", predict_str, ground_truth)

        assert result["score"] == 0.0


class TestFFTQAScorer:
    """Test ff_tqa.py scorer for FeTaQA."""

    def test_exact_match(self):
        """Test exact match between prediction and ground truth."""
        predict_str = '<think>Analysis...</think><answer>```json\n{"answer": "The total revenue is 100 million"}```</answer>'
        ground_truth = "The total revenue is 100 million"
        result = default_compute_score("FeTaQA", predict_str, ground_truth)

        assert result["format_score"] == 1.0
        assert result["bleu_score"] == pytest.approx(1.0, abs=0.01)
        assert result["rouge_score"] == pytest.approx(1.0, abs=0.01)
        assert result["accurate_score"] == pytest.approx(1.0, abs=0.01)

    def test_partial_match(self):
        """Test partial match with some overlap."""
        predict_str = '{"answer": "The revenue is 100 million dollars"}'
        ground_truth = "The total revenue is 100 million"
        result = default_compute_score("FeTaQA", predict_str, ground_truth)

        assert result["format_score"] == 0.0  # No tags
        assert 0.5 < result["accurate_score"] < 1.0  # Some overlap
        assert result["bleu_score"] > 0.0
        assert result["rouge_score"] > 0.0

    def test_no_match(self):
        """Test completely different answer."""
        predict_str = '{"answer": "XYZ ABC"}'
        ground_truth = "The total revenue is 100 million"
        result = default_compute_score("FeTaQA", predict_str, ground_truth)

        assert result["accurate_score"] < 0.1  # Very low score
        assert result["bleu_score"] < 0.1
        assert result["rouge_score"] < 0.1

    def test_ground_truth_list(self):
        """Test ground truth in list format."""
        predict_str = '{"answer": "100 million"}'
        ground_truth = ["100 million"]
        result = default_compute_score("FeTaQA", predict_str, ground_truth)

        assert result["bleu_score"] == pytest.approx(1.0, abs=0.01)

    def test_list_answer(self):
        """Test when answer is in list format."""
        predict_str = '{"answer": ["item1", "item2", "item3"]}'
        ground_truth = "item1 item2 item3"
        result = default_compute_score("FeTaQA", predict_str, ground_truth)

        # Should normalize list to string
        assert result["bleu_score"] > 0.8
        assert result["rouge_score"] > 0.8

    def test_empty_answer(self):
        """Test empty answer."""
        predict_str = '{"answer": ""}'
        ground_truth = "The answer"
        result = default_compute_score("FeTaQA", predict_str, ground_truth)

        assert result["score"] == 0.0

    def test_fallback_to_raw_text(self):
        """Test fallback when no JSON pattern found."""
        predict_str = "The total revenue is 100 million"
        ground_truth = "The total revenue is 100 million"
        result = default_compute_score("FeTaQA", predict_str, ground_truth)

        # Should use raw text and get high score
        assert result["format_score"] == 0.0
        assert result["bleu_score"] == pytest.approx(1.0, abs=0.01)


class TestIntegrationWithDefaultComputeScore:
    """Test integration with default_compute_score."""

    def test_wtq_routing(self):
        """Test WTQ data source routing."""
        predict_str = '{"answer": ["1"]}'
        ground_truth = ["1"]
        result = default_compute_score("WTQ", predict_str, ground_truth)

        assert "score" in result
        assert "accurate_score" in result

    def test_hitab_routing(self):
        """Test HiTab data source routing."""
        predict_str = '{"answer": ["1"]}'
        ground_truth = ["1"]
        result = default_compute_score("HiTab", predict_str, ground_truth)

        assert "score" in result
        assert "accurate_score" in result

    def test_tabfact_routing(self):
        """Test TabFact data source routing."""
        predict_str = '{"answer": "entailed"}'
        ground_truth = "entailed"
        result = default_compute_score("TabFact", predict_str, ground_truth)

        assert "score" in result
        assert "accurate_score" in result

    def test_fetaqa_routing(self):
        """Test FeTaQA data source routing."""
        predict_str = '{"answer": "test answer"}'
        ground_truth = "test answer"
        result = default_compute_score("FeTaQA", predict_str, ground_truth)

        assert "score" in result
        assert "bleu_score" in result
        assert "rouge_score" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
