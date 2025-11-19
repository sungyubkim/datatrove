"""Tests for RLVR ground truth parser."""

import json

import pytest


class TestGroundTruthParser:
    """Test parsing RLVR ground truth to IFEval format."""

    def test_parse_simple_no_params(self):
        """Test parsing constraint without parameters"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({
            "func_name": "validate_lowercase",
            "N": None,
            "quantifier": None,
            "end_phrase": None,
            "keyword_list": None,
        })

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [{"instruction_id": ["change_case:english_lowercase"], "kwargs": [None]}]
        assert result == expected

    def test_parse_paragraph_count(self):
        """Test parsing constraint with single parameter"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "verify_paragraph_count", "N": 5})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [{"instruction_id": ["length_constraints:number_paragraphs"], "kwargs": [{"num_paragraphs": 5}]}]
        assert result == expected

    def test_parse_word_constraint(self):
        """Test parsing constraint with multiple parameters"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "validate_word_constraint", "N": 100, "quantifier": "at least"})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [
            {"instruction_id": ["length_constraints:number_words"], "kwargs": [{"num_words": 100, "relation": "at least"}]}
        ]
        assert result == expected

    def test_parse_sentence_constraint(self):
        """Test parsing sentence constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "verify_sentence_constraint", "N": 5, "quantifier": "at most"})

        result = parse_rlvr_ground_truth(rlvr_gt)

        # "at most N" → "less than N+1"
        expected = [
            {
                "instruction_id": ["length_constraints:number_sentences"],
                "kwargs": [{"num_sentences": 6, "relation": "less than"}],
            }
        ]
        assert result == expected

    def test_parse_paragraphs_multi_params(self):
        """Test parsing constraint with multiple diverse parameters"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "validate_paragraphs", "N": 3, "i": 2, "first_word": "Hello"})

        result = parse_rlvr_ground_truth(rlvr_gt)

        assert result[0]["instruction_id"] == ["length_constraints:nth_paragraph_first_word"]
        assert result[0]["kwargs"][0]["num_paragraphs"] == 3
        assert result[0]["kwargs"][0]["nth_paragraph"] == 2
        assert result[0]["kwargs"][0]["first_word"] == "Hello"

    def test_parse_keyword_frequency(self):
        """Test parsing keyword frequency constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "verify_keyword_frequency", "word": "test", "N": 3})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [{"instruction_id": ["keywords:frequency"], "kwargs": [{"keyword": "test", "frequency": 3}]}]
        assert result == expected

    def test_parse_keywords_list(self):
        """Test parsing keywords list constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "verify_keywords", "keyword_list": ["test", "example"]})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["test", "example"]}]}]
        assert result == expected

    def test_parse_filters_null_values(self):
        """Test that null values are filtered from kwargs"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({
            "func_name": "verify_keywords",
            "keyword_list": ["test"],
            "N": None,
            "letter": None,
            "quantifier": None,
        })

        result = parse_rlvr_ground_truth(rlvr_gt)

        assert result[0]["kwargs"][0] == {"keywords": ["test"]}

    def test_parse_postscript(self):
        """Test parsing postscript constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "verify_postscript", "postscript_marker": "P.S."})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [{"instruction_id": ["detectable_content:postscript"], "kwargs": [{"postscript_marker": "P.S."}]}]
        assert result == expected

    def test_parse_forbidden_words(self):
        """Test parsing forbidden words constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "validate_forbidden_words", "forbidden_words": ["bad", "evil"]})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [{"instruction_id": ["keywords:forbidden_words"], "kwargs": [{"forbidden_words": ["bad", "evil"]}]}]
        assert result == expected

    def test_parse_letter_frequency(self):
        """Test parsing letter frequency constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "verify_letter_frequency", "letter": "a", "N": 5})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [{"instruction_id": ["keywords:letter_frequency"], "kwargs": [{"letter": "a", "let_frequency": 5}]}]
        assert result == expected

    def test_parse_response_language(self):
        """Test parsing response language constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "validate_response_language", "language": "en"})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [{"instruction_id": ["language:response_language"], "kwargs": [{"language": "en"}]}]
        assert result == expected

    def test_parse_invalid_json(self):
        """Test error handling for malformed JSON"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        with pytest.raises(json.JSONDecodeError):
            parse_rlvr_ground_truth("{invalid json")

    def test_parse_unknown_function(self):
        """Test error handling for unknown function name"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "unknown_function_xyz"})

        with pytest.raises(KeyError, match="Unknown RLVR function"):
            parse_rlvr_ground_truth(rlvr_gt)

    def test_parse_placeholders(self):
        """Test parsing placeholders constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "validate_placeholders", "N": 3})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [{"instruction_id": ["detectable_content:number_placeholders"], "kwargs": [{"num_placeholders": 3}]}]
        assert result == expected

    def test_parse_bullet_points(self):
        """Test parsing bullet points constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "verify_bullet_points", "N": 5})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [{"instruction_id": ["detectable_format:number_bullet_lists"], "kwargs": [{"num_bullets": 5}]}]
        assert result == expected

    def test_parse_highlighted_sections(self):
        """Test parsing highlighted sections constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "validate_highlighted_sections", "N": 2})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [
            {"instruction_id": ["detectable_format:number_highlighted_sections"], "kwargs": [{"num_highlights": 2}]}
        ]
        assert result == expected

    def test_parse_sections(self):
        """Test parsing sections constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "validate_sections", "N": 4, "section_splitter": "Section"})

        result = parse_rlvr_ground_truth(rlvr_gt)

        # Note: IFEval has typo "section_spliter" (one 't')
        expected = [
            {
                "instruction_id": ["detectable_format:multiple_sections"],
                "kwargs": [{"num_sections": 4, "section_spliter": "Section"}],
            }
        ]
        assert result == expected

    def test_parse_repeat_prompt(self):
        """Test parsing repeat prompt constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "validate_repeat_prompt", "original_prompt": "Repeat this"})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [{"instruction_id": ["combination:repeat_prompt"], "kwargs": [{"prompt_to_repeat": "Repeat this"}]}]
        assert result == expected

    def test_parse_capital_words(self):
        """Test parsing capital words frequency constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "validate_frequency_capital_words", "N": 3, "quantifier": "at least"})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [
            {
                "instruction_id": ["change_case:capital_word_frequency"],
                "kwargs": [{"capital_frequency": 3, "capital_relation": "at least"}],
            }
        ]
        assert result == expected

    def test_parse_end_checker(self):
        """Test parsing end checker constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "validate_end", "end_phrase": "The end."})

        result = parse_rlvr_ground_truth(rlvr_gt)

        expected = [{"instruction_id": ["startend:end_checker"], "kwargs": [{"end_phrase": "The end."}]}]
        assert result == expected

    def test_parse_choice(self):
        """Test parsing choice constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "validate_choice", "options": ["A", "B", "C"]})

        result = parse_rlvr_ground_truth(rlvr_gt)

        # validate_choice has no parameter mapping (IFEval uses hardcoded options)
        expected = [{"instruction_id": ["detectable_format:constrained_response"], "kwargs": [None]}]
        assert result == expected

    def test_result_is_list_of_single_dict(self):
        """Test result format is always list with single constraint dict"""
        from datatrove.preprocessing.rlvr_to_ifbench import parse_rlvr_ground_truth

        rlvr_gt = json.dumps({"func_name": "validate_lowercase"})

        result = parse_rlvr_ground_truth(rlvr_gt)

        # Must be a list
        assert isinstance(result, list)
        # Must have exactly one element (RLVR has single constraint per example)
        assert len(result) == 1
        # Element must be a dict with instruction_id and kwargs
        assert "instruction_id" in result[0]
        assert "kwargs" in result[0]
        # instruction_id must be a list
        assert isinstance(result[0]["instruction_id"], list)
        # kwargs must be a list
        assert isinstance(result[0]["kwargs"], list)
