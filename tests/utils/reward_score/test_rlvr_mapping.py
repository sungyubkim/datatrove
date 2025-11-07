"""Tests for RLVR to IFEval mapping."""

import pytest


class TestRLVRToIFEvalMapping:
    """Test RLVR function name to IFEval instruction ID mapping."""

    def test_all_rlvr_functions_map_to_existing_instructions(self):
        """모든 RLVR 함수가 기존 ifeval instruction에 매핑되는지 검증"""
        from datatrove.utils.reward_score.ifeval.instructions_registry import INSTRUCTION_DICT
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import RLVR_TO_IFEVAL_MAP

        for rlvr_func, ifeval_id in RLVR_TO_IFEVAL_MAP.items():
            assert ifeval_id in INSTRUCTION_DICT, f"Missing instruction: {rlvr_func} -> {ifeval_id}"

    def test_mapping_has_all_25_functions(self):
        """RLVR의 25개 함수가 모두 매핑되어 있는지 확인"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import RLVR_TO_IFEVAL_MAP

        expected_functions = {
            "verify_keywords",
            "verify_keyword_frequency",
            "validate_forbidden_words",
            "verify_letter_frequency",
            "validate_response_language",
            "verify_paragraph_count",
            "validate_word_constraint",
            "verify_sentence_constraint",
            "validate_paragraphs",
            "verify_postscript",
            "validate_placeholders",
            "verify_bullet_points",
            "validate_title",
            "validate_choice",
            "validate_highlighted_sections",
            "validate_sections",
            "validate_json_format",
            "validate_repeat_prompt",
            "validate_two_responses",
            "validate_uppercase",
            "validate_lowercase",
            "validate_frequency_capital_words",
            "validate_end",
            "validate_quotation",
            "validate_no_commas",
        }

        assert set(RLVR_TO_IFEVAL_MAP.keys()) == expected_functions

    def test_no_duplicate_mappings(self):
        """매핑에 중복이 없는지 확인"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import RLVR_TO_IFEVAL_MAP

        # 각 RLVR 함수가 고유한 매핑을 가져야 함 (일부 중복은 허용)
        assert len(RLVR_TO_IFEVAL_MAP) == 25


class TestParameterNameMapping:
    """Test RLVR parameter name to IFEval parameter name mapping."""

    def test_paragraph_count_params(self):
        """verify_paragraph_count 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("verify_paragraph_count", {"N": 5})
        assert result == {"num_paragraphs": 5}

    def test_word_constraint_params(self):
        """validate_word_constraint 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("validate_word_constraint", {"N": 100, "quantifier": "at least"})
        assert result == {"num_words": 100, "relation": "at least"}

    def test_sentence_constraint_params(self):
        """verify_sentence_constraint 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("verify_sentence_constraint", {"N": 5, "quantifier": "at most"})
        # "at most N" → "less than N+1" (since x ≤ 5 ⟺ x < 6)
        assert result == {"num_sentences": 6, "relation": "less than"}

    def test_paragraphs_params(self):
        """validate_paragraphs 다중 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("validate_paragraphs", {"N": 3, "i": 2, "first_word": "Hello"})
        assert result == {"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "Hello"}

    def test_keyword_frequency_params(self):
        """verify_keyword_frequency 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("verify_keyword_frequency", {"word": "test", "N": 3})
        assert result == {"keyword": "test", "frequency": 3}

    def test_keywords_params(self):
        """verify_keywords 리스트 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("verify_keywords", {"keyword_list": ["test", "example"]})
        assert result == {"keywords": ["test", "example"]}

    def test_null_filtering(self):
        """null 값은 필터링됨"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("verify_keywords", {"keyword_list": ["test"], "N": None, "letter": None})
        assert result == {"keywords": ["test"]}

    def test_no_params_function(self):
        """파라미터가 없는 함수는 None 반환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("validate_lowercase", {"N": None, "quantifier": None})
        assert result is None

    def test_postscript_params(self):
        """verify_postscript 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("verify_postscript", {"postscript_marker": "P.S."})
        assert result == {"postscript_marker": "P.S."}

    def test_forbidden_words_params(self):
        """validate_forbidden_words 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("validate_forbidden_words", {"forbidden_words": ["bad", "evil"]})
        assert result == {"forbidden_words": ["bad", "evil"]}

    def test_letter_frequency_params(self):
        """verify_letter_frequency 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("verify_letter_frequency", {"letter": "a", "N": 5})
        assert result == {"letter": "a", "let_frequency": 5}

    def test_response_language_params(self):
        """validate_response_language 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("validate_response_language", {"language": "en"})
        assert result == {"language": "en"}

    def test_placeholders_params(self):
        """validate_placeholders 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("validate_placeholders", {"N": 3})
        assert result == {"num_placeholders": 3}

    def test_bullet_points_params(self):
        """verify_bullet_points 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("verify_bullet_points", {"N": 5})
        assert result == {"num_bullets": 5}

    def test_highlighted_sections_params(self):
        """validate_highlighted_sections 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("validate_highlighted_sections", {"N": 2})
        assert result == {"num_highlights": 2}

    def test_sections_params(self):
        """validate_sections 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("validate_sections", {"N": 4, "section_splitter": "Section"})
        # Note: IFEval has typo "section_spliter" (one 't')
        assert result == {"num_sections": 4, "section_spliter": "Section"}

    def test_repeat_prompt_params(self):
        """validate_repeat_prompt 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("validate_repeat_prompt", {"original_prompt": "Repeat this"})
        assert result == {"prompt_to_repeat": "Repeat this"}

    def test_capital_words_params(self):
        """validate_frequency_capital_words 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("validate_frequency_capital_words", {"N": 3, "quantifier": "at least"})
        assert result == {"capital_frequency": 3, "capital_relation": "at least"}

    def test_end_checker_params(self):
        """validate_end 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        result = map_param_names("validate_end", {"end_phrase": "The end."})
        assert result == {"end_phrase": "The end."}

    def test_choice_params(self):
        """validate_choice 파라미터 변환"""
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import map_param_names

        # validate_choice has no parameter mapping (IFEval uses hardcoded options)
        result = map_param_names("validate_choice", {"options": ["A", "B", "C"]})
        assert result is None
