"""Tests for RLVR integration with IFEval scorer."""

import json

import pytest


class TestRLVRScorerIntegration:
    """Test that transformed RLVR data works with existing IFEval scorer."""

    def test_scorer_accepts_transformed_data(self):
        """Test existing scorer works with transformed RLVR data"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench
        from datatrove.utils.reward_score import compute_score

        rlvr_sample = {
            "messages": [{"content": "Write in lowercase", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "validate_lowercase", "N": None}),
            "dataset": "ifeval",
        }

        transformed = transform_to_ifbench(rlvr_sample, index=0)

        # Should not raise errors
        score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl",
            solution_str="all lowercase text here",
            ground_truth=transformed["reward_model"]["ground_truth"],
        )

        assert isinstance(score, dict)
        assert "score" in score
        assert "reward_fmt" in score
        assert 0.0 <= score["score"] <= 1.0

    def test_scorer_validates_lowercase_correctly(self):
        """Test scorer correctly validates lowercase constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench
        from datatrove.utils.reward_score import compute_score

        rlvr_sample = {
            "messages": [{"content": "Write in lowercase", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "validate_lowercase"}),
            "dataset": "ifeval",
        }

        transformed = transform_to_ifbench(rlvr_sample, index=0)
        gt = transformed["reward_model"]["ground_truth"]

        # Valid response (all lowercase)
        valid_score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl", solution_str="all lowercase text", ground_truth=gt
        )
        assert valid_score["score"] == 1.0

        # Invalid response (has uppercase)
        invalid_score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl", solution_str="Some Uppercase Text", ground_truth=gt
        )
        assert invalid_score["score"] == 0.0

    def test_scorer_validates_paragraph_count(self):
        """Test scorer correctly validates paragraph count constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench
        from datatrove.utils.reward_score import compute_score

        rlvr_sample = {
            "messages": [{"content": "Write 3 paragraphs", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "verify_paragraph_count", "N": 3}),
            "dataset": "ifeval",
        }

        transformed = transform_to_ifbench(rlvr_sample, index=0)
        gt = transformed["reward_model"]["ground_truth"]

        # Valid response (3 paragraphs) - use *** without spaces
        valid_response = "First paragraph\n***\nSecond paragraph\n***\nThird paragraph"
        valid_score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl", solution_str=valid_response, ground_truth=gt
        )
        assert valid_score["score"] == 1.0

        # Invalid response (2 paragraphs)
        invalid_response = "First paragraph\n***\nSecond paragraph"
        invalid_score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl", solution_str=invalid_response, ground_truth=gt
        )
        assert invalid_score["score"] == 0.0

    def test_scorer_validates_word_constraint(self):
        """Test scorer correctly validates word count constraint"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench
        from datatrove.utils.reward_score import compute_score

        rlvr_sample = {
            "messages": [{"content": "Write at least 10 words", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "validate_word_constraint", "N": 10, "quantifier": "at least"}),
            "dataset": "ifeval",
        }

        transformed = transform_to_ifbench(rlvr_sample, index=0)
        gt = transformed["reward_model"]["ground_truth"]

        # Valid response (15 words)
        valid_response = "This is a valid response with enough words to meet the minimum requirement."
        valid_score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl", solution_str=valid_response, ground_truth=gt
        )
        assert valid_score["score"] == 1.0

        # Invalid response (5 words)
        invalid_response = "Only five words here"
        invalid_score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl", solution_str=invalid_response, ground_truth=gt
        )
        assert invalid_score["score"] == 0.0

    def test_scorer_validates_keywords(self):
        """Test scorer correctly validates keyword existence"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench
        from datatrove.utils.reward_score import compute_score

        rlvr_sample = {
            "messages": [{"content": "Include keywords test and example", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "verify_keywords", "keyword_list": ["test", "example"]}),
            "dataset": "ifeval",
        }

        transformed = transform_to_ifbench(rlvr_sample, index=0)
        gt = transformed["reward_model"]["ground_truth"]

        # Valid response (has both keywords)
        valid_response = "This is a test response with an example."
        valid_score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl", solution_str=valid_response, ground_truth=gt
        )
        assert valid_score["score"] == 1.0

        # Invalid response (missing 'example')
        invalid_response = "This is only a test response."
        invalid_score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl", solution_str=invalid_response, ground_truth=gt
        )
        assert invalid_score["score"] == 0.0

    def test_scorer_with_xml_format(self):
        """Test scorer handles XML thinking format"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench
        from datatrove.utils.reward_score import compute_score

        rlvr_sample = {
            "messages": [{"content": "Write in lowercase", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "validate_lowercase"}),
            "dataset": "ifeval",
        }

        transformed = transform_to_ifbench(rlvr_sample, index=0)
        gt = transformed["reward_model"]["ground_truth"]

        # Response with XML thinking (should be removed before validation)
        response_with_thinking = "<think>Let me write in lowercase</think>\nall lowercase text here"

        score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl", solution_str=response_with_thinking, ground_truth=gt, format_type="xml"
        )

        # Thinking should be removed, remaining text is valid lowercase
        assert score["score"] == 1.0

    def test_scorer_with_gpt_oss_format(self):
        """Test scorer handles GPT-OSS format"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench
        from datatrove.utils.reward_score import compute_score

        rlvr_sample = {
            "messages": [{"content": "Write in lowercase", "role": "user"}],
            "ground_truth": json.dumps({"func_name": "validate_lowercase"}),
            "dataset": "ifeval",
        }

        transformed = transform_to_ifbench(rlvr_sample, index=0)
        gt = transformed["reward_model"]["ground_truth"]

        # GPT-OSS format response
        response = (
            "<|start|>assistant<|channel|>analysis<|message|>thinking<|end|>\n"
            "<|start|>assistant<|channel|>final<|message|>all lowercase text<|return|>"
        )

        score = compute_score(
            data_source="sungyub/ifeval-rlvr-verl", solution_str=response, ground_truth=gt, format_type="gpt_oss"
        )

        assert score["score"] == 1.0

    def test_all_constraint_types_scoreable(self):
        """Test all 25 RLVR constraint types can be scored"""
        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench
        from datatrove.utils.reward_score import compute_score
        from datatrove.utils.reward_score.ifeval.rlvr_mapping import RLVR_TO_IFEVAL_MAP

        # Sample parameters for constraints that require them
        constraint_params = {
            "verify_paragraph_count": {"N": 3},
            "verify_bullet_points": {"N": 3},
            "verify_keywords": {"keyword_list": ["test", "keyword"]},
            "validate_forbidden_words": {"forbidden_words": ["bad", "evil"]},
            "validate_repeat_prompt": {"original_prompt": "prompt"},
            "validate_end": {"end_phrase": "The end"},
            "verify_postscript": {"postscript_marker": "P.S."},
            "validate_placeholders": {"N": 2},
            "validate_highlighted_sections": {"N": 1},
            "validate_sections": {"N": 3, "section_splitter": "Section"},
            "validate_word_constraint": {"N": 8, "quantifier": "at least"},
            "verify_sentence_constraint": {"N": 3, "quantifier": "at least"},
            "verify_keyword_frequency": {"word": "test", "N": 3},
            "verify_letter_frequency": {"letter": "a", "N": 5},
            "validate_response_language": {"language": "en"},
            "validate_paragraphs": {"N": 3, "i": 1, "first_word": "Second"},
            "validate_frequency_capital_words": {"N": 3, "quantifier": "at least"},
            # Note: validate_choice doesn't support custom options in IFEval (uses hardcoded options)
        }

        # Sample valid responses for each constraint type
        valid_responses = {
            "validate_lowercase": "all lowercase text",
            "validate_uppercase": "ALL UPPERCASE TEXT",
            "validate_no_commas": "No commas in this text",
            "validate_quotation": '"This is wrapped in quotes"',
            "validate_title": "<<title>> Some text",
            "validate_json_format": '{"key": "value"}',
            "validate_two_responses": "First response\n******\nSecond response",
            "verify_paragraph_count": "Para 1\n***\nPara 2\n***\nPara 3",
            "verify_bullet_points": "* Bullet 1\n* Bullet 2\n* Bullet 3",
            "verify_keywords": "This text contains test keyword",
            "validate_forbidden_words": "This text is clean",
            "validate_repeat_prompt": "prompt Actual response",
            "validate_end": "Some text. The end",
            "verify_postscript": "Main text\n\nP.S. Additional note",
            "validate_placeholders": "Hello [name], your [item] is ready",
            "validate_highlighted_sections": "This has *highlight* text",
            "validate_sections": "Section 1\nSection 2\nSection 3",
            "validate_word_constraint": "This response has enough words to satisfy the constraint here",
            "verify_sentence_constraint": "Sentence one. Sentence two. Sentence three.",
            "verify_keyword_frequency": "The word test appears here test and test again",
            "verify_letter_frequency": "aaaaa has five letter a",
            "validate_response_language": "This is English text",
            "validate_paragraphs": "First\n***\nSecond with specific word\n***\nThird",
            "validate_highlighted_sections": "Text with *highlight* here",
            "validate_frequency_capital_words": "Here are SOME ALL CAPS WORDS",
            "validate_choice": "Option A",
        }

        for func_name in RLVR_TO_IFEVAL_MAP.keys():
            # Build ground truth with required parameters
            gt_dict = {"func_name": func_name}
            if func_name in constraint_params:
                gt_dict.update(constraint_params[func_name])

            rlvr_sample = {
                "messages": [{"content": f"Test {func_name}", "role": "user"}],
                "ground_truth": json.dumps(gt_dict),
                "dataset": "ifeval",
            }

            transformed = transform_to_ifbench(rlvr_sample, index=0)

            # Use appropriate valid response
            valid_response = valid_responses.get(func_name, "default test response")

            # Should not raise errors
            score = compute_score(
                data_source="sungyub/ifeval-rlvr-verl",
                solution_str=valid_response,
                ground_truth=transformed["reward_model"]["ground_truth"],
            )

            assert "score" in score, f"Scoring failed for {func_name}"
            assert 0.0 <= score["score"] <= 1.0, f"Invalid score for {func_name}: {score['score']}"


class TestRLVRRealDatasetIntegration:
    """Test integration with real RLVR dataset samples."""

    @pytest.mark.parametrize("num_samples", [10])
    def test_score_real_rlvr_samples(self, num_samples):
        """Test scoring real RLVR dataset samples"""
        from datasets import load_dataset

        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench
        from datatrove.utils.reward_score import compute_score

        # Load real RLVR samples
        dataset = load_dataset("allenai/RLVR-IFeval", split=f"train[:{num_samples}]")

        for i, example in enumerate(dataset):
            # Transform to IFBench format
            transformed = transform_to_ifbench(example, index=i)

            # Test with a simple response (will likely fail validation, but shouldn't crash)
            score = compute_score(
                data_source="sungyub/ifeval-rlvr-verl",
                solution_str="This is a test response",
                ground_truth=transformed["reward_model"]["ground_truth"],
            )

            # Should return valid score structure
            assert isinstance(score, dict)
            assert "score" in score
            assert 0.0 <= score["score"] <= 1.0

    @pytest.mark.parametrize("func_name", ["validate_lowercase", "verify_paragraph_count", "validate_word_constraint"])
    def test_score_specific_constraints_from_dataset(self, func_name):
        """Test scoring specific constraint types from real dataset"""
        from datasets import load_dataset

        from datatrove.preprocessing.rlvr_to_ifbench import transform_to_ifbench
        from datatrove.utils.reward_score import compute_score

        # Load dataset and find examples with specific constraint
        dataset = load_dataset("allenai/RLVR-IFeval", split="train[:1000]")

        for example in dataset:
            gt_dict = json.loads(example["ground_truth"])
            if gt_dict["func_name"] == func_name:
                # Transform and score
                transformed = transform_to_ifbench(example, index=0)

                score = compute_score(
                    data_source="sungyub/ifeval-rlvr-verl",
                    solution_str="test response",
                    ground_truth=transformed["reward_model"]["ground_truth"],
                )

                assert "score" in score
                # Found and scored at least one example
                return

        # If we get here, constraint type wasn't found in first 1000 samples
        pytest.skip(f"Constraint type {func_name} not found in first 1000 samples")
