"""Unit tests for SamplerNode."""

from unittest.mock import patch, MagicMock

import pytest

from policyflow.nodes.sampler import SamplerNode
from policyflow.config import WorkflowConfig


@pytest.fixture
def mock_config():
    """Return a WorkflowConfig with test defaults."""
    return WorkflowConfig(
        model="test-model",
        temperature=0.0,
        max_retries=1,
        retry_wait=0,
    )


def create_mock_llm_response(content: str):
    """Create a mock LiteLLM completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestSamplerNodeMajorityAggregation:
    """Tests for majority aggregation mode."""

    @patch("policyflow.llm.completion")
    def test_majority_aggregation_true(self, mock_completion, mock_config):
        """>50% true results should aggregate to true."""
        # 3 out of 5 return true
        responses = [
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: false\nreasoning: No"),
            create_mock_llm_response("result: false\nreasoning: No"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=5,
            aggregation="majority",
            inner_prompt="Is this appropriate?",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test content"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["aggregated_result"] is True
        assert exec_res["true_count"] == 3
        assert exec_res["false_count"] == 2

    @patch("policyflow.llm.completion")
    def test_majority_aggregation_false(self, mock_completion, mock_config):
        """<=50% true results should aggregate to false."""
        # 2 out of 5 return true
        responses = [
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: false\nreasoning: No"),
            create_mock_llm_response("result: false\nreasoning: No"),
            create_mock_llm_response("result: false\nreasoning: No"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=5,
            aggregation="majority",
            inner_prompt="Is this appropriate?",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test content"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["aggregated_result"] is False


class TestSamplerNodeUnanimousAggregation:
    """Tests for unanimous aggregation mode."""

    @patch("policyflow.llm.completion")
    def test_unanimous_aggregation_all_true(self, mock_completion, mock_config):
        """All samples true should aggregate to true."""
        responses = [
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=3,
            aggregation="unanimous",
            inner_prompt="Is this safe?",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test content"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["aggregated_result"] is True
        assert exec_res["true_count"] == 3

    @patch("policyflow.llm.completion")
    def test_unanimous_aggregation_not_all(self, mock_completion, mock_config):
        """Any false sample should aggregate to false in unanimous mode."""
        responses = [
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: false\nreasoning: No"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=3,
            aggregation="unanimous",
            inner_prompt="Is this safe?",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test content"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["aggregated_result"] is False


class TestSamplerNodeAnyAggregation:
    """Tests for 'any' aggregation mode."""

    @patch("policyflow.llm.completion")
    def test_any_aggregation_one_true(self, mock_completion, mock_config):
        """At least one true should aggregate to true."""
        responses = [
            create_mock_llm_response("result: false\nreasoning: No"),
            create_mock_llm_response("result: false\nreasoning: No"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=3,
            aggregation="any",
            inner_prompt="Contains issue?",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test content"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["aggregated_result"] is True

    @patch("policyflow.llm.completion")
    def test_any_aggregation_all_false(self, mock_completion, mock_config):
        """All false should aggregate to false."""
        responses = [
            create_mock_llm_response("result: false\nreasoning: No"),
            create_mock_llm_response("result: false\nreasoning: No"),
            create_mock_llm_response("result: false\nreasoning: No"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=3,
            aggregation="any",
            inner_prompt="Contains issue?",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test content"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["aggregated_result"] is False


class TestSamplerNodeActions:
    """Tests for routing actions based on agreement."""

    @patch("policyflow.llm.completion")
    def test_consensus_action(self, mock_completion, mock_config):
        """100% agreement should return 'consensus' action."""
        responses = [
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=3,
            aggregation="majority",
            inner_prompt="Test",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "consensus"
        assert exec_res["agreement_ratio"] == 1.0

    @patch("policyflow.llm.completion")
    def test_majority_action(self, mock_completion, mock_config):
        """Clear majority (not unanimous) should return 'majority' action."""
        responses = [
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: false\nreasoning: No"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=3,
            aggregation="majority",
            inner_prompt="Test",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "majority"
        assert exec_res["agreement_ratio"] == 2 / 3

    @patch("policyflow.llm.completion")
    def test_split_action(self, mock_completion, mock_config):
        """50/50 split should return 'split' action."""
        responses = [
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: false\nreasoning: No"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=2,
            aggregation="majority",
            inner_prompt="Test",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "split"
        assert exec_res["agreement_ratio"] == 0.5


class TestSamplerNodeAgreementRatio:
    """Tests for agreement ratio calculation."""

    @patch("policyflow.llm.completion")
    def test_agreement_ratio_calculated(self, mock_completion, mock_config):
        """Agreement ratio should be correctly calculated."""
        # 4 true, 1 false = 80% agreement
        responses = [
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: false\nreasoning: No"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=5,
            aggregation="majority",
            inner_prompt="Test",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["agreement_ratio"] == 0.8


class TestSamplerNodeSharedStore:
    """Tests for shared store interactions."""

    @patch("policyflow.llm.completion")
    def test_individual_results_stored(self, mock_completion, mock_config):
        """Individual sample results should be stored."""
        responses = [
            create_mock_llm_response("result: true\nreasoning: Reason 1"),
            create_mock_llm_response("result: false\nreasoning: Reason 2"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=2,
            aggregation="majority",
            inner_prompt="Test",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "sample_results" in shared
        assert len(shared["sample_results"]["individual_results"]) == 2
        assert shared["sample_results"]["individual_results"][0]["result"] is True
        assert shared["sample_results"]["individual_results"][1]["result"] is False
        assert shared["sample_results"]["individual_results"][0]["reasoning"] == "Reason 1"

    @patch("policyflow.llm.completion")
    def test_sample_results_structure(self, mock_completion, mock_config):
        """Sample results should have complete structure."""
        responses = [
            create_mock_llm_response("result: true\nreasoning: Yes"),
            create_mock_llm_response("result: true\nreasoning: Yes"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=2,
            aggregation="majority",
            inner_prompt="Test",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        results = shared["sample_results"]
        assert "individual_results" in results
        assert "aggregated_result" in results
        assert "agreement_ratio" in results
        assert "true_count" in results
        assert "false_count" in results
        assert "n_samples" in results
        assert "aggregation_mode" in results


class TestSamplerNodeEdgeCases:
    """Edge case tests for SamplerNode."""

    @patch("policyflow.llm.completion")
    def test_single_sample(self, mock_completion, mock_config):
        """Single sample should work correctly."""
        responses = [
            create_mock_llm_response("result: true\nreasoning: Yes"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=1,
            aggregation="majority",
            inner_prompt="Test",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert exec_res["aggregated_result"] is True
        assert action == "consensus"

    @patch("policyflow.llm.completion")
    def test_custom_input_key(self, mock_completion, mock_config):
        """Custom input_key should read from specified key."""
        responses = [
            create_mock_llm_response("result: true\nreasoning: Yes"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=1,
            aggregation="majority",
            inner_prompt="Test",
            config=mock_config,
            input_key="custom_text",
            cache_ttl=0,
        )
        shared = {
            "custom_text": "Custom content",
            "input_text": "Should be ignored",
        }

        prep_res = node.prep(shared)

        assert prep_res["input_text"] == "Custom content"

    @patch("policyflow.llm.completion")
    def test_system_prompt_passed(self, mock_completion, mock_config):
        """System prompt should be stored for use."""
        responses = [
            create_mock_llm_response("result: true\nreasoning: Yes"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=1,
            aggregation="majority",
            inner_prompt="Test prompt",
            config=mock_config,
            system_prompt="You are a helpful evaluator",
            cache_ttl=0,
        )

        assert node.system_prompt == "You are a helpful evaluator"

    @patch("policyflow.llm.completion")
    def test_missing_result_defaults_false(self, mock_completion, mock_config):
        """Missing result field should default to false."""
        responses = [
            create_mock_llm_response("reasoning: No result field"),
        ]
        mock_completion.side_effect = responses

        node = SamplerNode(
            n_samples=1,
            aggregation="any",
            inner_prompt="Test",
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["individual_results"][0]["result"] is False
