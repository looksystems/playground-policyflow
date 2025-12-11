"""Unit tests for CriterionEvaluationNode."""

from unittest.mock import patch, MagicMock

import pytest

from policyflow.nodes.criterion import CriterionEvaluationNode
from policyflow.models import Criterion, ParsedPolicy, LogicOperator
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


@pytest.fixture
def sample_criterion():
    """Return a sample Criterion for testing."""
    return Criterion(
        id="criterion_1",
        name="Test Criterion",
        description="This criterion checks for appropriate content",
    )


@pytest.fixture
def sample_parsed_policy(sample_criterion):
    """Return a sample ParsedPolicy for testing."""
    return ParsedPolicy(
        title="Test Policy",
        description="A test policy for unit testing",
        criteria=[sample_criterion],
        logic=LogicOperator.ALL,
        raw_text="# Test Policy",
    )


def create_mock_llm_response(content: str):
    """Create a mock LiteLLM completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestCriterionEvaluationNodeBasic:
    """Tests for basic criterion evaluation."""

    @patch("policyflow.llm.completion")
    def test_criterion_met(
        self, mock_completion, mock_config, sample_criterion, sample_parsed_policy
    ):
        """LLM says criterion is met should return met=True."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: The content is appropriate\nconfidence: 0.95"
        )

        node = CriterionEvaluationNode(
            criterion=sample_criterion,
            config=mock_config,
        )
        shared = {
            "input_text": "Hello, this is appropriate content",
            "parsed_policy": sample_parsed_policy,
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "default"
        result = shared["criterion_results"]["criterion_1"]
        assert result.met is True
        assert result.reasoning == "The content is appropriate"
        assert result.confidence == 0.95

    @patch("policyflow.llm.completion")
    def test_criterion_not_met(
        self, mock_completion, mock_config, sample_criterion, sample_parsed_policy
    ):
        """LLM says criterion is not met should return met=False."""
        mock_completion.return_value = create_mock_llm_response(
            "met: false\nreasoning: The content violates the policy\nconfidence: 0.88"
        )

        node = CriterionEvaluationNode(
            criterion=sample_criterion,
            config=mock_config,
        )
        shared = {
            "input_text": "This is inappropriate content",
            "parsed_policy": sample_parsed_policy,
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        result = shared["criterion_results"]["criterion_1"]
        assert result.met is False
        assert "violates" in result.reasoning


class TestCriterionEvaluationNodeConfidence:
    """Tests for confidence score handling."""

    @patch("policyflow.llm.completion")
    def test_confidence_stored(
        self, mock_completion, mock_config, sample_criterion, sample_parsed_policy
    ):
        """Confidence score should be captured correctly."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Good\nconfidence: 0.73"
        )

        node = CriterionEvaluationNode(
            criterion=sample_criterion,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            "parsed_policy": sample_parsed_policy,
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        result = shared["criterion_results"]["criterion_1"]
        assert result.confidence == 0.73

    @patch("policyflow.llm.completion")
    def test_missing_confidence_defaults_to_zero(
        self, mock_completion, mock_config, sample_criterion, sample_parsed_policy
    ):
        """Missing confidence should default to 0.0."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Good"
        )

        node = CriterionEvaluationNode(
            criterion=sample_criterion,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            "parsed_policy": sample_parsed_policy,
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        result = shared["criterion_results"]["criterion_1"]
        assert result.confidence == 0.0


class TestCriterionEvaluationNodeSharedStore:
    """Tests for shared store interactions."""

    @patch("policyflow.llm.completion")
    def test_result_stored_in_shared(
        self, mock_completion, mock_config, sample_criterion, sample_parsed_policy
    ):
        """Result should be stored in shared['criterion_results'][criterion_id]."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Good\nconfidence: 0.9"
        )

        node = CriterionEvaluationNode(
            criterion=sample_criterion,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            "parsed_policy": sample_parsed_policy,
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "criterion_results" in shared
        assert "criterion_1" in shared["criterion_results"]
        result = shared["criterion_results"]["criterion_1"]
        assert result.criterion_id == "criterion_1"
        assert result.criterion_name == "Test Criterion"

    @patch("policyflow.llm.completion")
    def test_multiple_criteria_results(
        self, mock_completion, mock_config, sample_parsed_policy
    ):
        """Multiple criteria should all be stored in shared store."""
        criterion_1 = Criterion(
            id="criterion_1",
            name="First",
            description="First criterion",
        )
        criterion_2 = Criterion(
            id="criterion_2",
            name="Second",
            description="Second criterion",
        )

        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Good\nconfidence: 0.9"
        )

        node1 = CriterionEvaluationNode(criterion=criterion_1, config=mock_config)
        node2 = CriterionEvaluationNode(criterion=criterion_2, config=mock_config)

        shared = {
            "input_text": "Test",
            "parsed_policy": sample_parsed_policy,
        }

        # Run first node
        prep_res = node1.prep(shared)
        exec_res = node1.exec(prep_res)
        node1.post(shared, prep_res, exec_res)

        # Run second node
        prep_res = node2.prep(shared)
        exec_res = node2.exec(prep_res)
        node2.post(shared, prep_res, exec_res)

        assert "criterion_1" in shared["criterion_results"]
        assert "criterion_2" in shared["criterion_results"]

    @patch("policyflow.llm.completion")
    def test_initializes_criterion_results_dict(
        self, mock_completion, mock_config, sample_criterion, sample_parsed_policy
    ):
        """Should initialize criterion_results dict if not present."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Good\nconfidence: 0.9"
        )

        node = CriterionEvaluationNode(
            criterion=sample_criterion,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            "parsed_policy": sample_parsed_policy,
            # No criterion_results key
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "criterion_results" in shared


class TestCriterionEvaluationNodeEdgeCases:
    """Edge case tests for CriterionEvaluationNode."""

    @patch("policyflow.llm.completion")
    def test_missing_met_defaults_to_false(
        self, mock_completion, mock_config, sample_criterion, sample_parsed_policy
    ):
        """Missing met field should default to False."""
        mock_completion.return_value = create_mock_llm_response(
            "reasoning: Some reasoning\nconfidence: 0.5"
        )

        node = CriterionEvaluationNode(
            criterion=sample_criterion,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            "parsed_policy": sample_parsed_policy,
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        result = shared["criterion_results"]["criterion_1"]
        assert result.met is False

    @patch("policyflow.llm.completion")
    def test_missing_reasoning_defaults_to_empty(
        self, mock_completion, mock_config, sample_criterion, sample_parsed_policy
    ):
        """Missing reasoning should default to empty string."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nconfidence: 0.9"
        )

        node = CriterionEvaluationNode(
            criterion=sample_criterion,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            "parsed_policy": sample_parsed_policy,
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        result = shared["criterion_results"]["criterion_1"]
        assert result.reasoning == ""

    @patch("policyflow.llm.completion")
    def test_prep_extracts_policy_context(
        self, mock_completion, mock_config, sample_criterion, sample_parsed_policy
    ):
        """prep should extract policy context from parsed_policy."""
        node = CriterionEvaluationNode(
            criterion=sample_criterion,
            config=mock_config,
        )
        shared = {
            "input_text": "Test input",
            "parsed_policy": sample_parsed_policy,
        }

        prep_res = node.prep(shared)

        assert prep_res["input_text"] == "Test input"
        assert prep_res["criterion"] == sample_criterion
        assert prep_res["policy_context"] == "A test policy for unit testing"

    def test_default_config(self, sample_criterion):
        """Node should work with default config."""
        node = CriterionEvaluationNode(criterion=sample_criterion)
        assert node.config is not None

    @patch("policyflow.llm.completion")
    def test_returns_default_action(
        self, mock_completion, mock_config, sample_criterion, sample_parsed_policy
    ):
        """CriterionEvaluationNode should always return 'default' action."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Good\nconfidence: 0.9"
        )

        node = CriterionEvaluationNode(
            criterion=sample_criterion,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            "parsed_policy": sample_parsed_policy,
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "default"
