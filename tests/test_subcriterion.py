"""Unit tests for SubCriterionNode."""

from unittest.mock import patch, MagicMock

import pytest

from policyflow.nodes.subcriterion import SubCriterionNode
from policyflow.models import Criterion, LogicOperator
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
def parent_criterion():
    """Return a parent Criterion for testing."""
    return Criterion(
        id="parent_1",
        name="Parent Criterion",
        description="A parent criterion with sub-criteria",
    )


@pytest.fixture
def sub_criterion():
    """Return a sub-criterion for testing."""
    return Criterion(
        id="sub_1",
        name="Sub Criterion 1",
        description="First sub-criterion to evaluate",
    )


def create_mock_llm_response(content: str):
    """Create a mock LiteLLM completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestSubCriterionNodeAnyLogic:
    """Tests for ANY logic (OR) early termination."""

    @patch("policyflow.llm.completion")
    def test_any_logic_satisfied_early_exit(
        self, mock_completion, mock_config, parent_criterion, sub_criterion
    ):
        """ANY logic, one met should return 'satisfied' for early exit."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Sub-criterion is met\nconfidence: 0.9"
        )

        node = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_criterion,
            sub_logic=LogicOperator.ANY,
            config=mock_config,
        )
        shared = {
            "input_text": "Test content",
            "policy_context": "Test policy context",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "satisfied"
        result = shared["sub_criterion_results"]["parent_1"]["sub_1"]
        assert result.met is True

    @patch("policyflow.llm.completion")
    def test_any_logic_not_met_continues(
        self, mock_completion, mock_config, parent_criterion, sub_criterion
    ):
        """ANY logic, not met should return 'default' to continue."""
        mock_completion.return_value = create_mock_llm_response(
            "met: false\nreasoning: Sub-criterion not met\nconfidence: 0.85"
        )

        node = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_criterion,
            sub_logic=LogicOperator.ANY,
            config=mock_config,
        )
        shared = {
            "input_text": "Test content",
            "policy_context": "Test policy context",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "default"


class TestSubCriterionNodeAllLogic:
    """Tests for ALL logic (AND) early termination."""

    @patch("policyflow.llm.completion")
    def test_all_logic_failed_early_exit(
        self, mock_completion, mock_config, parent_criterion, sub_criterion
    ):
        """ALL logic, one not met should return 'failed' for early exit."""
        mock_completion.return_value = create_mock_llm_response(
            "met: false\nreasoning: Sub-criterion failed\nconfidence: 0.88"
        )

        node = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_criterion,
            sub_logic=LogicOperator.ALL,
            config=mock_config,
        )
        shared = {
            "input_text": "Test content",
            "policy_context": "Test policy context",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "failed"

    @patch("policyflow.llm.completion")
    def test_all_logic_met_continues(
        self, mock_completion, mock_config, parent_criterion, sub_criterion
    ):
        """ALL logic, met should return 'default' to continue."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Sub-criterion passed\nconfidence: 0.9"
        )

        node = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_criterion,
            sub_logic=LogicOperator.ALL,
            config=mock_config,
        )
        shared = {
            "input_text": "Test content",
            "policy_context": "Test policy context",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "default"


class TestSubCriterionNodeNestedStorage:
    """Tests for nested result storage structure."""

    @patch("policyflow.llm.completion")
    def test_nested_storage(
        self, mock_completion, mock_config, parent_criterion, sub_criterion
    ):
        """Results should be stored in nested dict structure."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Good\nconfidence: 0.9"
        )

        node = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_criterion,
            sub_logic=LogicOperator.ALL,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            "policy_context": "Context",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        # Check nested structure: shared["sub_criterion_results"][parent_id][sub_id]
        assert "sub_criterion_results" in shared
        assert "parent_1" in shared["sub_criterion_results"]
        assert "sub_1" in shared["sub_criterion_results"]["parent_1"]

        result = shared["sub_criterion_results"]["parent_1"]["sub_1"]
        assert result.sub_criterion_id == "sub_1"
        assert result.sub_criterion_name == "Sub Criterion 1"

    @patch("policyflow.llm.completion")
    def test_multiple_sub_criteria_same_parent(
        self, mock_completion, mock_config, parent_criterion
    ):
        """Multiple sub-criteria for same parent should be stored correctly."""
        sub_1 = Criterion(id="sub_1", name="Sub 1", description="First")
        sub_2 = Criterion(id="sub_2", name="Sub 2", description="Second")

        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Good\nconfidence: 0.9"
        )

        node1 = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_1,
            sub_logic=LogicOperator.ALL,
            config=mock_config,
        )
        node2 = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_2,
            sub_logic=LogicOperator.ALL,
            config=mock_config,
        )

        shared = {
            "input_text": "Test",
            "policy_context": "Context",
        }

        # Run first sub-criterion
        prep_res = node1.prep(shared)
        exec_res = node1.exec(prep_res)
        node1.post(shared, prep_res, exec_res)

        # Run second sub-criterion
        prep_res = node2.prep(shared)
        exec_res = node2.exec(prep_res)
        node2.post(shared, prep_res, exec_res)

        # Both should be stored under same parent
        assert "sub_1" in shared["sub_criterion_results"]["parent_1"]
        assert "sub_2" in shared["sub_criterion_results"]["parent_1"]

    @patch("policyflow.llm.completion")
    def test_initializes_nested_dicts(
        self, mock_completion, mock_config, parent_criterion, sub_criterion
    ):
        """Should initialize nested dict structure if not present."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Good\nconfidence: 0.9"
        )

        node = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_criterion,
            sub_logic=LogicOperator.ALL,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            "policy_context": "Context",
            # No sub_criterion_results key
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "sub_criterion_results" in shared
        assert "parent_1" in shared["sub_criterion_results"]


class TestSubCriterionNodeEdgeCases:
    """Edge case tests for SubCriterionNode."""

    @patch("policyflow.llm.completion")
    def test_missing_met_defaults_to_false(
        self, mock_completion, mock_config, parent_criterion, sub_criterion
    ):
        """Missing met field should default to False."""
        mock_completion.return_value = create_mock_llm_response(
            "reasoning: No met field\nconfidence: 0.5"
        )

        node = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_criterion,
            sub_logic=LogicOperator.ALL,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            "policy_context": "Context",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        result = shared["sub_criterion_results"]["parent_1"]["sub_1"]
        assert result.met is False

    @patch("policyflow.llm.completion")
    def test_missing_confidence_defaults_to_zero(
        self, mock_completion, mock_config, parent_criterion, sub_criterion
    ):
        """Missing confidence should default to 0.0."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Good"
        )

        node = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_criterion,
            sub_logic=LogicOperator.ALL,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            "policy_context": "Context",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        result = shared["sub_criterion_results"]["parent_1"]["sub_1"]
        assert result.confidence == 0.0

    @patch("policyflow.llm.completion")
    def test_missing_reasoning_defaults_to_empty(
        self, mock_completion, mock_config, parent_criterion, sub_criterion
    ):
        """Missing reasoning should default to empty string."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nconfidence: 0.9"
        )

        node = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_criterion,
            sub_logic=LogicOperator.ALL,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            "policy_context": "Context",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        result = shared["sub_criterion_results"]["parent_1"]["sub_1"]
        assert result.reasoning == ""

    @patch("policyflow.llm.completion")
    def test_prep_extracts_policy_context(
        self, mock_completion, mock_config, parent_criterion, sub_criterion
    ):
        """prep should extract policy context."""
        node = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_criterion,
            sub_logic=LogicOperator.ALL,
            config=mock_config,
        )
        shared = {
            "input_text": "Test input",
            "policy_context": "Test policy context",
        }

        prep_res = node.prep(shared)

        assert prep_res["input_text"] == "Test input"
        assert prep_res["parent_criterion"] == parent_criterion
        assert prep_res["sub_criterion"] == sub_criterion
        assert prep_res["policy_context"] == "Test policy context"

    @patch("policyflow.llm.completion")
    def test_missing_policy_context(
        self, mock_completion, mock_config, parent_criterion, sub_criterion
    ):
        """Missing policy_context should default to empty string."""
        mock_completion.return_value = create_mock_llm_response(
            "met: true\nreasoning: Good\nconfidence: 0.9"
        )

        node = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_criterion,
            sub_logic=LogicOperator.ALL,
            config=mock_config,
        )
        shared = {
            "input_text": "Test",
            # No policy_context
        }

        prep_res = node.prep(shared)

        assert prep_res["policy_context"] == ""

    def test_default_config(self, parent_criterion, sub_criterion):
        """Node should work with default config."""
        node = SubCriterionNode(
            parent_criterion=parent_criterion,
            sub_criterion=sub_criterion,
            sub_logic=LogicOperator.ALL,
        )
        assert node.config is not None
