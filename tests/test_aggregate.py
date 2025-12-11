"""Unit tests for ResultAggregatorNode."""

import pytest

from policyflow.nodes.aggregate import ResultAggregatorNode
from policyflow.models import (
    Criterion,
    ParsedPolicy,
    LogicOperator,
)
from policyflow.nodes.criterion import CriterionResult


@pytest.fixture
def sample_criterion():
    """Return a sample Criterion."""
    return Criterion(
        id="criterion_1",
        name="Test Criterion",
        description="A test criterion",
    )


@pytest.fixture
def all_met_results():
    """Return criterion results where all are met."""
    return {
        "criterion_1": CriterionResult(
            criterion_id="criterion_1",
            criterion_name="First Criterion",
            met=True,
            reasoning="First criterion met",
            confidence=0.9,
        ),
        "criterion_2": CriterionResult(
            criterion_id="criterion_2",
            criterion_name="Second Criterion",
            met=True,
            reasoning="Second criterion met",
            confidence=0.85,
        ),
    }


@pytest.fixture
def some_not_met_results():
    """Return criterion results where some are not met."""
    return {
        "criterion_1": CriterionResult(
            criterion_id="criterion_1",
            criterion_name="First Criterion",
            met=True,
            reasoning="First criterion met",
            confidence=0.9,
        ),
        "criterion_2": CriterionResult(
            criterion_id="criterion_2",
            criterion_name="Second Criterion",
            met=False,
            reasoning="Second criterion not met",
            confidence=0.8,
        ),
    }


@pytest.fixture
def none_met_results():
    """Return criterion results where none are met."""
    return {
        "criterion_1": CriterionResult(
            criterion_id="criterion_1",
            criterion_name="First Criterion",
            met=False,
            reasoning="First criterion not met",
            confidence=0.7,
        ),
        "criterion_2": CriterionResult(
            criterion_id="criterion_2",
            criterion_name="Second Criterion",
            met=False,
            reasoning="Second criterion not met",
            confidence=0.6,
        ),
    }


@pytest.fixture
def all_logic_policy(sample_criterion):
    """Return a policy with ALL logic."""
    return ParsedPolicy(
        title="ALL Logic Policy",
        description="Policy requiring all criteria met",
        criteria=[sample_criterion],
        logic=LogicOperator.ALL,
        raw_text="# Policy",
    )


@pytest.fixture
def any_logic_policy(sample_criterion):
    """Return a policy with ANY logic."""
    return ParsedPolicy(
        title="ANY Logic Policy",
        description="Policy requiring any criterion met",
        criteria=[sample_criterion],
        logic=LogicOperator.ANY,
        raw_text="# Policy",
    )


class TestResultAggregatorNodeAllLogic:
    """Tests for ALL (AND) logic aggregation."""

    def test_all_logic_all_met(self, all_logic_policy, all_met_results):
        """ALL logic with all criteria met should satisfy policy."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": all_logic_policy,
            "criterion_results": all_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "done"
        assert exec_res["policy_satisfied"] is True
        assert shared["evaluation_result"].policy_satisfied is True

    def test_all_logic_some_not_met(self, all_logic_policy, some_not_met_results):
        """ALL logic with some criteria not met should not satisfy policy."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": all_logic_policy,
            "criterion_results": some_not_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert exec_res["policy_satisfied"] is False
        assert shared["evaluation_result"].policy_satisfied is False


class TestResultAggregatorNodeAnyLogic:
    """Tests for ANY (OR) logic aggregation."""

    def test_any_logic_one_met(self, any_logic_policy, some_not_met_results):
        """ANY logic with one criterion met should satisfy policy."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": any_logic_policy,
            "criterion_results": some_not_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert exec_res["policy_satisfied"] is True
        assert shared["evaluation_result"].policy_satisfied is True

    def test_any_logic_none_met(self, any_logic_policy, none_met_results):
        """ANY logic with no criteria met should not satisfy policy."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": any_logic_policy,
            "criterion_results": none_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert exec_res["policy_satisfied"] is False
        assert shared["evaluation_result"].policy_satisfied is False

    def test_any_logic_all_met(self, any_logic_policy, all_met_results):
        """ANY logic with all criteria met should satisfy policy."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": any_logic_policy,
            "criterion_results": all_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert exec_res["policy_satisfied"] is True


class TestResultAggregatorNodeConfidence:
    """Tests for confidence calculation."""

    def test_overall_confidence_averaged(self, all_logic_policy, all_met_results):
        """Overall confidence should be average of criterion confidences."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": all_logic_policy,
            "criterion_results": all_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        # (0.9 + 0.85) / 2 = 0.875
        assert exec_res["overall_confidence"] == 0.875
        assert shared["evaluation_result"].overall_confidence == 0.875

    def test_confidence_single_criterion(self, all_logic_policy):
        """Single criterion should have its confidence as overall."""
        results = {
            "criterion_1": CriterionResult(
                criterion_id="criterion_1",
                criterion_name="Only One",
                met=True,
                reasoning="Met",
                confidence=0.77,
            ),
        }

        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": all_logic_policy,
            "criterion_results": results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["overall_confidence"] == 0.77


class TestResultAggregatorNodeReasoning:
    """Tests for reasoning summary generation."""

    def test_reasoning_summary_met_criteria(self, all_logic_policy, all_met_results):
        """Reasoning should list met criteria."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": all_logic_policy,
            "criterion_results": all_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert "Criteria met:" in exec_res["overall_reasoning"]
        assert "First Criterion" in exec_res["overall_reasoning"]
        assert "Second Criterion" in exec_res["overall_reasoning"]

    def test_reasoning_summary_unmet_criteria(self, all_logic_policy, some_not_met_results):
        """Reasoning should list unmet criteria."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": all_logic_policy,
            "criterion_results": some_not_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert "Criteria not met:" in exec_res["overall_reasoning"]
        assert "Second Criterion" in exec_res["overall_reasoning"]

    def test_reasoning_summary_mixed(self, all_logic_policy, some_not_met_results):
        """Reasoning should list both met and unmet criteria."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": all_logic_policy,
            "criterion_results": some_not_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert "Criteria met:" in exec_res["overall_reasoning"]
        assert "Criteria not met:" in exec_res["overall_reasoning"]


class TestResultAggregatorNodeSharedStore:
    """Tests for shared store interactions."""

    def test_evaluation_result_stored(self, all_logic_policy, all_met_results):
        """EvaluationResult should be stored in shared['evaluation_result']."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": all_logic_policy,
            "criterion_results": all_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "evaluation_result" in shared
        result = shared["evaluation_result"]
        assert result.policy_satisfied is True
        assert result.input_text == "Test input"
        assert result.policy_title == "ALL Logic Policy"

    def test_criterion_results_in_evaluation_result(
        self, all_logic_policy, all_met_results
    ):
        """Criterion results should be included in evaluation result."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": all_logic_policy,
            "criterion_results": all_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        result = shared["evaluation_result"]
        assert len(result.criterion_results) == 2


class TestResultAggregatorNodeEdgeCases:
    """Edge case tests for ResultAggregatorNode."""

    def test_empty_results(self, all_logic_policy):
        """Empty results should not satisfy policy."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": all_logic_policy,
            "criterion_results": {},
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert exec_res["policy_satisfied"] is False
        assert exec_res["overall_confidence"] == 0.0

    def test_empty_results_any_logic(self, any_logic_policy):
        """Empty results with ANY logic should not satisfy policy."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": any_logic_policy,
            "criterion_results": {},
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["policy_satisfied"] is False

    def test_empty_reasoning_message(self, all_logic_policy):
        """Empty results should have appropriate reasoning."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": all_logic_policy,
            "criterion_results": {},
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["overall_reasoning"] == "No criteria evaluated."

    def test_returns_done_action(self, all_logic_policy, all_met_results):
        """ResultAggregatorNode should always return 'done' action."""
        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": all_logic_policy,
            "criterion_results": all_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "done"

    def test_default_logic_is_all(self, sample_criterion, all_met_results):
        """Default logic (when not specified) should behave like ALL."""
        # Create a policy without explicit logic (defaults to ALL)
        policy = ParsedPolicy(
            title="Default Logic Policy",
            description="Policy with default logic",
            criteria=[sample_criterion],
            logic=LogicOperator.ALL,  # Default value
            raw_text="# Policy",
        )

        node = ResultAggregatorNode()
        shared = {
            "parsed_policy": policy,
            "criterion_results": all_met_results,
            "input_text": "Test input",
        }

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["policy_satisfied"] is True
