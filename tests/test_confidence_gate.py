"""Unit tests for ConfidenceGateNode."""

import pytest

from policyflow.nodes.confidence_gate import ConfidenceGateNode
from policyflow.models import ConfidenceLevel
from policyflow.nodes.criterion import CriterionResult
from policyflow.config import WorkflowConfig, ConfidenceGateConfig


@pytest.fixture
def mock_config():
    """Return a WorkflowConfig with test defaults."""
    return WorkflowConfig(
        temperature=0.0,
        max_retries=1,
        retry_wait=0,
        confidence_gate=ConfidenceGateConfig(
            high_threshold=0.8,
            low_threshold=0.5,
        ),
    )


@pytest.fixture
def high_confidence_results():
    """Return criterion results with all high confidence scores."""
    return {
        "criterion_1": CriterionResult(
            criterion_id="criterion_1",
            criterion_name="High Confidence 1",
            met=True,
            reasoning="Very confident",
            confidence=0.95,
        ),
        "criterion_2": CriterionResult(
            criterion_id="criterion_2",
            criterion_name="High Confidence 2",
            met=True,
            reasoning="Very confident",
            confidence=0.90,
        ),
    }


@pytest.fixture
def medium_confidence_results():
    """Return criterion results with some medium confidence scores."""
    return {
        "criterion_1": CriterionResult(
            criterion_id="criterion_1",
            criterion_name="High Confidence",
            met=True,
            reasoning="Confident",
            confidence=0.85,
        ),
        "criterion_2": CriterionResult(
            criterion_id="criterion_2",
            criterion_name="Medium Confidence",
            met=True,
            reasoning="Somewhat confident",
            confidence=0.65,
        ),
    }


@pytest.fixture
def low_confidence_results():
    """Return criterion results with some low confidence scores."""
    return {
        "criterion_1": CriterionResult(
            criterion_id="criterion_1",
            criterion_name="Low Confidence",
            met=True,
            reasoning="Not confident",
            confidence=0.3,
        ),
        "criterion_2": CriterionResult(
            criterion_id="criterion_2",
            criterion_name="Medium Confidence",
            met=True,
            reasoning="Somewhat confident",
            confidence=0.6,
        ),
    }


class TestConfidenceGateNodeHighConfidence:
    """Tests for high confidence routing."""

    def test_all_high_confidence(self, mock_config, high_confidence_results):
        """All criteria above high threshold should return 'high_confidence'."""
        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": high_confidence_results}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "high_confidence"
        assert exec_res["confidence_level"] == ConfidenceLevel.HIGH
        assert exec_res["needs_review"] is False
        assert len(exec_res["low_confidence_criteria"]) == 0


class TestConfidenceGateNodeMediumConfidence:
    """Tests for medium confidence routing."""

    def test_some_medium_confidence(self, mock_config, medium_confidence_results):
        """Some criteria in medium range should return 'needs_review'."""
        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": medium_confidence_results}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "needs_review"
        assert exec_res["confidence_level"] == ConfidenceLevel.MEDIUM
        assert exec_res["needs_review"] is True
        assert "criterion_2" in exec_res["medium_confidence_criteria"]


class TestConfidenceGateNodeLowConfidence:
    """Tests for low confidence routing."""

    def test_low_confidence_criteria(self, mock_config, low_confidence_results):
        """Any criterion below low threshold should return 'low_confidence'."""
        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": low_confidence_results}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "low_confidence"
        assert exec_res["confidence_level"] == ConfidenceLevel.LOW
        assert exec_res["needs_review"] is True
        assert "criterion_1" in exec_res["low_confidence_criteria"]


class TestConfidenceGateNodeEmptyResults:
    """Tests for empty results handling."""

    def test_no_results_returns_low(self, mock_config):
        """Empty results dict should return 'low_confidence'."""
        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": {}}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "low_confidence"
        assert exec_res["confidence_level"] == ConfidenceLevel.LOW
        assert exec_res["needs_review"] is True
        assert "No criterion results" in exec_res["reason"]

    def test_missing_criterion_results_key(self, mock_config):
        """Missing criterion_results key should return 'low_confidence'."""
        node = ConfidenceGateNode(config=mock_config)
        shared = {}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "low_confidence"


class TestConfidenceGateNodeSharedStore:
    """Tests for shared store interactions."""

    def test_gate_result_stored(self, mock_config, high_confidence_results):
        """Result should be stored in shared['confidence_gate_result']."""
        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": high_confidence_results}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "confidence_gate_result" in shared
        assert shared["confidence_gate_result"]["confidence_level"] == ConfidenceLevel.HIGH

    def test_average_confidence_calculated(self, mock_config, high_confidence_results):
        """Average confidence should be calculated correctly."""
        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": high_confidence_results}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        # (0.95 + 0.90) / 2 = 0.925
        assert exec_res["average_confidence"] == 0.925


class TestConfidenceGateNodeThresholdBoundaries:
    """Tests for threshold boundary conditions."""

    def test_exactly_at_high_threshold(self, mock_config):
        """Confidence exactly at high threshold should be high confidence."""
        results = {
            "criterion_1": CriterionResult(
                criterion_id="criterion_1",
                criterion_name="Exactly High",
                met=True,
                reasoning="At threshold",
                confidence=0.8,  # Exactly at high threshold
            ),
        }

        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": results}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "high_confidence"

    def test_just_below_high_threshold(self, mock_config):
        """Confidence just below high threshold should be medium."""
        results = {
            "criterion_1": CriterionResult(
                criterion_id="criterion_1",
                criterion_name="Just Below High",
                met=True,
                reasoning="Below threshold",
                confidence=0.79,
            ),
        }

        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": results}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "needs_review"
        assert exec_res["confidence_level"] == ConfidenceLevel.MEDIUM

    def test_exactly_at_low_threshold(self, mock_config):
        """Confidence exactly at low threshold should be medium."""
        results = {
            "criterion_1": CriterionResult(
                criterion_id="criterion_1",
                criterion_name="Exactly Low",
                met=True,
                reasoning="At threshold",
                confidence=0.5,  # Exactly at low threshold
            ),
        }

        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": results}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "needs_review"
        assert exec_res["confidence_level"] == ConfidenceLevel.MEDIUM

    def test_just_below_low_threshold(self, mock_config):
        """Confidence just below low threshold should be low."""
        results = {
            "criterion_1": CriterionResult(
                criterion_id="criterion_1",
                criterion_name="Just Below Low",
                met=True,
                reasoning="Below threshold",
                confidence=0.49,
            ),
        }

        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": results}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "low_confidence"
        assert exec_res["confidence_level"] == ConfidenceLevel.LOW


class TestConfidenceGateNodeEdgeCases:
    """Edge case tests for ConfidenceGateNode."""

    def test_default_config(self):
        """Node should work with default config."""
        node = ConfidenceGateNode()
        assert node.config is not None
        assert node.gate_config is not None

    def test_single_criterion(self, mock_config):
        """Single criterion should work correctly."""
        results = {
            "criterion_1": CriterionResult(
                criterion_id="criterion_1",
                criterion_name="Only One",
                met=True,
                reasoning="Single criterion",
                confidence=0.9,
            ),
        }

        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": results}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "high_confidence"

    def test_mixed_confidences_low_wins(self, mock_config):
        """If any criterion is low confidence, overall should be low."""
        results = {
            "criterion_1": CriterionResult(
                criterion_id="criterion_1",
                criterion_name="High",
                met=True,
                reasoning="High confidence",
                confidence=0.95,
            ),
            "criterion_2": CriterionResult(
                criterion_id="criterion_2",
                criterion_name="Low",
                met=True,
                reasoning="Low confidence",
                confidence=0.2,
            ),
        }

        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": results}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        # Low confidence takes precedence
        assert action == "low_confidence"
        assert "criterion_2" in exec_res["low_confidence_criteria"]

    def test_reason_includes_criteria_ids(self, mock_config, low_confidence_results):
        """Reason should include IDs of problematic criteria."""
        node = ConfidenceGateNode(config=mock_config)
        shared = {"criterion_results": low_confidence_results}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert "criterion_1" in exec_res["reason"]
