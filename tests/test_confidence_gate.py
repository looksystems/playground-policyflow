"""Unit tests for ConfidenceGateNode."""

import pytest

from policyflow.nodes.confidence_gate import ConfidenceGateNode
from policyflow.models import ConfidenceLevel
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
    """Return results with all high confidence scores."""
    return {
        "clause_1_result": {
            "met": True,
            "reasoning": "Very confident",
            "confidence": 0.95,
        },
        "clause_2_result": {
            "met": True,
            "reasoning": "Very confident",
            "confidence": 0.90,
        },
    }


@pytest.fixture
def medium_confidence_results():
    """Return results with some medium confidence scores."""
    return {
        "clause_1_result": {
            "met": True,
            "reasoning": "Confident",
            "confidence": 0.85,
        },
        "clause_2_result": {
            "met": True,
            "reasoning": "Somewhat confident",
            "confidence": 0.65,
        },
    }


@pytest.fixture
def low_confidence_results():
    """Return results with some low confidence scores."""
    return {
        "clause_1_result": {
            "met": True,
            "reasoning": "Not confident",
            "confidence": 0.3,
        },
        "clause_2_result": {
            "met": True,
            "reasoning": "Somewhat confident",
            "confidence": 0.6,
        },
    }


class TestConfidenceGateNodeHighConfidence:
    """Tests for high confidence routing."""

    def test_all_high_confidence(self, mock_config, high_confidence_results):
        """All results above high threshold should return 'high_confidence'."""
        node = ConfidenceGateNode(config=mock_config)
        shared = high_confidence_results.copy()

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "high_confidence"
        assert exec_res["confidence_level"] == ConfidenceLevel.HIGH
        assert exec_res["needs_review"] is False
        assert len(exec_res["low_confidence_items"]) == 0


class TestConfidenceGateNodeMediumConfidence:
    """Tests for medium confidence routing."""

    def test_some_medium_confidence(self, mock_config, medium_confidence_results):
        """Some results in medium range should return 'needs_review'."""
        node = ConfidenceGateNode(config=mock_config)
        shared = medium_confidence_results.copy()

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "needs_review"
        assert exec_res["confidence_level"] == ConfidenceLevel.MEDIUM
        assert exec_res["needs_review"] is True
        assert "clause_2_result" in exec_res["medium_confidence_items"]


class TestConfidenceGateNodeLowConfidence:
    """Tests for low confidence routing."""

    def test_low_confidence_results(self, mock_config, low_confidence_results):
        """Any result below low threshold should return 'low_confidence'."""
        node = ConfidenceGateNode(config=mock_config)
        shared = low_confidence_results.copy()

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "low_confidence"
        assert exec_res["confidence_level"] == ConfidenceLevel.LOW
        assert exec_res["needs_review"] is True
        assert "clause_1_result" in exec_res["low_confidence_items"]


class TestConfidenceGateNodeEmptyResults:
    """Tests for empty results handling."""

    def test_no_results_returns_low(self, mock_config):
        """Empty shared dict should return 'low_confidence'."""
        node = ConfidenceGateNode(config=mock_config)
        shared = {}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "low_confidence"
        assert exec_res["confidence_level"] == ConfidenceLevel.LOW
        assert exec_res["needs_review"] is True


class TestConfidenceGateNodeSharedStore:
    """Tests for shared store interactions."""

    def test_gate_result_stored(self, mock_config, high_confidence_results):
        """Result should be stored in shared['confidence_gate_result']."""
        node = ConfidenceGateNode(config=mock_config)
        shared = high_confidence_results.copy()

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "confidence_gate_result" in shared
        assert shared["confidence_gate_result"]["confidence_level"] == ConfidenceLevel.HIGH

    def test_average_confidence_calculated(self, mock_config, high_confidence_results):
        """Average confidence should be calculated correctly."""
        node = ConfidenceGateNode(config=mock_config)
        shared = high_confidence_results.copy()

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        # (0.95 + 0.90) / 2 = 0.925
        assert exec_res["average_confidence"] == 0.925


class TestConfidenceGateNodeThresholdBoundaries:
    """Tests for threshold boundary conditions."""

    def test_exactly_at_high_threshold(self, mock_config):
        """Confidence exactly at high threshold should be high confidence."""
        shared = {
            "clause_1_result": {
                "met": True,
                "reasoning": "At threshold",
                "confidence": 0.8,  # Exactly at high threshold
            },
        }

        node = ConfidenceGateNode(config=mock_config)

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "high_confidence"

    def test_just_below_high_threshold(self, mock_config):
        """Confidence just below high threshold should be medium."""
        shared = {
            "clause_1_result": {
                "met": True,
                "reasoning": "Below threshold",
                "confidence": 0.79,
            },
        }

        node = ConfidenceGateNode(config=mock_config)

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "needs_review"
        assert exec_res["confidence_level"] == ConfidenceLevel.MEDIUM

    def test_exactly_at_low_threshold(self, mock_config):
        """Confidence exactly at low threshold should be medium."""
        shared = {
            "clause_1_result": {
                "met": True,
                "reasoning": "At threshold",
                "confidence": 0.5,  # Exactly at low threshold
            },
        }

        node = ConfidenceGateNode(config=mock_config)

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "needs_review"
        assert exec_res["confidence_level"] == ConfidenceLevel.MEDIUM

    def test_just_below_low_threshold(self, mock_config):
        """Confidence just below low threshold should be low."""
        shared = {
            "clause_1_result": {
                "met": True,
                "reasoning": "Below threshold",
                "confidence": 0.49,
            },
        }

        node = ConfidenceGateNode(config=mock_config)

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

    def test_single_result(self, mock_config):
        """Single result should work correctly."""
        shared = {
            "clause_1_result": {
                "met": True,
                "reasoning": "Single result",
                "confidence": 0.9,
            },
        }

        node = ConfidenceGateNode(config=mock_config)

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "high_confidence"

    def test_mixed_confidences_low_wins(self, mock_config):
        """If any result is low confidence, overall should be low."""
        shared = {
            "clause_1_result": {
                "met": True,
                "reasoning": "High confidence",
                "confidence": 0.95,
            },
            "clause_2_result": {
                "met": True,
                "reasoning": "Low confidence",
                "confidence": 0.2,
            },
        }

        node = ConfidenceGateNode(config=mock_config)

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        # Low confidence takes precedence
        assert action == "low_confidence"
        assert "clause_2_result" in exec_res["low_confidence_items"]

    def test_reason_includes_item_ids(self, mock_config, low_confidence_results):
        """Reason should include IDs of problematic items."""
        node = ConfidenceGateNode(config=mock_config)
        shared = low_confidence_results.copy()

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert "clause_1_result" in exec_res["reason"]
