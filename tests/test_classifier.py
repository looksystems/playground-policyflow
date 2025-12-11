"""Unit tests for ClassifierNode."""

from unittest.mock import patch, MagicMock

import pytest

from policyflow.nodes.classifier import ClassifierNode
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


class TestClassifierNodeBasic:
    """Tests for basic classification behavior."""

    @patch("policyflow.llm.completion")
    def test_valid_classification(self, mock_completion, mock_config):
        """LLM returns valid category should work correctly."""
        mock_completion.return_value = create_mock_llm_response(
            "category: spam\nconfidence: 0.95\nreasoning: Contains spam keywords"
        )

        node = ClassifierNode(
            categories=["spam", "legitimate", "unclear"],
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Buy now! Limited offer!"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "spam"
        assert exec_res["category"] == "spam"
        assert exec_res["confidence"] == 0.95
        assert "spam keywords" in exec_res["reasoning"]

    @patch("policyflow.llm.completion")
    def test_invalid_category_fallback(self, mock_completion, mock_config):
        """LLM returns invalid category should fallback to first category."""
        mock_completion.return_value = create_mock_llm_response(
            "category: unknown_category\nconfidence: 0.8\nreasoning: Some reasoning"
        )

        node = ClassifierNode(
            categories=["spam", "legitimate"],
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test message"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "spam"  # Fallback to first category
        assert exec_res["category"] == "spam"
        assert exec_res["confidence"] == 0.0  # Fallback has 0 confidence
        assert "invalid category" in exec_res["reasoning"].lower()

    @patch("policyflow.llm.completion")
    def test_confidence_score(self, mock_completion, mock_config):
        """Confidence score should be captured correctly."""
        mock_completion.return_value = create_mock_llm_response(
            "category: legitimate\nconfidence: 0.87\nreasoning: Looks legit"
        )

        node = ClassifierNode(
            categories=["spam", "legitimate"],
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Hello, how are you?"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["confidence"] == 0.87

    def test_empty_categories_raises(self, mock_config):
        """Empty categories list should raise ValueError."""
        with pytest.raises(ValueError, match="At least one category"):
            ClassifierNode(
                categories=[],
                config=mock_config,
            )


class TestClassifierNodeDescriptions:
    """Tests for category descriptions feature."""

    @patch("policyflow.llm.completion")
    def test_category_descriptions(self, mock_completion, mock_config):
        """Descriptions should be passed to prompt."""
        mock_completion.return_value = create_mock_llm_response(
            "category: complaint\nconfidence: 0.9\nreasoning: User is unhappy"
        )

        node = ClassifierNode(
            categories=["question", "complaint", "feedback"],
            config=mock_config,
            descriptions={
                "question": "User is asking for information",
                "complaint": "User is expressing dissatisfaction",
                "feedback": "User is providing suggestions",
            },
            cache_ttl=0,
        )
        shared = {"input_text": "I'm very unhappy with the service!"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "complaint"
        # Verify descriptions are in prep_res
        assert prep_res["descriptions"]["complaint"] == "User is expressing dissatisfaction"

    @patch("policyflow.llm.completion")
    def test_partial_descriptions(self, mock_completion, mock_config):
        """Partial descriptions (not all categories) should work."""
        mock_completion.return_value = create_mock_llm_response(
            "category: question\nconfidence: 0.85\nreasoning: Asking something"
        )

        node = ClassifierNode(
            categories=["question", "complaint", "feedback"],
            config=mock_config,
            descriptions={"question": "User is asking for info"},
            cache_ttl=0,
        )
        shared = {"input_text": "What is the return policy?"}

        prep_res = node.prep(shared)

        # Only question has a description
        assert "question" in prep_res["descriptions"]
        assert "complaint" not in prep_res["descriptions"]


class TestClassifierNodeSharedStore:
    """Tests for shared store interactions."""

    @patch("policyflow.llm.completion")
    def test_result_stored_in_shared(self, mock_completion, mock_config):
        """Classification result should be stored in shared['classification']."""
        mock_completion.return_value = create_mock_llm_response(
            "category: spam\nconfidence: 0.9\nreasoning: Spam detected"
        )

        node = ClassifierNode(
            categories=["spam", "legitimate"],
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Buy cheap products!"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "classification" in shared
        assert shared["classification"]["category"] == "spam"
        assert shared["classification"]["confidence"] == 0.9
        assert shared["classification"]["reasoning"] == "Spam detected"

    @patch("policyflow.llm.completion")
    def test_missing_input_text(self, mock_completion, mock_config):
        """Missing input_text should default to empty string."""
        mock_completion.return_value = create_mock_llm_response(
            "category: unclear\nconfidence: 0.5\nreasoning: No text to analyze"
        )

        node = ClassifierNode(
            categories=["spam", "legitimate", "unclear"],
            config=mock_config,
            cache_ttl=0,
        )
        shared = {}

        prep_res = node.prep(shared)

        assert prep_res["input_text"] == ""


class TestClassifierNodeEdgeCases:
    """Edge case tests for ClassifierNode."""

    @patch("policyflow.llm.completion")
    def test_missing_confidence_in_response(self, mock_completion, mock_config):
        """Missing confidence should default to 0.0."""
        mock_completion.return_value = create_mock_llm_response(
            "category: spam\nreasoning: Spam detected"
        )

        node = ClassifierNode(
            categories=["spam", "legitimate"],
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["confidence"] == 0.0

    @patch("policyflow.llm.completion")
    def test_missing_reasoning_in_response(self, mock_completion, mock_config):
        """Missing reasoning should default to empty string."""
        mock_completion.return_value = create_mock_llm_response(
            "category: spam\nconfidence: 0.9"
        )

        node = ClassifierNode(
            categories=["spam", "legitimate"],
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["reasoning"] == ""

    @patch("policyflow.llm.completion")
    def test_single_category(self, mock_completion, mock_config):
        """Single category should work."""
        mock_completion.return_value = create_mock_llm_response(
            "category: only_option\nconfidence: 1.0\nreasoning: Only choice"
        )

        node = ClassifierNode(
            categories=["only_option"],
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "only_option"

    @patch("policyflow.llm.completion")
    def test_action_is_category_name(self, mock_completion, mock_config):
        """Returned action should be the category name for routing."""
        mock_completion.return_value = create_mock_llm_response(
            "category: feedback\nconfidence: 0.8\nreasoning: Contains feedback"
        )

        node = ClassifierNode(
            categories=["question", "complaint", "feedback"],
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "I think you should improve X"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        # Action should match category exactly for workflow routing
        assert action == "feedback"
        assert action in ["question", "complaint", "feedback"]
