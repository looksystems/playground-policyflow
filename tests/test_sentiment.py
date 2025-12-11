"""Unit tests for SentimentNode."""

from unittest.mock import patch, MagicMock

import pytest

from policyflow.nodes.sentiment import SentimentNode
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


class TestSentimentNodeBasic:
    """Tests for basic sentiment analysis."""

    @patch("policyflow.llm.completion")
    def test_positive_sentiment(self, mock_completion, mock_config):
        """Positive sentiment should return 'positive' action."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: positive\nconfidence: 0.95"
        )

        node = SentimentNode(config=mock_config, cache_ttl=0)
        shared = {"input_text": "I love this product! It's amazing!"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "positive"
        assert shared["sentiment"]["label"] == "positive"
        assert shared["sentiment"]["confidence"] == 0.95

    @patch("policyflow.llm.completion")
    def test_negative_sentiment(self, mock_completion, mock_config):
        """Negative sentiment should return 'negative' action."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: negative\nconfidence: 0.88"
        )

        node = SentimentNode(config=mock_config, cache_ttl=0)
        shared = {"input_text": "This is terrible. I'm very disappointed."}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "negative"
        assert shared["sentiment"]["label"] == "negative"

    @patch("policyflow.llm.completion")
    def test_neutral_sentiment(self, mock_completion, mock_config):
        """Neutral sentiment should return 'neutral' action."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: neutral\nconfidence: 0.75"
        )

        node = SentimentNode(config=mock_config, cache_ttl=0)
        shared = {"input_text": "The meeting is scheduled for 3pm."}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "neutral"
        assert shared["sentiment"]["label"] == "neutral"

    @patch("policyflow.llm.completion")
    def test_mixed_sentiment(self, mock_completion, mock_config):
        """Mixed sentiment should return 'mixed' action."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: mixed\nconfidence: 0.65"
        )

        node = SentimentNode(config=mock_config, cache_ttl=0)
        shared = {"input_text": "I love the product but hate the service."}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "mixed"
        assert shared["sentiment"]["label"] == "mixed"


class TestSentimentNodeValidation:
    """Tests for sentiment validation and edge cases."""

    @patch("policyflow.llm.completion")
    def test_invalid_sentiment_defaults_neutral(self, mock_completion, mock_config):
        """Invalid sentiment label should default to neutral."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: happy\nconfidence: 0.9"  # 'happy' is not valid
        )

        node = SentimentNode(config=mock_config, cache_ttl=0)
        shared = {"input_text": "Test text"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "neutral"
        assert shared["sentiment"]["label"] == "neutral"

    @patch("policyflow.llm.completion")
    def test_case_insensitive_sentiment(self, mock_completion, mock_config):
        """Sentiment label should be case-insensitive."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: POSITIVE\nconfidence: 0.9"
        )

        node = SentimentNode(config=mock_config, cache_ttl=0)
        shared = {"input_text": "Great!"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "positive"


class TestSentimentNodeGranularity:
    """Tests for granularity levels."""

    @patch("policyflow.llm.completion")
    def test_basic_granularity(self, mock_completion, mock_config):
        """Basic mode should return only label and confidence."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: positive\nconfidence: 0.9"
        )

        node = SentimentNode(config=mock_config, granularity="basic", cache_ttl=0)
        shared = {"input_text": "Great product!"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "label" in shared["sentiment"]
        assert "confidence" in shared["sentiment"]
        assert "intensity" not in shared["sentiment"]
        assert "emotions" not in shared["sentiment"]

    @patch("policyflow.llm.completion")
    def test_detailed_granularity(self, mock_completion, mock_config):
        """Detailed mode should include intensity and emotions."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: positive\nconfidence: 0.95\nintensity: strong\nemotions:\n  - joy\n  - excitement"
        )

        node = SentimentNode(config=mock_config, granularity="detailed", cache_ttl=0)
        shared = {"input_text": "This is absolutely amazing! I'm thrilled!"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["sentiment"]["label"] == "positive"
        assert shared["sentiment"]["intensity"] == "strong"
        assert "joy" in shared["sentiment"]["emotions"]
        assert "excitement" in shared["sentiment"]["emotions"]

    @patch("policyflow.llm.completion")
    def test_detailed_granularity_missing_fields(self, mock_completion, mock_config):
        """Missing detailed fields should have defaults."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: negative\nconfidence: 0.8"
        )

        node = SentimentNode(config=mock_config, granularity="detailed", cache_ttl=0)
        shared = {"input_text": "This is bad"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["sentiment"]["intensity"] == "moderate"  # default
        assert shared["sentiment"]["emotions"] == []  # default


class TestSentimentNodeCustomInputKey:
    """Tests for custom input key configuration."""

    @patch("policyflow.llm.completion")
    def test_custom_input_key(self, mock_completion, mock_config):
        """Custom input_key should read from specified key."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: positive\nconfidence: 0.9"
        )

        node = SentimentNode(
            config=mock_config,
            input_key="custom_text",
            cache_ttl=0,
        )
        shared = {
            "custom_text": "I love it!",
            "input_text": "I hate it!",  # Should be ignored
        }

        prep_res = node.prep(shared)

        assert prep_res["input_text"] == "I love it!"

    @patch("policyflow.llm.completion")
    def test_missing_input_key(self, mock_completion, mock_config):
        """Missing input_key should default to empty string."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: neutral\nconfidence: 0.5"
        )

        node = SentimentNode(config=mock_config, cache_ttl=0)
        shared = {}

        prep_res = node.prep(shared)

        assert prep_res["input_text"] == ""


class TestSentimentNodeSharedStore:
    """Tests for shared store interactions."""

    @patch("policyflow.llm.completion")
    def test_result_stored_in_shared(self, mock_completion, mock_config):
        """Sentiment result should be stored in shared['sentiment']."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: positive\nconfidence: 0.85"
        )

        node = SentimentNode(config=mock_config, cache_ttl=0)
        shared = {"input_text": "Great work!"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "sentiment" in shared
        assert shared["sentiment"]["label"] == "positive"
        assert shared["sentiment"]["confidence"] == 0.85

    @patch("policyflow.llm.completion")
    def test_missing_confidence_defaults_to_zero(self, mock_completion, mock_config):
        """Missing confidence should default to 0.0."""
        mock_completion.return_value = create_mock_llm_response(
            "sentiment: positive"
        )

        node = SentimentNode(config=mock_config, cache_ttl=0)
        shared = {"input_text": "Good!"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert shared["sentiment"]["confidence"] == 0.0


class TestSentimentNodeValidSentiments:
    """Tests to verify VALID_SENTIMENTS constant behavior."""

    def test_valid_sentiments_set(self, mock_config):
        """VALID_SENTIMENTS should contain expected values."""
        assert SentimentNode.VALID_SENTIMENTS == {
            "positive",
            "negative",
            "neutral",
            "mixed",
        }

    @patch("policyflow.llm.completion")
    def test_all_valid_sentiments_work(self, mock_completion, mock_config):
        """All valid sentiment labels should be accepted."""
        for sentiment in ["positive", "negative", "neutral", "mixed"]:
            mock_completion.return_value = create_mock_llm_response(
                f"sentiment: {sentiment}\nconfidence: 0.9"
            )

            node = SentimentNode(config=mock_config, cache_ttl=0)
            shared = {"input_text": "Test"}

            prep_res = node.prep(shared)
            exec_res = node.exec(prep_res)
            action = node.post(shared, prep_res, exec_res)

            assert action == sentiment
            assert shared["sentiment"]["label"] == sentiment
