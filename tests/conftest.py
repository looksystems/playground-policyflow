"""Shared pytest fixtures for node type tests."""

from unittest.mock import MagicMock

import pytest

from policyflow.config import WorkflowConfig, ConfidenceGateConfig
from policyflow.models import (
    Criterion,
    ParsedPolicy,
    LogicOperator,
)
from policyflow.nodes.criterion import CriterionResult


@pytest.fixture
def mock_config():
    """Return a WorkflowConfig with test defaults."""
    return WorkflowConfig(
        temperature=0.0,
        max_retries=1,
        retry_wait=0,
    )


@pytest.fixture
def sample_texts():
    """Common test input strings."""
    return {
        "simple": "Hello world",
        "urgent_email": "URGENT: Please review this critical issue ASAP",
        "html_content": "<p>This is <b>bold</b> text</p>",
        "with_urls": "Check out https://example.com for more info",
        "with_emails": "Contact support@example.com for help",
        "multiline": "First line\nSecond line\nThird line",
        "whitespace": "  Too   many    spaces   ",
        "long_text": "Lorem ipsum " * 100,
        "empty": "",
        "pii_ssn": "My SSN is 123-45-6789",
        "password": "The password is secret123",
    }


@pytest.fixture
def mock_llm_response():
    """Factory fixture for creating mock LLM responses."""

    def _create_response(content: str):
        """Create a mock LiteLLM completion response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        return mock_response

    return _create_response


@pytest.fixture
def sample_criterion():
    """Return a sample Criterion for testing."""
    return Criterion(
        id="criterion_1",
        name="Test Criterion",
        description="This is a test criterion for unit testing",
    )


@pytest.fixture
def sample_criterion_with_sub():
    """Return a Criterion with sub-criteria for testing."""
    return Criterion(
        id="criterion_1",
        name="Parent Criterion",
        description="Parent criterion with sub-criteria",
        sub_criteria=[
            Criterion(
                id="sub_1",
                name="Sub Criterion 1",
                description="First sub-criterion",
            ),
            Criterion(
                id="sub_2",
                name="Sub Criterion 2",
                description="Second sub-criterion",
            ),
        ],
        sub_logic=LogicOperator.ALL,
    )


@pytest.fixture
def sample_parsed_policy(sample_criterion):
    """Return a sample ParsedPolicy for testing."""
    return ParsedPolicy(
        title="Test Policy",
        description="A test policy for unit testing",
        criteria=[sample_criterion],
        logic=LogicOperator.ALL,
        raw_text="# Test Policy\n\nThis is a test.",
    )


@pytest.fixture
def sample_criterion_result():
    """Return a sample CriterionResult for testing."""
    return CriterionResult(
        criterion_id="criterion_1",
        criterion_name="Test Criterion",
        met=True,
        reasoning="The criterion was met because...",
        confidence=0.85,
    )


@pytest.fixture
def sample_criterion_results():
    """Return multiple CriterionResults for testing aggregation."""
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
            confidence=0.8,
        ),
        "criterion_3": CriterionResult(
            criterion_id="criterion_3",
            criterion_name="Third Criterion",
            met=False,
            reasoning="Third criterion not met",
            confidence=0.7,
        ),
    }


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
def low_confidence_results():
    """Return criterion results with some low confidence scores."""
    return {
        "criterion_1": CriterionResult(
            criterion_id="criterion_1",
            criterion_name="Low Confidence",
            met=True,
            reasoning="Not very confident",
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
