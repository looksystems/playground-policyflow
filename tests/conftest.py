"""Shared pytest fixtures for node type tests."""

from unittest.mock import MagicMock

import pytest

from policyflow.config import WorkflowConfig, ConfidenceGateConfig
from policyflow.models import (
    Clause,
    Section,
    NormalizedPolicy,
    LogicOperator,
    ClauseType,
)


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
def sample_clause():
    """Return a sample Clause for testing."""
    return Clause(
        number="1.1",
        title="Test Clause",
        text="This is a test clause for unit testing",
        clause_type=ClauseType.REQUIREMENT,
    )


@pytest.fixture
def sample_clause_with_sub():
    """Return a Clause with sub-clauses for testing."""
    return Clause(
        number="1.1",
        title="Parent Clause",
        text="Parent clause with sub-clauses",
        clause_type=ClauseType.REQUIREMENT,
        sub_clauses=[
            Clause(
                number="1.1.a",
                title="Sub Clause 1",
                text="First sub-clause",
            ),
            Clause(
                number="1.1.b",
                title="Sub Clause 2",
                text="Second sub-clause",
            ),
        ],
        logic=LogicOperator.ALL,
    )


@pytest.fixture
def sample_normalized_policy(sample_clause):
    """Return a sample NormalizedPolicy for testing."""
    return NormalizedPolicy(
        title="Test Policy",
        description="A test policy for unit testing",
        sections=[
            Section(
                number="1",
                title="Test Section",
                clauses=[sample_clause],
            )
        ],
        logic=LogicOperator.ALL,
        raw_text="# Test Policy\n\nThis is a test.",
    )


@pytest.fixture
def sample_clause_result():
    """Return a sample clause result dict for testing."""
    return {
        "clause_id": "clause_1_1",
        "clause_name": "Test Clause",
        "met": True,
        "reasoning": "The clause was met because...",
        "confidence": 0.85,
    }


@pytest.fixture
def sample_clause_results():
    """Return multiple clause results for testing aggregation."""
    return {
        "clause_1_1_result": {
            "met": True,
            "reasoning": "First clause met",
            "confidence": 0.9,
        },
        "clause_1_2_result": {
            "met": True,
            "reasoning": "Second clause met",
            "confidence": 0.8,
        },
        "clause_1_3_result": {
            "met": False,
            "reasoning": "Third clause not met",
            "confidence": 0.7,
        },
    }


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
def low_confidence_results():
    """Return results with some low confidence scores."""
    return {
        "clause_1_result": {
            "met": True,
            "reasoning": "Not very confident",
            "confidence": 0.3,
        },
        "clause_2_result": {
            "met": True,
            "reasoning": "Somewhat confident",
            "confidence": 0.6,
        },
    }
