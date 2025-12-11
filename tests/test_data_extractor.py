"""Unit tests for DataExtractorNode."""

from unittest.mock import patch, MagicMock

import pytest

from policyflow.nodes.data_extractor import DataExtractorNode
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


class TestDataExtractorNodeEntities:
    """Tests for entity extraction."""

    @patch("policyflow.llm.completion")
    def test_extract_entities(self, mock_completion, mock_config):
        """Entities should be extracted correctly."""
        mock_completion.return_value = create_mock_llm_response("""
entities:
  people:
    - John Smith
    - Jane Doe
  organizations:
    - Acme Corp
    - Tech Inc
""")

        node = DataExtractorNode(
            schema={
                "entities": {
                    "people": "list of person names",
                    "organizations": "list of company names",
                }
            },
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "John Smith from Acme Corp met with Jane Doe from Tech Inc"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "default"
        assert "John Smith" in exec_res["entities"]["people"]
        assert "Acme Corp" in exec_res["entities"]["organizations"]

    @patch("policyflow.llm.completion")
    def test_extract_entities_empty_list(self, mock_completion, mock_config):
        """Empty entity lists should be handled."""
        mock_completion.return_value = create_mock_llm_response("""
entities:
  people: []
  organizations: []
""")

        node = DataExtractorNode(
            schema={
                "entities": {
                    "people": "list of person names",
                    "organizations": "list of company names",
                }
            },
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "No names or companies mentioned here."}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["entities"]["people"] == []
        assert exec_res["entities"]["organizations"] == []


class TestDataExtractorNodeValues:
    """Tests for value extraction."""

    @patch("policyflow.llm.completion")
    def test_extract_values(self, mock_completion, mock_config):
        """Values should be extracted correctly."""
        mock_completion.return_value = create_mock_llm_response("""
values:
  amount: $1,500.00
  date: December 15, 2024
""")

        node = DataExtractorNode(
            schema={
                "values": {
                    "amount": "monetary amount",
                    "date": "date reference",
                }
            },
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "The payment of $1,500.00 is due on December 15, 2024"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["values"]["amount"] == "$1,500.00"
        assert exec_res["values"]["date"] == "December 15, 2024"

    @patch("policyflow.llm.completion")
    def test_extract_values_null(self, mock_completion, mock_config):
        """Missing values should return None."""
        mock_completion.return_value = create_mock_llm_response("""
values:
  amount: null
  date: null
""")

        node = DataExtractorNode(
            schema={
                "values": {
                    "amount": "monetary amount",
                    "date": "date reference",
                }
            },
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "No specific amounts or dates mentioned."}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["values"]["amount"] is None
        assert exec_res["values"]["date"] is None


class TestDataExtractorNodeFacts:
    """Tests for fact extraction."""

    @patch("policyflow.llm.completion")
    def test_extract_facts(self, mock_completion, mock_config):
        """Facts should be extracted correctly."""
        mock_completion.return_value = create_mock_llm_response("""
facts:
  main topic: project deadline
  urgency level: high
""")

        node = DataExtractorNode(
            schema={
                "facts": ["main topic", "urgency level"]
            },
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "URGENT: The project deadline is approaching!"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["facts"]["main topic"] == "project deadline"
        assert exec_res["facts"]["urgency level"] == "high"


class TestDataExtractorNodeCombinedSchema:
    """Tests for combined extraction schema."""

    @patch("policyflow.llm.completion")
    def test_combined_schema(self, mock_completion, mock_config):
        """All three types (entities, values, facts) should work together."""
        mock_completion.return_value = create_mock_llm_response("""
entities:
  people:
    - John Smith
  organizations:
    - Acme Corp
values:
  amount: $5000
  date: 2024-01-15
facts:
  main topic: contract negotiation
  sentiment: positive
""")

        node = DataExtractorNode(
            schema={
                "entities": {
                    "people": "list of person names",
                    "organizations": "list of company names",
                },
                "values": {
                    "amount": "monetary amount",
                    "date": "date reference",
                },
                "facts": ["main topic", "sentiment"],
            },
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "John Smith from Acme Corp is negotiating a $5000 contract for 2024-01-15"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert "John Smith" in exec_res["entities"]["people"]
        assert "Acme Corp" in exec_res["entities"]["organizations"]
        assert exec_res["values"]["amount"] == "$5000"
        assert exec_res["facts"]["main topic"] == "contract negotiation"


class TestDataExtractorNodeMissingFields:
    """Tests for handling missing fields in LLM response."""

    @patch("policyflow.llm.completion")
    def test_missing_entity_type(self, mock_completion, mock_config):
        """Missing entity type in response should return empty list."""
        mock_completion.return_value = create_mock_llm_response("""
entities:
  people:
    - John Smith
""")
        # organizations is missing from response

        node = DataExtractorNode(
            schema={
                "entities": {
                    "people": "list of person names",
                    "organizations": "list of company names",
                }
            },
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "John Smith mentioned"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert "John Smith" in exec_res["entities"]["people"]
        assert exec_res["entities"]["organizations"] == []  # Default empty list

    @patch("policyflow.llm.completion")
    def test_missing_value_field(self, mock_completion, mock_config):
        """Missing value field in response should return None."""
        mock_completion.return_value = create_mock_llm_response("""
values:
  amount: $100
""")
        # date is missing from response

        node = DataExtractorNode(
            schema={
                "values": {
                    "amount": "monetary amount",
                    "date": "date reference",
                }
            },
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Cost is $100"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["values"]["amount"] == "$100"
        assert exec_res["values"]["date"] is None  # Default None

    @patch("policyflow.llm.completion")
    def test_missing_fact(self, mock_completion, mock_config):
        """Missing fact in response should return None."""
        mock_completion.return_value = create_mock_llm_response("""
facts:
  main topic: testing
""")
        # urgency is missing from response

        node = DataExtractorNode(
            schema={
                "facts": ["main topic", "urgency"]
            },
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Testing something"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["facts"]["main topic"] == "testing"
        assert exec_res["facts"]["urgency"] is None  # Default None


class TestDataExtractorNodeSharedStore:
    """Tests for shared store interactions."""

    @patch("policyflow.llm.completion")
    def test_result_stored_in_shared(self, mock_completion, mock_config):
        """Extracted data should be stored in shared['extracted_data']."""
        mock_completion.return_value = create_mock_llm_response("""
entities:
  people:
    - Test Person
""")

        node = DataExtractorNode(
            schema={"entities": {"people": "list of names"}},
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test Person is here"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "extracted_data" in shared
        assert "Test Person" in shared["extracted_data"]["entities"]["people"]

    @patch("policyflow.llm.completion")
    def test_missing_input_text(self, mock_completion, mock_config):
        """Missing input_text should default to empty string."""
        mock_completion.return_value = create_mock_llm_response("""
entities:
  people: []
""")

        node = DataExtractorNode(
            schema={"entities": {"people": "list of names"}},
            config=mock_config,
            cache_ttl=0,
        )
        shared = {}

        prep_res = node.prep(shared)

        assert prep_res["input_text"] == ""


class TestDataExtractorNodeEdgeCases:
    """Edge case tests for DataExtractorNode."""

    @patch("policyflow.llm.completion")
    def test_returns_default_action(self, mock_completion, mock_config):
        """DataExtractorNode should always return 'default' action."""
        mock_completion.return_value = create_mock_llm_response("""
entities:
  people: []
""")

        node = DataExtractorNode(
            schema={"entities": {"people": "list of names"}},
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "default"

    @patch("policyflow.llm.completion")
    def test_empty_schema_sections(self, mock_completion, mock_config):
        """Only requested schema sections should appear in result."""
        mock_completion.return_value = create_mock_llm_response("""
facts:
  topic: testing
""")

        node = DataExtractorNode(
            schema={"facts": ["topic"]},
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        # Only facts should be present, not entities or values
        assert "facts" in exec_res
        assert "entities" not in exec_res
        assert "values" not in exec_res

    @patch("policyflow.llm.completion")
    def test_schema_passed_to_prep(self, mock_completion, mock_config):
        """Schema should be available in prep result."""
        mock_completion.return_value = create_mock_llm_response("entities: {}")

        schema = {"entities": {"people": "list of names"}}
        node = DataExtractorNode(
            schema=schema,
            config=mock_config,
            cache_ttl=0,
        )
        shared = {"input_text": "Test"}

        prep_res = node.prep(shared)

        assert prep_res["schema"] == schema

    def test_default_config(self):
        """Node should work with default config."""
        node = DataExtractorNode(
            schema={"facts": ["topic"]},
            cache_ttl=0,
        )
        # Should not raise, config defaults to WorkflowConfig()
        assert node.config is not None
