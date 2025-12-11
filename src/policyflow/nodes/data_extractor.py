"""Data extraction node for PocketFlow."""

from .llm_node import LLMNode
from .schema import NodeParameter, NodeSchema
from ..config import WorkflowConfig
from ..templates import render


class DataExtractorNode(LLMNode):
    """
    LLM-based node that extracts structured data from input text.

    Uses a flexible schema to define what to extract:
    - entities: Named entity types (people, organizations, etc.)
    - values: Specific value types (amounts, dates, etc.)
    - facts: Simple facts to extract (topic, sentiment, etc.)

    Shared Store:
        Reads: shared["input_text"]
        Writes: shared["extracted_data"]
    """

    parser_schema = NodeSchema(
        name="DataExtractorNode",
        description="Extract structured data (entities, values, facts) from text using LLM",
        category="llm",
        parameters=[
            NodeParameter(
                "schema",
                "dict",
                "Extraction schema with 'entities' (dict), 'values' (dict), and/or 'facts' (list)",
                required=True,
            ),
        ],
        actions=["default"],
        yaml_example="""- type: DataExtractorNode
  id: extract_info
  params:
    schema:
      entities:
        people: list of person names
        organizations: list of company/org names
      values:
        amounts: monetary amounts mentioned
        dates: date references
      facts:
        - main topic
        - urgency level
  routes:
    default: process_extracted""",
        parser_exposed=True,
    )

    def __init__(
        self,
        schema: dict,
        config: WorkflowConfig | None = None,
        model: str | None = None,
        cache_ttl: int = 3600,
        rate_limit: int | None = None,
    ):
        """
        Initialize data extractor node.

        Args:
            schema: Extraction schema with optional keys:
                - entities: dict[str, str] - Entity types and descriptions
                  e.g. {"people": "list of person names", "orgs": "organizations"}
                - values: dict[str, str] - Value types and descriptions
                  e.g. {"amounts": "monetary amounts", "dates": "date references"}
                - facts: list[str] - Simple facts to extract
                  e.g. ["main topic", "sentiment"]
            config: Workflow configuration
            model: LLM model identifier (uses class default_model if not provided)
            cache_ttl: Cache time-to-live in seconds (0 = disabled)
            rate_limit: Requests per minute (None = unlimited)
        """
        self.schema = schema
        super().__init__(
            config=config or WorkflowConfig(),
            model=model,
            cache_ttl=cache_ttl,
            rate_limit=rate_limit,
        )

    def prep(self, shared: dict) -> dict:
        """Prepare extraction context."""
        return {
            "input_text": shared.get("input_text", ""),
            "schema": self.schema,
        }

    def exec(self, prep_res: dict) -> dict:
        """Extract structured data using LLM."""
        # Build system prompt from schema
        system_prompt = render("extractor.j2", schema=prep_res["schema"])

        # Call LLM with the input text
        result = self.call_llm(
            prompt=f"Extract information from this text:\n\n{prep_res['input_text']}",
            system_prompt=system_prompt,
            yaml_response=True,
            span_name="data_extraction",
        )

        # Ensure we have a valid structure, handling empty/missing extractions
        extracted_data = {}

        # Process entities (dict of lists)
        if "entities" in self.schema:
            extracted_data["entities"] = {}
            entities_result = result.get("entities", {})
            for entity_type in self.schema["entities"].keys():
                extracted_data["entities"][entity_type] = entities_result.get(
                    entity_type, []
                )

        # Process values (dict of values)
        if "values" in self.schema:
            extracted_data["values"] = {}
            values_result = result.get("values", {})
            for value_type in self.schema["values"].keys():
                extracted_data["values"][value_type] = values_result.get(
                    value_type, None
                )

        # Process facts (dict of facts)
        if "facts" in self.schema:
            extracted_data["facts"] = {}
            facts_result = result.get("facts", {})
            for fact in self.schema["facts"]:
                extracted_data["facts"][fact] = facts_result.get(fact, None)

        return extracted_data

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store extracted data."""
        shared["extracted_data"] = exec_res
        return "default"
