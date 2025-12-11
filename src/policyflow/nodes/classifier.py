"""Classifier node for LLM-based text classification."""

from pydantic import BaseModel, Field

from .llm_node import LLMNode
from .schema import NodeParameter, NodeSchema
from ..config import WorkflowConfig
from ..templates import render


class ClassificationResult(BaseModel):
    """Result from ClassifierNode."""

    category: str = Field(description="Classified category")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(default="", description="Explanation for classification")


class ClassifierNode(LLMNode):
    """
    LLM-based node that classifies input into predefined categories.

    Shared Store:
        Reads: shared["input_text"]
        Writes: shared["classification"] with {category, confidence, reasoning}

    Actions:
        Returns the classified category name (one of the provided categories)
    """

    parser_schema = NodeSchema(
        name="ClassifierNode",
        description="Classify input into predefined categories using LLM",
        category="llm",
        parameters=[
            NodeParameter(
                "categories",
                "list[str]",
                "List of category names to classify into",
                required=True,
            ),
            NodeParameter(
                "descriptions",
                "dict[str, str]",
                "Optional descriptions for each category to guide classification",
                required=False,
                default=None,
            ),
        ],
        actions=["<category_name>"],
        yaml_example="""- type: ClassifierNode
  id: intent_classifier
  params:
    categories:
      - question
      - complaint
      - feedback
      - request
    descriptions:
      question: User is asking for information
      complaint: User is expressing dissatisfaction
      feedback: User is providing suggestions
      request: User is asking for action
  routes:
    question: answer_node
    complaint: escalate_node
    feedback: log_node
    request: action_node""",
        parser_exposed=True,
    )

    def __init__(
        self,
        categories: list[str],
        config: WorkflowConfig,
        model: str | None = None,
        descriptions: dict[str, str] | None = None,
        cache_ttl: int = 3600,
        rate_limit: int | None = None,
    ):
        """
        Initialize the classifier node.

        Args:
            categories: List of category names (e.g., ["spam", "legitimate", "unclear"])
            config: Workflow configuration
            model: LLM model identifier (uses class default_model if not provided)
            descriptions: Optional per-category guidance for the LLM
            cache_ttl: Cache time-to-live in seconds (default: 3600)
            rate_limit: Requests per minute limit (default: None = unlimited)
        """
        super().__init__(config=config, model=model, cache_ttl=cache_ttl, rate_limit=rate_limit)

        if not categories:
            raise ValueError("At least one category must be provided")

        self.categories = categories
        self.descriptions = descriptions or {}

    def prep(self, shared: dict) -> dict:
        """Prepare classification context."""
        return {
            "input_text": shared.get("input_text", ""),
            "categories": self.categories,
            "descriptions": self.descriptions,
        }

    def exec(self, prep_res: dict) -> dict:
        """Classify the input using LLM."""
        # Build the system prompt using the template
        system_prompt = render(
            "classifier.j2",
            categories=prep_res["categories"],
            descriptions=prep_res["descriptions"],
        )

        # Call LLM with caching
        result = self.call_llm(
            prompt=f"Classify this text:\n\n{prep_res['input_text']}",
            system_prompt=system_prompt,
            yaml_response=True,
            span_name="classifier",
        )

        # Validate and normalize the result
        category = result.get("category", "")

        # Check if category is valid
        if category not in self.categories:
            # Fallback to first category with low confidence
            return {
                "category": self.categories[0],
                "confidence": 0.0,
                "reasoning": f"LLM returned invalid category '{category}'. Falling back to '{self.categories[0]}'.",
            }

        return {
            "category": category,
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", ""),
        }

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store classification result and return the category as the action."""
        # Store the full classification result in shared store
        shared["classification"] = {
            "category": exec_res["category"],
            "confidence": exec_res["confidence"],
            "reasoning": exec_res["reasoning"],
        }

        # Return the category name as the action for routing
        return exec_res["category"]
