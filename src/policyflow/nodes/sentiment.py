"""Sentiment analysis node for PocketFlow."""

from typing import Literal

from pydantic import BaseModel, Field

from .llm_node import LLMNode
from .schema import NodeParameter, NodeSchema
from ..config import WorkflowConfig
from ..templates import render


class SentimentResult(BaseModel):
    """Result from SentimentNode."""

    label: str = Field(description="Sentiment label: positive, negative, neutral, mixed")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    intensity: str | None = Field(
        default=None, description="Intensity: strong, moderate, weak (detailed mode)"
    )
    emotions: list[str] = Field(
        default_factory=list, description="Detected emotions (detailed mode)"
    )


class SentimentNode(LLMNode):
    """
    Classifies sentiment/emotional tone of input text using LLM.

    Supports two granularity levels:
    - basic: Returns positive/negative/neutral/mixed with confidence
    - detailed: Also includes intensity (strong/moderate/weak) and specific emotions

    Shared Store:
        Reads: shared["input_text"]
        Writes: shared["sentiment"] with label, confidence, and optional details

    Actions:
        Returns the sentiment label: "positive" | "negative" | "neutral" | "mixed"
    """

    VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}

    parser_schema = NodeSchema(
        name="SentimentNode",
        description="Analyze sentiment/emotional tone of text using LLM",
        category="llm",
        parameters=[
            NodeParameter(
                "granularity",
                "str",
                "'basic' (sentiment + confidence) or 'detailed' (adds intensity + emotions)",
                required=False,
                default="basic",
            ),
        ],
        actions=["positive", "negative", "neutral", "mixed"],
        yaml_example="""- type: SentimentNode
  id: sentiment_check
  params:
    granularity: detailed
  routes:
    positive: happy_path
    negative: escalate
    neutral: standard_path
    mixed: review_needed""",
        parser_exposed=True,
    )

    def __init__(
        self,
        config: WorkflowConfig,
        model: str | None = None,
        granularity: Literal["basic", "detailed"] = "basic",
        input_key: str = "input_text",
        cache_ttl: int = 3600,
        rate_limit: int = None,
    ):
        """
        Initialize sentiment analysis node.

        Args:
            config: Workflow configuration
            model: LLM model identifier (uses class default_model if not provided)
            granularity: Analysis detail level ("basic" or "detailed")
            input_key: Key to read input text from shared store
            cache_ttl: Cache time-to-live in seconds (0 = disabled)
            rate_limit: Requests per minute (None = unlimited)
        """
        super().__init__(config=config, model=model, cache_ttl=cache_ttl, rate_limit=rate_limit)
        self.granularity = granularity
        self.input_key = input_key

    def prep(self, shared: dict) -> dict:
        """Retrieve input text from shared store."""
        return {
            "input_text": shared.get(self.input_key, ""),
        }

    def exec(self, prep_res: dict) -> dict:
        """Analyze sentiment using LLM."""
        # Build system prompt from template
        system_prompt = render("sentiment.j2", granularity=self.granularity)

        # Call LLM with caching and rate limiting
        result = self.call_llm(
            prompt=f"Analyze this text:\n\n{prep_res['input_text']}",
            system_prompt=system_prompt,
            yaml_response=True,
            span_name="sentiment_analysis",
        )

        return result

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store sentiment result and return action."""
        # Validate sentiment label
        sentiment_label = exec_res.get("sentiment", "neutral").lower()
        if sentiment_label not in self.VALID_SENTIMENTS:
            # Default to neutral if invalid
            sentiment_label = "neutral"

        # Build sentiment result
        sentiment_result = {
            "label": sentiment_label,
            "confidence": exec_res.get("confidence", 0.0),
        }

        # Add detailed fields if in detailed mode
        if self.granularity == "detailed":
            sentiment_result["intensity"] = exec_res.get("intensity", "moderate")
            sentiment_result["emotions"] = exec_res.get("emotions", [])

        # Store in shared store
        shared["sentiment"] = sentiment_result

        # Return sentiment label as action for routing
        return sentiment_label
