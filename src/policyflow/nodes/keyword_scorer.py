"""Keyword scoring node for deterministic input scoring."""

import re

from pocketflow import Node
from pydantic import BaseModel, Field

from .schema import NodeParameter, NodeSchema


class KeywordScoreResult(BaseModel):
    """Result from KeywordScorerNode."""

    score: float = Field(description="Total weighted score")
    level: str = Field(description="Score level: high, medium, or low")
    matched_keywords: dict[str, float] = Field(
        default_factory=dict,
        description="Keyword -> weight for matched keywords",
    )


class KeywordScorerNode(Node):
    """
    Scores input text based on weighted keyword presence.
    This is a deterministic node (no LLM) that matches keywords
    using case-insensitive word boundary matching.

    Shared Store:
        Reads: shared["input_text"]
        Writes: shared["keyword_score"] with score and matched keywords breakdown

    Actions:
        - "high": score >= high threshold
        - "medium": score >= medium threshold (but < high)
        - "low": score < medium threshold
    """

    parser_schema = NodeSchema(
        name="KeywordScorerNode",
        description="Score input based on weighted keyword presence (deterministic, no LLM)",
        category="deterministic",
        parameters=[
            NodeParameter(
                "keywords",
                "dict[str, float]",
                "Map keywords to weight scores (can be negative)",
                required=True,
            ),
            NodeParameter(
                "thresholds",
                "dict[str, float]",
                "Map score levels ('high', 'medium') to minimum scores",
                required=True,
            ),
        ],
        actions=["high", "medium", "low"],
        yaml_example="""- type: KeywordScorerNode
  id: urgency_scorer
  params:
    keywords:
      urgent: 0.5
      critical: 0.8
      asap: 0.3
      spam: -1.0
    thresholds:
      high: 0.7
      medium: 0.3
  routes:
    high: priority_queue
    medium: standard_queue
    low: low_priority_queue""",
        parser_exposed=True,
    )

    def __init__(self, keywords: dict[str, float], thresholds: dict[str, float]):
        """
        Initialize the keyword scorer.

        Args:
            keywords: Dict mapping keyword/phrase to weight score
                     Example: {"urgent": 0.5, "critical": 0.8, "spam": -1.0}
            thresholds: Dict mapping score level to minimum score
                       Example: {"high": 0.7, "medium": 0.3}
        """
        super().__init__()
        self.keywords = keywords
        self.thresholds = thresholds

        # Compile regex patterns for each keyword (case-insensitive, word boundaries)
        self.patterns = {}
        for keyword, weight in keywords.items():
            # Use word boundaries to avoid matching "urgently" for "urgent"
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            self.patterns[keyword] = (pattern, weight)

    def prep(self, shared: dict) -> dict:
        """Prepare input text for scoring."""
        return {
            "input_text": shared.get("input_text", ""),
        }

    def exec(self, prep_res: dict) -> dict:
        """Score the input based on keyword matches."""
        input_text = prep_res["input_text"]

        # Track matches and calculate score
        matched_keywords = {}
        total_score = 0.0

        for keyword, (pattern, weight) in self.patterns.items():
            matches = pattern.findall(input_text)
            match_count = len(matches)

            if match_count > 0:
                # Each match contributes the weight to the score
                contribution = weight * match_count
                matched_keywords[keyword] = {
                    "count": match_count,
                    "weight": weight,
                    "contribution": contribution,
                }
                total_score += contribution

        # Determine score level based on thresholds
        high_threshold = self.thresholds.get("high", float('inf'))
        medium_threshold = self.thresholds.get("medium", float('-inf'))

        if total_score >= high_threshold:
            score_level = "high"
        elif total_score >= medium_threshold:
            score_level = "medium"
        else:
            score_level = "low"

        return {
            "score": total_score,
            "score_level": score_level,
            "matched_keywords": matched_keywords,
            "total_matches": sum(k["count"] for k in matched_keywords.values()),
        }

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store keyword score result and return action based on score level."""
        shared["keyword_score"] = {
            "score": exec_res["score"],
            "score_level": exec_res["score_level"],
            "matched_keywords": exec_res["matched_keywords"],
            "total_matches": exec_res["total_matches"],
        }

        # Return the score level as the action for routing
        return exec_res["score_level"]
