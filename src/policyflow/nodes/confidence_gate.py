"""Confidence gate node for routing based on confidence thresholds."""

from pocketflow import Node

from .schema import NodeSchema
from ..models import ConfidenceLevel
from ..config import WorkflowConfig, ConfidenceGateConfig


class ConfidenceGateNode(Node):
    """
    Routes workflow based on confidence thresholds.

    Evaluates confidence scores from results stored in the shared dict
    and routes to different paths based on configured thresholds.

    Shared Store:
        Reads: Results from shared dict (looking for .confidence attributes or 'confidence' keys)
        Writes: shared["confidence_gate_result"]

    Actions:
        - "high_confidence": All results above high threshold
        - "needs_review": Some results between thresholds
        - "low_confidence": Any result below low threshold
    """

    parser_schema = NodeSchema(
        name="ConfidenceGateNode",
        description="Internal node for routing based on confidence thresholds",
        category="internal",
        parameters=[],
        actions=["high_confidence", "needs_review", "low_confidence"],
        parser_exposed=False,
    )

    def __init__(self, config: WorkflowConfig | None = None):
        super().__init__()
        self.config = config or WorkflowConfig()
        self.gate_config = self.config.confidence_gate

    def prep(self, shared: dict) -> dict:
        """Gather results with confidence scores."""
        # Look for any results that have confidence scores
        results = {}
        for key, value in shared.items():
            if key.endswith("_result") and isinstance(value, dict):
                if "confidence" in value:
                    results[key] = value
            elif hasattr(value, "confidence"):
                results[key] = {"confidence": value.confidence}

        return {
            "results": results,
            "high_threshold": self.gate_config.high_threshold,
            "low_threshold": self.gate_config.low_threshold,
        }

    def exec(self, prep_res: dict) -> dict:
        """Evaluate confidence levels against thresholds."""
        results = prep_res["results"]
        high_threshold = prep_res["high_threshold"]
        low_threshold = prep_res["low_threshold"]

        if not results:
            return {
                "confidence_level": ConfidenceLevel.LOW,
                "low_confidence_items": [],
                "needs_review": True,
                "reason": "No results with confidence scores to evaluate",
            }

        confidences = []
        low_confidence_items = []
        medium_confidence_items = []

        for item_id, result in results.items():
            confidence = result.get("confidence", 0.5)
            confidences.append(confidence)

            if confidence < low_threshold:
                low_confidence_items.append(item_id)
            elif confidence < high_threshold:
                medium_confidence_items.append(item_id)

        avg_confidence = sum(confidences) / len(confidences)

        # Determine overall confidence level
        if low_confidence_items:
            confidence_level = ConfidenceLevel.LOW
            needs_review = True
            reason = f"Low confidence on: {', '.join(low_confidence_items)}"
        elif medium_confidence_items:
            confidence_level = ConfidenceLevel.MEDIUM
            needs_review = True
            reason = f"Medium confidence on: {', '.join(medium_confidence_items)}"
        else:
            confidence_level = ConfidenceLevel.HIGH
            needs_review = False
            reason = "All items evaluated with high confidence"

        return {
            "confidence_level": confidence_level,
            "low_confidence_items": low_confidence_items,
            "medium_confidence_items": medium_confidence_items,
            "needs_review": needs_review,
            "reason": reason,
            "average_confidence": avg_confidence,
        }

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store gate result and return routing action."""
        shared["confidence_gate_result"] = exec_res

        confidence_level = exec_res["confidence_level"]

        if confidence_level == ConfidenceLevel.HIGH:
            return "high_confidence"
        elif confidence_level == ConfidenceLevel.LOW:
            return "low_confidence"
        else:
            return "needs_review"
