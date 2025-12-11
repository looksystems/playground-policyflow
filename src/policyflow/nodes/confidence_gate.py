"""Confidence gate node for routing based on confidence thresholds."""

from pocketflow import Node

from .schema import NodeSchema
from .criterion import CriterionResult
from ..models import ConfidenceLevel
from ..config import WorkflowConfig, ConfidenceGateConfig


class ConfidenceGateNode(Node):
    """
    Routes workflow based on confidence thresholds.

    Shared Store:
        Reads: shared["criterion_results"]
        Writes: shared["confidence_gate_result"]

    Actions:
        - "high_confidence": All criteria above high threshold
        - "needs_review": Some criteria between thresholds
        - "low_confidence": Any criterion below low threshold
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
        """Gather criterion results."""
        return {
            "criterion_results": shared.get("criterion_results", {}),
            "high_threshold": self.gate_config.high_threshold,
            "low_threshold": self.gate_config.low_threshold,
        }

    def exec(self, prep_res: dict) -> dict:
        """Evaluate confidence levels against thresholds."""
        results: dict[str, CriterionResult] = prep_res["criterion_results"]
        high_threshold = prep_res["high_threshold"]
        low_threshold = prep_res["low_threshold"]

        if not results:
            return {
                "confidence_level": ConfidenceLevel.LOW,
                "low_confidence_criteria": [],
                "needs_review": True,
                "reason": "No criterion results to evaluate",
            }

        confidences = []
        low_confidence_criteria = []
        medium_confidence_criteria = []

        for criterion_id, result in results.items():
            confidence = result.confidence
            confidences.append(confidence)

            if confidence < low_threshold:
                low_confidence_criteria.append(criterion_id)
            elif confidence < high_threshold:
                medium_confidence_criteria.append(criterion_id)

        avg_confidence = sum(confidences) / len(confidences)

        # Determine overall confidence level
        if low_confidence_criteria:
            confidence_level = ConfidenceLevel.LOW
            needs_review = True
            reason = f"Low confidence on: {', '.join(low_confidence_criteria)}"
        elif medium_confidence_criteria:
            confidence_level = ConfidenceLevel.MEDIUM
            needs_review = True
            reason = f"Medium confidence on: {', '.join(medium_confidence_criteria)}"
        else:
            confidence_level = ConfidenceLevel.HIGH
            needs_review = False
            reason = "All criteria evaluated with high confidence"

        return {
            "confidence_level": confidence_level,
            "low_confidence_criteria": low_confidence_criteria,
            "medium_confidence_criteria": medium_confidence_criteria,
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
