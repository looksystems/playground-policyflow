"""Result aggregation node for PocketFlow."""

from pocketflow import Node

from .schema import NodeSchema
from .criterion import CriterionResult
from ..models import (
    ParsedPolicy,
    EvaluationResult,
    LogicOperator,
)


class ResultAggregatorNode(Node):
    """
    Aggregates criterion results according to policy logic.

    Shared Store:
        Reads: shared["parsed_policy"], shared["criterion_results"],
               shared["input_text"]
        Writes: shared["evaluation_result"]
    """

    parser_schema = NodeSchema(
        name="ResultAggregatorNode",
        description="Internal node for aggregating criterion results",
        category="internal",
        parameters=[],
        actions=["done"],
        parser_exposed=False,
    )

    def prep(self, shared: dict) -> dict:
        """Gather all criterion results."""
        return {
            "policy": shared["parsed_policy"],
            "results": shared["criterion_results"],
            "input_text": shared["input_text"],
        }

    def exec(self, prep_res: dict) -> dict:
        """Apply policy logic to determine overall result."""
        policy: ParsedPolicy = prep_res["policy"]
        results: dict[str, CriterionResult] = prep_res["results"]

        # Apply logic based on policy.logic
        criterion_results = list(results.values())
        met_values = [r.met for r in criterion_results]
        confidences = [r.confidence for r in criterion_results]

        if policy.logic == LogicOperator.ALL:
            policy_satisfied = all(met_values) if met_values else False
        elif policy.logic == LogicOperator.ANY:
            policy_satisfied = any(met_values) if met_values else False
        else:
            policy_satisfied = all(met_values) if met_values else False

        # Calculate overall confidence
        if confidences:
            overall_confidence = sum(confidences) / len(confidences)
        else:
            overall_confidence = 0.0

        # Build reasoning summary
        met_criteria = [r for r in criterion_results if r.met]
        unmet_criteria = [r for r in criterion_results if not r.met]

        reasoning_parts = []
        if met_criteria:
            reasoning_parts.append(
                f"Criteria met: {', '.join(r.criterion_name for r in met_criteria)}"
            )
        if unmet_criteria:
            reasoning_parts.append(
                f"Criteria not met: {', '.join(r.criterion_name for r in unmet_criteria)}"
            )

        return {
            "policy_satisfied": policy_satisfied,
            "overall_confidence": overall_confidence,
            "overall_reasoning": ". ".join(reasoning_parts) if reasoning_parts else "No criteria evaluated.",
            "criterion_results": criterion_results,
        }

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store final evaluation result."""
        shared["evaluation_result"] = EvaluationResult(
            policy_satisfied=exec_res["policy_satisfied"],
            input_text=prep_res["input_text"],
            policy_title=prep_res["policy"].title,
            criterion_results=exec_res["criterion_results"],
            overall_reasoning=exec_res["overall_reasoning"],
            overall_confidence=exec_res["overall_confidence"],
        )
        return "done"
