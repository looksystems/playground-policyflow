"""
Policy Evaluator - Generic policy evaluation using PocketFlow and LiteLLM.

Usage:
    from policyflow import parse_policy, evaluate

    # Parse policy to workflow
    parsed = parse_policy(open("policy.md").read())

    # Evaluate text
    from policyflow.workflow_builder import DynamicWorkflowBuilder
    builder = DynamicWorkflowBuilder(parsed)
    shared = builder.run("text to evaluate")

    # Or use CLI:
    #   policyflow parse --policy policy.md --save-workflow workflow.yaml
    #   policyflow eval --policy policy.md --input "text to evaluate"
"""

from pathlib import Path

from .models import (
    NormalizedPolicy,
    ParsedWorkflowPolicy,
    EvaluationResult,
    ClauseResult,
    LogicOperator,
    ConfidenceLevel,
    YAMLMixin,
    Clause,
    Section,
    ClauseType,
)
from .config import WorkflowConfig, ConfidenceGateConfig, get_config
from .parser import parse_policy, normalize_policy, generate_workflow_from_normalized
from .workflow_builder import DynamicWorkflowBuilder

__all__ = [
    "evaluate",
    "parse_policy",
    "normalize_policy",
    "generate_workflow_from_normalized",
    "DynamicWorkflowBuilder",
    "NormalizedPolicy",
    "ParsedWorkflowPolicy",
    "EvaluationResult",
    "ClauseResult",
    "WorkflowConfig",
    "ConfidenceGateConfig",
    "LogicOperator",
    "ConfidenceLevel",
    "YAMLMixin",
    "Clause",
    "Section",
    "ClauseType",
    "get_config",
]


def evaluate(
    input_text: str,
    policy_path: str | Path | None = None,
    policy_text: str | None = None,
    config: WorkflowConfig | None = None,
) -> EvaluationResult:
    """
    Evaluate input text against a policy.

    Args:
        input_text: The text to evaluate
        policy_path: Path to a markdown policy file
        policy_text: Raw policy markdown (alternative to policy_path)
        config: Optional workflow configuration

    Returns:
        EvaluationResult with evaluation details

    Example:
        >>> result = evaluate(
        ...     input_text="I recommend you buy XYZ stock based on your goals",
        ...     policy_path="policies/personal_recommendation.md"
        ... )
        >>> print(result.policy_satisfied)
        True
    """
    if policy_path:
        policy_text = Path(policy_path).read_text()
    elif not policy_text:
        raise ValueError("Must provide either policy_path or policy_text")

    config = config or WorkflowConfig()

    # Parse the policy using two-step parser
    parsed = parse_policy(policy_text, config)

    # Build and run workflow
    builder = DynamicWorkflowBuilder(parsed, config)
    shared = builder.run(input_text)

    # Build result from shared store
    from .models import ClauseResult

    policy_satisfied = shared.get("policy_satisfied", shared.get("satisfied", False))
    confidence = shared.get("confidence", shared.get("overall_confidence", 0.5))

    clause_results = []
    for key, value in shared.items():
        if key.endswith("_result") and isinstance(value, dict):
            clause_results.append(ClauseResult(
                clause_id=key.replace("_result", ""),
                clause_name=key.replace("_result", "").replace("_", " ").title(),
                met=value.get("met", False),
                reasoning=value.get("reasoning", ""),
                confidence=value.get("confidence", 0.5),
            ))

    return EvaluationResult(
        policy_title=parsed.title,
        policy_satisfied=policy_satisfied,
        input_text=input_text,
        overall_confidence=confidence,
        confidence_level="high" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "low",
        needs_review=confidence < 0.8,
        clause_results=clause_results,
        overall_reasoning=shared.get("reasoning", "Evaluated using dynamic workflow"),
    )
