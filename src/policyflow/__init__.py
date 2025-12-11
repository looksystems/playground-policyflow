"""
Policy Evaluator - Generic policy evaluation using PocketFlow and LiteLLM.

Usage:
    from policyflow import evaluate, parse_policy, PolicyEvaluationWorkflow

    # Quick evaluation
    result = evaluate(policy_path="policy.md", input_text="...")

    # Or with more control
    policy = parse_policy(open("policy.md").read())
    workflow = PolicyEvaluationWorkflow(policy)
    result = workflow.run("text to evaluate")
"""

from pathlib import Path

from .models import (
    ParsedPolicy,
    Criterion,
    EvaluationResult,
    LogicOperator,
    ConfidenceLevel,
    YAMLMixin,
)
from .nodes.criterion import CriterionResult
from .nodes.subcriterion import SubCriterionResult
from .config import WorkflowConfig, ConfidenceGateConfig, get_config
from .parser import parse_policy
from .workflow import PolicyEvaluationWorkflow

__all__ = [
    "evaluate",
    "parse_policy",
    "PolicyEvaluationWorkflow",
    "ParsedPolicy",
    "Criterion",
    "CriterionResult",
    "SubCriterionResult",
    "EvaluationResult",
    "WorkflowConfig",
    "ConfidenceGateConfig",
    "LogicOperator",
    "ConfidenceLevel",
    "YAMLMixin",
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
        EvaluationResult with detailed criterion-by-criterion results

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

    # Parse the policy
    parsed_policy = parse_policy(policy_text, config)

    # Build and run workflow
    workflow = PolicyEvaluationWorkflow(parsed_policy, config)
    return workflow.run(input_text)
