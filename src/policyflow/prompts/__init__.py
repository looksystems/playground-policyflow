"""Prompt management using Jinja2 templates."""

from ..templates import render
from ..models import Criterion


def get_criteria_parser_prompt() -> str:
    """Get the system prompt for parsing policies into criteria structure.

    This is the original parsing mode that extracts criteria hierarchy.
    """
    return render("policy_parser_criteria.j2")


def get_workflow_parser_prompt() -> str:
    """Get the system prompt for parsing policies into workflows.

    Dynamically includes documentation for all parser-exposed nodes.
    """
    # Import here to avoid circular import (nodes -> prompts -> nodes.registry)
    from ..nodes.registry import get_parser_schemas

    schemas = get_parser_schemas()
    return render("policy_parser.j2", available_nodes=schemas)


# Keep old name for backwards compatibility (maps to workflow parser)
def get_policy_parser_prompt() -> str:
    """Alias for get_workflow_parser_prompt()."""
    return get_workflow_parser_prompt()


def build_criterion_prompt(criterion: Criterion, policy_context: str) -> str:
    """
    Build the evaluation prompt for a specific criterion.

    Args:
        criterion: The criterion to evaluate
        policy_context: Description of the overall policy

    Returns:
        Rendered prompt string
    """
    return render(
        "criterion_eval.j2",
        criterion=criterion,
        policy_context=policy_context,
    )


def build_subcriterion_prompt(
    parent_criterion: Criterion,
    sub_criterion: Criterion,
    policy_context: str,
) -> str:
    """
    Build the evaluation prompt for a specific sub-criterion.

    Args:
        parent_criterion: The parent criterion
        sub_criterion: The sub-criterion to evaluate
        policy_context: Description of the overall policy

    Returns:
        Rendered prompt string
    """
    return render(
        "subcriterion_eval.j2",
        parent_criterion=parent_criterion,
        sub_criterion=sub_criterion,
        policy_context=policy_context,
    )


# ============================================================================
# Two-Step Parser Prompts
# ============================================================================


def get_normalize_policy_prompt() -> str:
    """Get the system prompt for normalizing policies (Step 1).

    This prompt instructs the LLM to parse raw policy markdown into
    a structured NormalizedPolicy with hierarchical numbering.
    """
    return render("policy_normalizer.j2")


def get_workflow_from_normalized_prompt() -> str:
    """Get the system prompt for generating workflows from normalized policies (Step 2).

    This prompt instructs the LLM to generate a workflow where node IDs
    match clause numbers for explainability and audit trails.

    Dynamically includes documentation for all parser-exposed nodes.
    """
    from ..nodes.registry import get_parser_schemas

    schemas = get_parser_schemas()
    return render("workflow_from_normalized.j2", available_nodes=schemas)


__all__ = [
    "get_policy_parser_prompt",
    "get_criteria_parser_prompt",
    "get_workflow_parser_prompt",
    "get_normalize_policy_prompt",
    "get_workflow_from_normalized_prompt",
    "build_criterion_prompt",
    "build_subcriterion_prompt",
]
