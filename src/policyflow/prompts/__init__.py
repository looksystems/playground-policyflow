"""Prompt management using Jinja2 templates."""

from ..templates import render


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
    "get_normalize_policy_prompt",
    "get_workflow_from_normalized_prompt",
]
