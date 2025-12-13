"""Policy parsing using LLM."""

from pathlib import Path

from .models import (
    NormalizedPolicy,
    ParsedWorkflowPolicy,
)
from .config import WorkflowConfig
from .llm import call_llm
from .prompts import (
    get_normalize_policy_prompt,
    get_workflow_from_normalized_prompt,
)

# Default model for parsing operations
DEFAULT_PARSER_MODEL = "anthropic/claude-sonnet-4-20250514"


def normalize_policy(
    policy_markdown: str,
    config: WorkflowConfig | None = None,
    model: str | None = None,
) -> NormalizedPolicy:
    """
    Step 1: Normalize a markdown policy into structured sections and clauses.

    This is the first step of the two-step parsing process. The resulting
    NormalizedPolicy can be:
    - Reviewed and edited before workflow generation
    - Persisted to YAML for audit trails
    - Used as input to generate_workflow_from_normalized()

    Args:
        policy_markdown: Raw markdown text of the policy
        config: Optional workflow configuration
        model: LLM model identifier (uses DEFAULT_PARSER_MODEL if not provided)

    Returns:
        NormalizedPolicy with hierarchical structure and consistent numbering

    Example:
        >>> policy = open("policy.md").read()
        >>> normalized = normalize_policy(policy)
        >>> normalized.save_yaml("normalized_policy.yaml")  # For review
        >>> print(normalized.sections[0].clauses[0].number)  # "1.1"
    """
    config = config or WorkflowConfig()
    model = model or DEFAULT_PARSER_MODEL

    data = call_llm(
        prompt=f"Normalize this policy document:\n\n{policy_markdown}",
        system_prompt=get_normalize_policy_prompt(),
        model=model,
        config=config,
        yaml_response=True,
        span_name="normalize_policy",
    )

    return NormalizedPolicy.model_validate({
        **data,
        "raw_text": policy_markdown,
    })


def generate_workflow_from_normalized(
    normalized: NormalizedPolicy,
    config: WorkflowConfig | None = None,
    model: str | None = None,
    normalized_policy_path: str | None = None,
) -> ParsedWorkflowPolicy:
    """
    Step 2: Generate a workflow from a normalized policy document.

    This is the second step of the two-step parsing process. It takes
    the structured NormalizedPolicy and generates an executable workflow
    where node IDs correspond to clause numbers for explainability.

    Args:
        normalized: A NormalizedPolicy from normalize_policy()
        config: Optional workflow configuration
        model: LLM model identifier (uses DEFAULT_PARSER_MODEL if not provided)
        normalized_policy_path: Optional path to store reference in output

    Returns:
        ParsedWorkflowPolicy with hierarchical workflow and clause mapping

    Example:
        >>> normalized = NormalizedPolicy.load_yaml("normalized.yaml")
        >>> workflow = generate_workflow_from_normalized(normalized)
        >>> workflow.save_yaml("workflow.yaml")
    """
    config = config or WorkflowConfig()
    model = model or DEFAULT_PARSER_MODEL

    data = call_llm(
        prompt=f"Generate a workflow for this normalized policy:\n\n{normalized.to_yaml()}",
        system_prompt=get_workflow_from_normalized_prompt(),
        model=model,
        config=config,
        yaml_response=True,
        span_name="generate_workflow_from_normalized",
    )

    return ParsedWorkflowPolicy.model_validate({
        **data,
        "normalized_policy_ref": normalized_policy_path,
        "raw_text": normalized.raw_text,
    })


def parse_policy(
    policy_markdown: str,
    config: WorkflowConfig | None = None,
    model: str | None = None,
    save_normalized: str | Path | None = None,
) -> ParsedWorkflowPolicy:
    """
    Parse a markdown policy into an executable workflow.

    Uses two-step parsing: first normalizes the policy into structured
    sections/clauses, then generates a workflow with node IDs matching
    clause numbers for explainability.

    Args:
        policy_markdown: Raw markdown text of the policy
        config: Optional workflow configuration
        model: LLM model identifier
        save_normalized: Optional path to save intermediate normalized policy YAML

    Returns:
        ParsedWorkflowPolicy with hierarchical workflow

    Example:
        >>> parsed = parse_policy(policy_text)
        >>> from policyflow.workflow_builder import DynamicWorkflowBuilder
        >>> builder = DynamicWorkflowBuilder(parsed)
        >>> result = builder.run("Check this text")
    """
    config = config or WorkflowConfig()

    # Step 1: Normalize
    normalized = normalize_policy(policy_markdown, config, model)

    # Save intermediate result if requested
    normalized_path = None
    if save_normalized:
        normalized_path = str(save_normalized)
        normalized.save_yaml(save_normalized)

    # Step 2: Generate workflow
    return generate_workflow_from_normalized(
        normalized,
        config,
        model,
        normalized_policy_path=normalized_path,
    )
