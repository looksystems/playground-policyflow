"""Hypothesis generator for workflow improvement.

Provides both template-based and LLM-powered hypothesis generation.
"""

from __future__ import annotations

import re
import uuid
from typing import Literal

from policyflow.benchmark.models import (
    AnalysisReport,
    FailurePattern,
    Hypothesis,
)
from policyflow.models import ParsedWorkflowPolicy


# Template hypotheses for common failure patterns
HYPOTHESIS_TEMPLATES = {
    "criterion_systematic": [
        {
            "change_type": "prompt_tuning",
            "description": "Clarify prompt for {criterion}",
            "suggested_change": {"action": "revise_prompt", "criterion": "{criterion}"},
            "rationale": "Systematic failures often indicate unclear or ambiguous prompts",
            "expected_impact": "Reduce systematic false negatives/positives",
        },
        {
            "change_type": "node_param",
            "description": "Adjust confidence threshold for {criterion}",
            "suggested_change": {"confidence_threshold": 0.6, "criterion": "{criterion}"},
            "rationale": "Lower threshold may capture more edge cases",
            "expected_impact": "Improve recall at slight precision cost",
        },
    ],
    "false_positive_heavy": [
        {
            "change_type": "threshold",
            "description": "Increase confidence threshold to reduce false positives",
            "suggested_change": {"confidence_threshold": 0.85},
            "rationale": "Higher threshold filters out uncertain positive predictions",
            "expected_impact": "Reduce false positives, may increase false negatives",
        },
        {
            "change_type": "workflow_structure",
            "description": "Add confidence gate after initial classification",
            "suggested_change": {
                "add_node": {
                    "type": "ConfidenceGateNode",
                    "params": {"high_threshold": 0.85, "low_threshold": 0.6},
                }
            },
            "rationale": "Route uncertain predictions to secondary verification",
            "expected_impact": "Improve precision while maintaining recall",
        },
    ],
    "false_negative_heavy": [
        {
            "change_type": "threshold",
            "description": "Lower confidence threshold to reduce false negatives",
            "suggested_change": {"confidence_threshold": 0.5},
            "rationale": "Lower threshold captures more positive cases",
            "expected_impact": "Reduce false negatives, may increase false positives",
        },
        {
            "change_type": "prompt_tuning",
            "description": "Broaden pattern matching in prompts",
            "suggested_change": {"action": "broaden_patterns"},
            "rationale": "False negatives may indicate overly strict matching",
            "expected_impact": "Improve recall",
        },
    ],
    "confidence_miscalibration": [
        {
            "change_type": "threshold",
            "description": "Recalibrate confidence thresholds",
            "suggested_change": {
                "high_threshold": 0.9,
                "low_threshold": 0.6,
            },
            "rationale": "Confidence scores don't match actual accuracy",
            "expected_impact": "Better alignment between confidence and correctness",
        },
        {
            "change_type": "workflow_structure",
            "description": "Add calibration node for confidence adjustment",
            "suggested_change": {
                "add_node": {
                    "type": "ConfidenceCalibrationNode",
                    "params": {"method": "platt_scaling"},
                }
            },
            "rationale": "Post-hoc calibration can improve confidence reliability",
            "expected_impact": "More reliable confidence scores",
        },
    ],
    "category_cluster": [
        {
            "change_type": "prompt_tuning",
            "description": "Add specific examples for {category} category",
            "suggested_change": {
                "action": "add_examples",
                "category": "{category}",
            },
            "rationale": "Specific examples help with edge cases in this category",
            "expected_impact": "Improve accuracy on {category} cases",
        },
    ],
}


class TemplateBasedHypothesisGenerator:
    """Generates hypotheses using predefined templates.

    Maps failure patterns to appropriate hypothesis templates and
    instantiates them with pattern-specific details.
    """

    def __init__(self):
        """Initialize the template-based generator."""
        self.templates = HYPOTHESIS_TEMPLATES

    def generate(
        self,
        analysis: AnalysisReport,
        workflow: ParsedWorkflowPolicy,
    ) -> list[Hypothesis]:
        """Generate hypotheses for workflow improvement.

        Args:
            analysis: Analysis report with identified patterns
            workflow: The current workflow

        Returns:
            List of improvement hypotheses
        """
        hypotheses: list[Hypothesis] = []
        seen_ids: set[str] = set()

        for pattern in analysis.patterns:
            pattern_hypotheses = self._generate_for_pattern(pattern)
            for h in pattern_hypotheses:
                if h.id not in seen_ids:
                    hypotheses.append(h)
                    seen_ids.add(h.id)

        return hypotheses

    def _generate_for_pattern(
        self, pattern: FailurePattern
    ) -> list[Hypothesis]:
        """Generate hypotheses for a single failure pattern."""
        templates = self.templates.get(pattern.pattern_type, [])
        hypotheses = []

        # Extract variables from pattern description
        variables = self._extract_variables(pattern)

        for i, template in enumerate(templates):
            hypothesis = self._instantiate_template(template, variables, pattern, i)
            hypotheses.append(hypothesis)

        return hypotheses

    def _extract_variables(self, pattern: FailurePattern) -> dict[str, str]:
        """Extract variables from pattern for template substitution.

        Prefers structured metadata over regex parsing of description.
        Falls back to regex for backward compatibility with patterns that
        don't have metadata populated.
        """
        variables = {}

        # Prefer metadata if available (structured, reliable)
        if pattern.metadata:
            if "criterion" in pattern.metadata:
                variables["criterion"] = pattern.metadata["criterion"]
            if "category" in pattern.metadata:
                variables["category"] = pattern.metadata["category"]
            # Return early if we got what we need from metadata
            if variables:
                return variables

        # Fallback to regex parsing for backward compatibility
        # Extract criterion name from description
        criterion_match = re.search(r"[Cc]riterion['\s]+['\"]?([^'\"]+)['\"]?", pattern.description)
        if criterion_match:
            variables["criterion"] = criterion_match.group(1).strip()

        # Extract category name from description
        category_match = re.search(r"['\"]([^'\"]+)['\"].*category", pattern.description)
        if category_match:
            variables["category"] = category_match.group(1)

        return variables

    def _instantiate_template(
        self,
        template: dict,
        variables: dict[str, str],
        pattern: FailurePattern,
        index: int,
    ) -> Hypothesis:
        """Create a Hypothesis from a template with variable substitution."""
        # Generate unique ID
        hyp_id = f"hyp_{pattern.pattern_type}_{index}_{uuid.uuid4().hex[:6]}"

        # Safe format that keeps placeholders if variable is missing
        def safe_format(s: str, vars: dict[str, str]) -> str:
            """Format string, keeping placeholders for missing variables."""
            if not vars:
                # Remove placeholders if no variables
                return re.sub(r"\{[^}]+\}", "target", s)
            try:
                return s.format(**vars)
            except KeyError:
                # Keep original if substitution fails
                return re.sub(r"\{[^}]+\}", "target", s)

        # Substitute variables in strings
        description = safe_format(template["description"], variables)
        rationale = safe_format(template["rationale"], variables)
        expected_impact = safe_format(template["expected_impact"], variables)

        # Handle suggested_change substitution
        suggested_change = {}
        for k, v in template["suggested_change"].items():
            if isinstance(v, str) and "{" in v:
                suggested_change[k] = safe_format(v, variables)
            else:
                suggested_change[k] = v

        # Determine target
        target = variables.get("criterion", variables.get("category", "workflow"))

        return Hypothesis(
            id=hyp_id,
            description=description,
            change_type=template["change_type"],
            target=target,
            suggested_change=suggested_change,
            rationale=rationale,
            expected_impact=expected_impact,
        )


class LLMHypothesisGenerator:
    """LLM-powered hypothesis generator.

    Uses LLM to generate creative hypotheses based on failure analysis.
    Can optionally fall back to template-based generation when LLM is not configured.
    """

    def __init__(
        self,
        model: str | None = None,
        template_based: TemplateBasedHypothesisGenerator | None = None,
    ):
        """Initialize with optional LLM model and template-based fallback.

        Args:
            model: LLM model identifier (e.g., "gpt-4", "claude-3-opus")
            template_based: Template generator for fallback
        """
        self.model = model
        self.template_based = template_based or TemplateBasedHypothesisGenerator()

    def generate(
        self,
        analysis: AnalysisReport,
        workflow: ParsedWorkflowPolicy,
    ) -> list[Hypothesis]:
        """Generate hypotheses using LLM, with template fallback.

        If no model is configured, falls back to template-based generation.
        """
        # Fall back to template if no model configured
        if not self.model:
            return self.template_based.generate(analysis, workflow)

        try:
            return self._generate_with_llm(analysis, workflow)
        except Exception:
            # Fall back to template on LLM errors
            return self.template_based.generate(analysis, workflow)

    def _generate_with_llm(
        self,
        analysis: AnalysisReport,
        workflow: ParsedWorkflowPolicy,
    ) -> list[Hypothesis]:
        """Generate hypotheses using LLM."""
        from policyflow.llm import call_llm

        # Build prompt for LLM
        patterns_text = "\n".join(
            f"- {p.pattern_type}: {p.description} (severity: {p.severity})"
            for p in analysis.patterns
        )

        criteria_text = "\n".join(
            f"- {c.criterion_id}: failure_rate={c.failure_rate:.1%}, "
            f"FP={c.false_positive_rate:.1%}, FN={c.false_negative_rate:.1%}"
            for c in analysis.problematic_criteria
        )

        prompt = f"""Analyze these workflow benchmark failures and suggest improvement hypotheses.

FAILURE PATTERNS:
{patterns_text or "None detected"}

PROBLEMATIC CRITERIA:
{criteria_text or "None detected"}

RECOMMENDATIONS FROM ANALYSIS:
{chr(10).join(f'- {r}' for r in analysis.recommendations) or "None"}

WORKFLOW TITLE: {workflow.title}
WORKFLOW NODES: {[n.id for n in workflow.workflow.nodes]}

Generate 3-5 specific, actionable hypotheses for improving the workflow. For each hypothesis, suggest a concrete change that could address the identified issues.

Respond in YAML format with this structure:
```yaml
hypotheses:
  - id: hyp_001
    description: Brief description of the change
    change_type: one of [prompt_tuning, threshold, node_param, workflow_structure]
    target: target node id or criterion
    suggested_change:
      key: value
    rationale: Why this change should help
    expected_impact: Expected improvement
```"""

        system_prompt = """You are an expert at analyzing ML/AI workflow failures and suggesting improvements.
Focus on practical, actionable changes that can be tested empirically.
Use specific node IDs and parameter names from the workflow when possible."""

        result = call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.model,
            yaml_response=True,
            span_name="hypothesis_generation",
        )

        return self._parse_hypotheses(result)

    def _parse_hypotheses(self, result: dict) -> list[Hypothesis]:
        """Parse LLM response into Hypothesis objects."""
        hypotheses = []

        if not isinstance(result, dict) or "hypotheses" not in result:
            return hypotheses

        for h in result.get("hypotheses", []):
            try:
                hypotheses.append(
                    Hypothesis(
                        id=h.get("id", f"hyp_llm_{uuid.uuid4().hex[:6]}"),
                        description=h.get("description", "LLM-generated hypothesis"),
                        change_type=h.get("change_type", "prompt_tuning"),
                        target=h.get("target", "workflow"),
                        suggested_change=h.get("suggested_change", {}),
                        rationale=h.get("rationale", "Generated by LLM"),
                        expected_impact=h.get("expected_impact", "Unknown"),
                    )
                )
            except Exception:
                # Skip malformed hypotheses
                continue

        return hypotheses


def create_hypothesis_generator(
    mode: Literal["template", "llm", "hybrid"] = "hybrid",
    model: str | None = None,
) -> TemplateBasedHypothesisGenerator | LLMHypothesisGenerator:
    """Factory function to create a hypothesis generator.

    Args:
        mode: Generation mode
            - "template": Only use predefined templates
            - "llm": Use LLM for generation (requires model)
            - "hybrid": Combine template and LLM approaches
        model: LLM model identifier for "llm" and "hybrid" modes

    Returns:
        Configured generator instance

    Raises:
        ValueError: If mode is invalid
    """
    if mode == "template":
        return TemplateBasedHypothesisGenerator()
    elif mode in ("llm", "hybrid"):
        return LLMHypothesisGenerator(model=model)
    else:
        raise ValueError(f"Invalid hypothesis generator mode: {mode}")
