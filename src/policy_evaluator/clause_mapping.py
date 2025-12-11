"""Utilities for mapping between clauses and workflow results.

This module provides tools for extracting clause-level evaluation results
from workflow execution, enabling explainability and audit trails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Clause, NormalizedPolicy
from .numbering import clause_sort_key, node_id_to_clause_number


@dataclass
class ClauseResult:
    """Evaluation result for a specific clause."""

    clause_number: str
    clause_text: str
    node_id: str
    satisfied: bool
    confidence: float
    reasoning: str
    sub_results: list[ClauseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "clause_number": self.clause_number,
            "clause_text": self.clause_text,
            "node_id": self.node_id,
            "satisfied": self.satisfied,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "sub_results": [sr.to_dict() for sr in self.sub_results],
        }


def extract_clause_results(
    shared: dict[str, Any],
    normalized: NormalizedPolicy,
) -> list[ClauseResult]:
    """Extract clause-level results from workflow shared store.

    Maps workflow node results back to clause numbers for explainability.

    Args:
        shared: The workflow shared store after execution
        normalized: The normalized policy for clause text lookup

    Returns:
        List of ClauseResult objects (flat list, sorted by clause number)

    Example:
        >>> results = extract_clause_results(shared, normalized_policy)
        >>> for r in results:
        ...     print(f"{r.clause_number}: {'PASS' if r.satisfied else 'FAIL'}")
    """
    results: list[ClauseResult] = []

    # Look for results stored with clause_ prefix in keys
    for key, value in shared.items():
        if not isinstance(value, dict):
            continue

        # Try to extract clause number from key
        # Common patterns: clause_1_1_result, clause_1_1, etc.
        node_id = key.replace("_result", "")
        clause_number = node_id_to_clause_number(node_id)

        if not clause_number:
            continue

        # Get clause text from normalized policy
        clause = normalized.get_clause_by_number(clause_number)
        clause_text = clause.text if clause else ""

        # Extract result fields (handle different result formats)
        satisfied = _extract_satisfied(value)
        confidence = _extract_confidence(value)
        reasoning = _extract_reasoning(value)

        results.append(ClauseResult(
            clause_number=clause_number,
            clause_text=clause_text,
            node_id=node_id,
            satisfied=satisfied,
            confidence=confidence,
            reasoning=reasoning,
            sub_results=[],  # Will be populated by build_hierarchy if needed
        ))

    # Sort by clause number for consistent ordering
    results.sort(key=lambda r: clause_sort_key(r.clause_number))

    return results


def _extract_satisfied(value: dict[str, Any]) -> bool:
    """Extract satisfied/met status from result dict."""
    # Try common field names
    for field_name in ["satisfied", "met", "passed", "matches", "result"]:
        if field_name in value:
            v = value[field_name]
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() in ("true", "yes", "satisfied", "met", "passed")
    return False


def _extract_confidence(value: dict[str, Any]) -> float:
    """Extract confidence score from result dict."""
    for field_name in ["confidence", "score", "certainty"]:
        if field_name in value:
            v = value[field_name]
            if isinstance(v, (int, float)):
                return float(v)
    return 0.5  # Default confidence


def _extract_reasoning(value: dict[str, Any]) -> str:
    """Extract reasoning/explanation from result dict."""
    for field_name in ["reasoning", "reason", "explanation", "rationale"]:
        if field_name in value:
            v = value[field_name]
            if isinstance(v, str):
                return v
    return ""


def build_hierarchical_results(
    flat_results: list[ClauseResult],
    normalized: NormalizedPolicy,
) -> list[ClauseResult]:
    """Build hierarchical result structure matching document structure.

    Takes flat results and nests them according to clause hierarchy.

    Args:
        flat_results: Flat list of ClauseResult objects
        normalized: The normalized policy for structure reference

    Returns:
        List of top-level ClauseResult with nested sub_results
    """
    # Create lookup by clause number
    results_by_number = {r.clause_number: r for r in flat_results}

    # Build hierarchy based on normalized policy structure
    hierarchical: list[ClauseResult] = []

    def process_clause(clause: Clause, parent_result: ClauseResult | None = None) -> ClauseResult | None:
        result = results_by_number.get(clause.number)

        if result is None:
            # No result for this clause - create placeholder
            result = ClauseResult(
                clause_number=clause.number,
                clause_text=clause.text,
                node_id=clause.node_id,
                satisfied=False,
                confidence=0.0,
                reasoning="No evaluation result found",
                sub_results=[],
            )

        # Process sub-clauses
        for sub_clause in clause.sub_clauses:
            sub_result = process_clause(sub_clause, result)
            if sub_result:
                result.sub_results.append(sub_result)

        return result

    # Process all sections and clauses
    for section in normalized.sections:
        for clause in section.clauses:
            result = process_clause(clause)
            if result:
                hierarchical.append(result)

    return hierarchical


def format_clause_results_report(
    results: list[ClauseResult],
    indent: int = 0,
    show_reasoning: bool = True,
) -> str:
    """Format clause results as a human-readable report.

    Args:
        results: List of ClauseResult objects
        indent: Current indentation level
        show_reasoning: Whether to include reasoning in output

    Returns:
        Formatted string report

    Example:
        >>> report = format_clause_results_report(results)
        >>> print(report)
        [+] Clause 1.1: PASS (92% confidence)
            Reasoning: Content addresses investor directly
          [+] Clause 1.1.a: PASS (95% confidence)
          [-] Clause 1.1.b: FAIL (88% confidence)
    """
    lines: list[str] = []
    prefix = "  " * indent

    for result in results:
        status = "PASS" if result.satisfied else "FAIL"
        status_icon = "[+]" if result.satisfied else "[-]"

        lines.append(
            f"{prefix}{status_icon} Clause {result.clause_number}: {status} "
            f"({result.confidence:.0%} confidence)"
        )

        if show_reasoning and result.reasoning:
            # Truncate long reasoning
            reasoning = result.reasoning
            if len(reasoning) > 100:
                reasoning = reasoning[:97] + "..."
            lines.append(f"{prefix}    Reasoning: {reasoning}")

        if result.sub_results:
            sub_report = format_clause_results_report(
                result.sub_results,
                indent + 1,
                show_reasoning,
            )
            lines.append(sub_report)

    return "\n".join(lines)


def summarize_results(results: list[ClauseResult]) -> dict[str, Any]:
    """Generate summary statistics from clause results.

    Args:
        results: List of ClauseResult objects

    Returns:
        Dictionary with summary statistics

    Example:
        >>> summary = summarize_results(results)
        >>> print(summary)
        {'total': 5, 'passed': 4, 'failed': 1, 'pass_rate': 0.8, ...}
    """
    all_results = _flatten_results(results)

    total = len(all_results)
    passed = sum(1 for r in all_results if r.satisfied)
    failed = total - passed

    confidences = [r.confidence for r in all_results]
    avg_confidence = sum(confidences) / total if total > 0 else 0.0
    min_confidence = min(confidences) if confidences else 0.0

    low_confidence = [r.clause_number for r in all_results if r.confidence < 0.7]

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total > 0 else 0.0,
        "average_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "low_confidence_clauses": low_confidence,
        "failed_clauses": [r.clause_number for r in all_results if not r.satisfied],
    }


def _flatten_results(results: list[ClauseResult]) -> list[ClauseResult]:
    """Flatten hierarchical results into a single list."""
    flat: list[ClauseResult] = []

    def collect(result_list: list[ClauseResult]) -> None:
        for r in result_list:
            flat.append(r)
            if r.sub_results:
                collect(r.sub_results)

    collect(results)
    return flat
