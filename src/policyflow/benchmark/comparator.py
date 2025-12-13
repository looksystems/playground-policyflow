"""Result comparator for comparing actual vs expected evaluation results."""

from __future__ import annotations

from policyflow.benchmark.models import CriterionExpectation, ExpectedResult
from policyflow.benchmark.protocols import ComparisonResult
from policyflow.models import ClauseResult, EvaluationResult


def _build_clause_result_map(
    clause_results: list[ClauseResult],
) -> dict[str, ClauseResult]:
    """Build a flat map of clause_id -> ClauseResult including nested results.

    Args:
        clause_results: List of clause results (may have sub_results)

    Returns:
        Flat dictionary mapping clause_id to ClauseResult
    """
    result_map: dict[str, ClauseResult] = {}

    def add_results(results: list[ClauseResult]) -> None:
        for result in results:
            result_map[result.clause_id] = result
            if result.sub_results:
                add_results(result.sub_results)

    add_results(clause_results)
    return result_map


class SimpleResultComparator:
    """Simple comparator for evaluation results.

    Performs deep comparison of actual vs expected results,
    including nested criterion and sub-criterion results.
    """

    def compare(
        self,
        actual: EvaluationResult,
        expected: ExpectedResult,
    ) -> ComparisonResult:
        """Compare actual evaluation result against expected.

        Args:
            actual: The actual result from workflow execution
            expected: The expected result from the golden dataset

        Returns:
            Detailed comparison result
        """
        # Check policy_satisfied match
        policy_match = actual.policy_satisfied == expected.policy_satisfied

        # Build map of actual clause results
        actual_map = _build_clause_result_map(actual.clause_results)

        # Compare each expected criterion
        criterion_matches: dict[str, bool] = {}
        mismatched: list[str] = []
        details: dict[str, str] = {}

        self._compare_criteria(
            expected.criterion_results,
            actual_map,
            criterion_matches,
            mismatched,
            details,
        )

        # Overall match requires both policy_satisfied and all criteria to match
        overall_match = policy_match and len(mismatched) == 0

        return ComparisonResult(
            matches=overall_match,
            policy_satisfied_match=policy_match,
            criterion_matches=criterion_matches,
            mismatched_criteria=mismatched,
            details=details,
        )

    def _compare_criteria(
        self,
        expected_criteria: dict[str, CriterionExpectation],
        actual_map: dict[str, ClauseResult],
        criterion_matches: dict[str, bool],
        mismatched: list[str],
        details: dict[str, str],
    ) -> None:
        """Recursively compare criteria including sub-results.

        Args:
            expected_criteria: Expected criterion results
            actual_map: Flat map of actual clause results
            criterion_matches: Output dict for match status
            mismatched: Output list for mismatched criteria
            details: Output dict for mismatch details
        """
        for criterion_id, expected in expected_criteria.items():
            actual_result = actual_map.get(criterion_id)

            if actual_result is None:
                # Criterion missing in actual results
                criterion_matches[criterion_id] = False
                mismatched.append(criterion_id)
                details[criterion_id] = f"Expected {expected.met}, but criterion not found in actual"
                continue

            # Compare the met status
            matches = actual_result.met == expected.met
            criterion_matches[criterion_id] = matches

            if not matches:
                mismatched.append(criterion_id)
                details[criterion_id] = (
                    f"Expected met={expected.met}, got met={actual_result.met}"
                )

            # Recursively compare sub-results if present
            if expected.sub_results:
                self._compare_criteria(
                    expected.sub_results,
                    actual_map,
                    criterion_matches,
                    mismatched,
                    details,
                )
