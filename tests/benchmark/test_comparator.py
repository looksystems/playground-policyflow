"""Tests for result comparator."""

import pytest

from policyflow.benchmark.models import CriterionExpectation, ExpectedResult
from policyflow.models import ClauseResult, EvaluationResult


class TestSimpleResultComparator:
    """Tests for SimpleResultComparator."""

    def test_compare_matching_results(self):
        from policyflow.benchmark.comparator import SimpleResultComparator

        expected = ExpectedResult(
            policy_satisfied=True,
            criterion_results={
                "criterion_1": CriterionExpectation(met=True),
                "criterion_2": CriterionExpectation(met=True),
            },
        )

        actual = EvaluationResult(
            policy_satisfied=True,
            policy_title="Test",
            clause_results=[
                ClauseResult(
                    clause_id="criterion_1",
                    clause_name="C1",
                    met=True,
                    reasoning="R1",
                    confidence=0.9,
                ),
                ClauseResult(
                    clause_id="criterion_2",
                    clause_name="C2",
                    met=True,
                    reasoning="R2",
                    confidence=0.85,
                ),
            ],
            overall_reasoning="Pass",
            overall_confidence=0.87,
        )

        comparator = SimpleResultComparator()
        result = comparator.compare(actual, expected)

        assert result.matches is True
        assert result.policy_satisfied_match is True
        assert len(result.mismatched_criteria) == 0

    def test_compare_policy_satisfied_mismatch(self):
        from policyflow.benchmark.comparator import SimpleResultComparator

        expected = ExpectedResult(
            policy_satisfied=True,
            criterion_results={"c1": CriterionExpectation(met=True)},
        )

        actual = EvaluationResult(
            policy_satisfied=False,
            policy_title="Test",
            clause_results=[
                ClauseResult(
                    clause_id="c1",
                    clause_name="C1",
                    met=True,
                    reasoning="R",
                    confidence=0.9,
                )
            ],
            overall_reasoning="Fail",
            overall_confidence=0.9,
        )

        comparator = SimpleResultComparator()
        result = comparator.compare(actual, expected)

        assert result.matches is False
        assert result.policy_satisfied_match is False

    def test_compare_criterion_mismatch(self):
        from policyflow.benchmark.comparator import SimpleResultComparator

        expected = ExpectedResult(
            policy_satisfied=True,
            criterion_results={
                "criterion_1": CriterionExpectation(met=True),
                "criterion_2": CriterionExpectation(met=True),
            },
        )

        actual = EvaluationResult(
            policy_satisfied=True,
            policy_title="Test",
            clause_results=[
                ClauseResult(
                    clause_id="criterion_1",
                    clause_name="C1",
                    met=True,
                    reasoning="R1",
                    confidence=0.9,
                ),
                ClauseResult(
                    clause_id="criterion_2",
                    clause_name="C2",
                    met=False,  # Mismatch!
                    reasoning="R2",
                    confidence=0.85,
                ),
            ],
            overall_reasoning="Pass",
            overall_confidence=0.87,
        )

        comparator = SimpleResultComparator()
        result = comparator.compare(actual, expected)

        assert result.matches is False
        assert result.policy_satisfied_match is True
        assert "criterion_2" in result.mismatched_criteria
        assert result.criterion_matches["criterion_1"] is True
        assert result.criterion_matches["criterion_2"] is False

    def test_compare_with_sub_results(self):
        from policyflow.benchmark.comparator import SimpleResultComparator

        expected = ExpectedResult(
            policy_satisfied=True,
            criterion_results={
                "criterion_1": CriterionExpectation(
                    met=True,
                    sub_results={
                        "criterion_1a": CriterionExpectation(met=True),
                        "criterion_1b": CriterionExpectation(met=False),
                    },
                ),
            },
        )

        actual = EvaluationResult(
            policy_satisfied=True,
            policy_title="Test",
            clause_results=[
                ClauseResult(
                    clause_id="criterion_1",
                    clause_name="C1",
                    met=True,
                    reasoning="R1",
                    confidence=0.9,
                    sub_results=[
                        ClauseResult(
                            clause_id="criterion_1a",
                            clause_name="C1a",
                            met=True,
                            reasoning="Ra",
                            confidence=0.9,
                        ),
                        ClauseResult(
                            clause_id="criterion_1b",
                            clause_name="C1b",
                            met=False,
                            reasoning="Rb",
                            confidence=0.8,
                        ),
                    ],
                ),
            ],
            overall_reasoning="Pass",
            overall_confidence=0.85,
        )

        comparator = SimpleResultComparator()
        result = comparator.compare(actual, expected)

        assert result.matches is True
        assert "criterion_1a" in result.criterion_matches
        assert "criterion_1b" in result.criterion_matches

    def test_compare_sub_result_mismatch(self):
        from policyflow.benchmark.comparator import SimpleResultComparator

        expected = ExpectedResult(
            policy_satisfied=True,
            criterion_results={
                "criterion_1": CriterionExpectation(
                    met=True,
                    sub_results={
                        "criterion_1a": CriterionExpectation(met=True),
                        "criterion_1b": CriterionExpectation(met=True),  # Expected True
                    },
                ),
            },
        )

        actual = EvaluationResult(
            policy_satisfied=True,
            policy_title="Test",
            clause_results=[
                ClauseResult(
                    clause_id="criterion_1",
                    clause_name="C1",
                    met=True,
                    reasoning="R1",
                    confidence=0.9,
                    sub_results=[
                        ClauseResult(
                            clause_id="criterion_1a",
                            clause_name="C1a",
                            met=True,
                            reasoning="Ra",
                            confidence=0.9,
                        ),
                        ClauseResult(
                            clause_id="criterion_1b",
                            clause_name="C1b",
                            met=False,  # Actual False - mismatch!
                            reasoning="Rb",
                            confidence=0.8,
                        ),
                    ],
                ),
            ],
            overall_reasoning="Pass",
            overall_confidence=0.85,
        )

        comparator = SimpleResultComparator()
        result = comparator.compare(actual, expected)

        assert result.matches is False
        assert "criterion_1b" in result.mismatched_criteria

    def test_compare_missing_criterion_in_actual(self):
        from policyflow.benchmark.comparator import SimpleResultComparator

        expected = ExpectedResult(
            policy_satisfied=True,
            criterion_results={
                "criterion_1": CriterionExpectation(met=True),
                "criterion_2": CriterionExpectation(met=True),
            },
        )

        actual = EvaluationResult(
            policy_satisfied=True,
            policy_title="Test",
            clause_results=[
                ClauseResult(
                    clause_id="criterion_1",
                    clause_name="C1",
                    met=True,
                    reasoning="R1",
                    confidence=0.9,
                ),
                # criterion_2 is missing!
            ],
            overall_reasoning="Pass",
            overall_confidence=0.9,
        )

        comparator = SimpleResultComparator()
        result = comparator.compare(actual, expected)

        # Missing criterion should be flagged
        assert result.matches is False
        assert "criterion_2" in result.mismatched_criteria


class TestCompareHelpers:
    """Tests for comparison helper functions."""

    def test_build_clause_result_map(self):
        from policyflow.benchmark.comparator import _build_clause_result_map

        clause_results = [
            ClauseResult(
                clause_id="c1",
                clause_name="C1",
                met=True,
                reasoning="R",
                confidence=0.9,
            ),
            ClauseResult(
                clause_id="c2",
                clause_name="C2",
                met=False,
                reasoning="R",
                confidence=0.8,
                sub_results=[
                    ClauseResult(
                        clause_id="c2a",
                        clause_name="C2a",
                        met=True,
                        reasoning="R",
                        confidence=0.85,
                    ),
                ],
            ),
        ]

        result_map = _build_clause_result_map(clause_results)

        assert "c1" in result_map
        assert "c2" in result_map
        assert "c2a" in result_map
        assert result_map["c1"].met is True
        assert result_map["c2a"].met is True
