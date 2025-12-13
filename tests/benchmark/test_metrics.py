"""Tests for metrics calculator."""

import pytest

from policyflow.benchmark.models import (
    CriterionExpectation,
    ExpectedResult,
    TestCaseResult,
)
from policyflow.models import ClauseResult, EvaluationResult


def make_test_result(
    test_id: str,
    expected_satisfied: bool,
    actual_satisfied: bool,
    expected_criteria: dict[str, bool],
    actual_criteria: dict[str, bool],
    confidence: float = 0.9,
) -> TestCaseResult:
    """Helper to create test results."""
    expected = ExpectedResult(
        policy_satisfied=expected_satisfied,
        criterion_results={
            k: CriterionExpectation(met=v) for k, v in expected_criteria.items()
        },
    )

    actual = EvaluationResult(
        policy_satisfied=actual_satisfied,
        policy_title="Test",
        clause_results=[
            ClauseResult(
                clause_id=k,
                clause_name=k,
                met=v,
                reasoning="R",
                confidence=confidence,
            )
            for k, v in actual_criteria.items()
        ],
        overall_reasoning="Done",
        overall_confidence=confidence,
    )

    return TestCaseResult(
        test_id=test_id,
        expected=expected,
        actual=actual,
        error=None,
        duration_ms=100.0,
    )


class TestSimpleMetricsCalculator:
    """Tests for SimpleMetricsCalculator."""

    def test_calculate_perfect_accuracy(self):
        from policyflow.benchmark.metrics import SimpleMetricsCalculator

        results = [
            make_test_result(
                "t1", True, True, {"c1": True, "c2": True}, {"c1": True, "c2": True}
            ),
            make_test_result(
                "t2", False, False, {"c1": False, "c2": True}, {"c1": False, "c2": True}
            ),
        ]

        calculator = SimpleMetricsCalculator()
        metrics = calculator.calculate(results)

        assert metrics.overall_accuracy == 1.0

    def test_calculate_partial_accuracy(self):
        from policyflow.benchmark.metrics import SimpleMetricsCalculator

        results = [
            make_test_result(
                "t1", True, True, {"c1": True}, {"c1": True}
            ),
            make_test_result(
                "t2", True, False, {"c1": True}, {"c1": True}  # Policy mismatch
            ),
        ]

        calculator = SimpleMetricsCalculator()
        metrics = calculator.calculate(results)

        assert metrics.overall_accuracy == 0.5

    def test_calculate_zero_accuracy(self):
        from policyflow.benchmark.metrics import SimpleMetricsCalculator

        results = [
            make_test_result(
                "t1", True, False, {"c1": True}, {"c1": False}
            ),
            make_test_result(
                "t2", False, True, {"c1": False}, {"c1": True}
            ),
        ]

        calculator = SimpleMetricsCalculator()
        metrics = calculator.calculate(results)

        assert metrics.overall_accuracy == 0.0

    def test_calculate_criterion_metrics(self):
        from policyflow.benchmark.metrics import SimpleMetricsCalculator

        # Create test cases with known outcomes for c1
        # t1: expected c1=True, actual c1=True -> TP
        # t2: expected c1=True, actual c1=False -> FN
        # t3: expected c1=False, actual c1=True -> FP
        # t4: expected c1=False, actual c1=False -> TN
        results = [
            make_test_result("t1", True, True, {"c1": True}, {"c1": True}),
            make_test_result("t2", False, False, {"c1": True}, {"c1": False}),
            make_test_result("t3", True, True, {"c1": False}, {"c1": True}),
            make_test_result("t4", False, False, {"c1": False}, {"c1": False}),
        ]

        calculator = SimpleMetricsCalculator()
        metrics = calculator.calculate(results)

        assert "c1" in metrics.criterion_metrics
        c1_metrics = metrics.criterion_metrics["c1"]

        # TP=1, FN=1, FP=1, TN=1
        assert c1_metrics.confusion.tp == 1
        assert c1_metrics.confusion.tn == 1
        assert c1_metrics.confusion.fp == 1
        assert c1_metrics.confusion.fn == 1

        # Accuracy: (1+1)/4 = 0.5
        assert c1_metrics.accuracy == 0.5
        # Precision: 1/(1+1) = 0.5
        assert c1_metrics.precision == 0.5
        # Recall: 1/(1+1) = 0.5
        assert c1_metrics.recall == 0.5

    def test_calculate_category_accuracy(self):
        from policyflow.benchmark.metrics import SimpleMetricsCalculator

        # Create test results with categories
        results = [
            TestCaseResult(
                test_id="t1",
                expected=ExpectedResult(policy_satisfied=True, criterion_results={}),
                actual=EvaluationResult(
                    policy_satisfied=True,
                    policy_title="T",
                    clause_results=[],
                    overall_reasoning="R",
                    overall_confidence=0.9,
                ),
                error=None,
                duration_ms=100.0,
            ),
            TestCaseResult(
                test_id="t2",
                expected=ExpectedResult(policy_satisfied=True, criterion_results={}),
                actual=EvaluationResult(
                    policy_satisfied=True,
                    policy_title="T",
                    clause_results=[],
                    overall_reasoning="R",
                    overall_confidence=0.9,
                ),
                error=None,
                duration_ms=100.0,
            ),
            TestCaseResult(
                test_id="t3",
                expected=ExpectedResult(policy_satisfied=False, criterion_results={}),
                actual=EvaluationResult(
                    policy_satisfied=True,  # Wrong!
                    policy_title="T",
                    clause_results=[],
                    overall_reasoning="R",
                    overall_confidence=0.9,
                ),
                error=None,
                duration_ms=100.0,
            ),
        ]

        # Provide test case categories
        test_categories = {"t1": "clear_pass", "t2": "clear_pass", "t3": "clear_fail"}

        calculator = SimpleMetricsCalculator()
        metrics = calculator.calculate(results, test_categories=test_categories)

        assert "clear_pass" in metrics.category_accuracy
        assert "clear_fail" in metrics.category_accuracy
        assert metrics.category_accuracy["clear_pass"] == 1.0  # 2/2 correct
        assert metrics.category_accuracy["clear_fail"] == 0.0  # 0/1 correct

    def test_calculate_confidence_calibration(self):
        from policyflow.benchmark.metrics import SimpleMetricsCalculator

        # High confidence (>=0.8): 2 correct, 1 wrong -> 66.6%
        # Medium confidence (0.5-0.8): 1 correct -> 100%
        # Low confidence (<0.5): 1 wrong -> 0%
        results = [
            make_test_result("t1", True, True, {}, {}, confidence=0.9),  # High, correct
            make_test_result("t2", True, True, {}, {}, confidence=0.85),  # High, correct
            make_test_result("t3", True, False, {}, {}, confidence=0.82),  # High, wrong
            make_test_result("t4", True, True, {}, {}, confidence=0.65),  # Medium, correct
            make_test_result("t5", True, False, {}, {}, confidence=0.35),  # Low, wrong
        ]

        calculator = SimpleMetricsCalculator()
        metrics = calculator.calculate(results)

        # Check calibration exists
        cal = metrics.confidence_calibration
        assert cal.high_confidence_accuracy == pytest.approx(2 / 3, rel=0.01)
        assert cal.medium_confidence_accuracy == 1.0
        assert cal.low_confidence_accuracy == 0.0

    def test_handle_error_results(self):
        from policyflow.benchmark.metrics import SimpleMetricsCalculator

        results = [
            make_test_result("t1", True, True, {"c1": True}, {"c1": True}),
            TestCaseResult(
                test_id="t2",
                expected=ExpectedResult(
                    policy_satisfied=True,
                    criterion_results={"c1": CriterionExpectation(met=True)},
                ),
                actual=None,  # Error case
                error="Connection failed",
                duration_ms=50.0,
            ),
        ]

        calculator = SimpleMetricsCalculator()
        metrics = calculator.calculate(results)

        # Error cases should count as failures
        assert metrics.overall_accuracy == 0.5


class TestMetricsEdgeCases:
    """Test edge cases for metrics calculation."""

    def test_empty_results(self):
        from policyflow.benchmark.metrics import SimpleMetricsCalculator

        calculator = SimpleMetricsCalculator()
        metrics = calculator.calculate([])

        assert metrics.overall_accuracy == 0.0
        assert len(metrics.criterion_metrics) == 0

    def test_all_errors(self):
        from policyflow.benchmark.metrics import SimpleMetricsCalculator

        results = [
            TestCaseResult(
                test_id="t1",
                expected=ExpectedResult(policy_satisfied=True, criterion_results={}),
                actual=None,
                error="Error 1",
                duration_ms=50.0,
            ),
            TestCaseResult(
                test_id="t2",
                expected=ExpectedResult(policy_satisfied=True, criterion_results={}),
                actual=None,
                error="Error 2",
                duration_ms=50.0,
            ),
        ]

        calculator = SimpleMetricsCalculator()
        metrics = calculator.calculate(results)

        assert metrics.overall_accuracy == 0.0
