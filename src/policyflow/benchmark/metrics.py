"""Metrics calculator for benchmark results."""

from __future__ import annotations

from collections import defaultdict

from policyflow.benchmark.models import (
    BenchmarkMetrics,
    ConfidenceCalibration,
    ConfusionMatrix,
    CriterionExpectation,
    CriterionMetrics,
    TestCaseResult,
)


class SimpleMetricsCalculator:
    """Simple metrics calculator for benchmark results.

    Computes:
    - Overall accuracy (policy_satisfied matches)
    - Per-criterion precision/recall/F1
    - Category-wise accuracy
    - Confidence calibration
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.8,
        low_confidence_threshold: float = 0.5,
    ):
        """Initialize the calculator.

        Args:
            high_confidence_threshold: Threshold for high confidence (>=)
            low_confidence_threshold: Threshold for low confidence (<)
        """
        self.high_threshold = high_confidence_threshold
        self.low_threshold = low_confidence_threshold

    def calculate(
        self,
        results: list[TestCaseResult],
        test_categories: dict[str, str] | None = None,
    ) -> BenchmarkMetrics:
        """Calculate aggregate metrics from test results.

        Args:
            results: List of individual test case results
            test_categories: Optional mapping of test_id -> category

        Returns:
            Aggregate benchmark metrics
        """
        if not results:
            return self._empty_metrics()

        # Calculate overall accuracy
        overall_accuracy = self._calculate_overall_accuracy(results)

        # Calculate per-criterion metrics
        criterion_metrics = self._calculate_criterion_metrics(results)

        # Calculate category accuracy
        category_accuracy = self._calculate_category_accuracy(results, test_categories)

        # Calculate confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(results)

        return BenchmarkMetrics(
            overall_accuracy=overall_accuracy,
            criterion_metrics=criterion_metrics,
            category_accuracy=category_accuracy,
            confidence_calibration=confidence_calibration,
        )

    def _empty_metrics(self) -> BenchmarkMetrics:
        """Return empty metrics for no results."""
        return BenchmarkMetrics(
            overall_accuracy=0.0,
            criterion_metrics={},
            category_accuracy={},
            confidence_calibration=ConfidenceCalibration(
                high_confidence_accuracy=0.0,
                medium_confidence_accuracy=0.0,
                low_confidence_accuracy=0.0,
            ),
        )

    def _calculate_overall_accuracy(self, results: list[TestCaseResult]) -> float:
        """Calculate overall accuracy based on policy_satisfied matches."""
        if not results:
            return 0.0

        correct = sum(1 for r in results if r.passed)
        return correct / len(results)

    def _calculate_criterion_metrics(
        self, results: list[TestCaseResult]
    ) -> dict[str, CriterionMetrics]:
        """Calculate per-criterion metrics with confusion matrices."""
        # Collect all criterion IDs
        criterion_ids: set[str] = set()
        for result in results:
            criterion_ids.update(result.expected.criterion_results.keys())
            self._collect_sub_criterion_ids(
                result.expected.criterion_results, criterion_ids
            )

        # Build confusion matrix for each criterion
        metrics: dict[str, CriterionMetrics] = {}
        for crit_id in criterion_ids:
            confusion = self._build_confusion_matrix(results, crit_id)
            metrics[crit_id] = CriterionMetrics(
                accuracy=confusion.accuracy,
                precision=confusion.precision,
                recall=confusion.recall,
                f1=confusion.f1,
                confusion=confusion,
            )

        return metrics

    def _collect_sub_criterion_ids(
        self,
        criterion_results: dict[str, CriterionExpectation],
        ids: set[str],
    ) -> None:
        """Recursively collect all criterion IDs including sub-results."""
        for crit_id, expectation in criterion_results.items():
            ids.add(crit_id)
            if expectation.sub_results:
                self._collect_sub_criterion_ids(expectation.sub_results, ids)

    def _build_confusion_matrix(
        self, results: list[TestCaseResult], criterion_id: str
    ) -> ConfusionMatrix:
        """Build confusion matrix for a specific criterion."""
        tp = tn = fp = fn = 0

        for result in results:
            expected_met = self._get_expected_met(result, criterion_id)
            actual_met = self._get_actual_met(result, criterion_id)

            if expected_met is None:
                # Criterion not in this test case
                continue

            if actual_met is None:
                # Error or missing - count as wrong
                if expected_met:
                    fn += 1
                else:
                    fp += 1
                continue

            # Standard confusion matrix logic
            if expected_met and actual_met:
                tp += 1
            elif not expected_met and not actual_met:
                tn += 1
            elif not expected_met and actual_met:
                fp += 1
            else:  # expected_met and not actual_met
                fn += 1

        return ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)

    def _get_expected_met(
        self, result: TestCaseResult, criterion_id: str
    ) -> bool | None:
        """Get expected met status for a criterion."""

        def find_in_expectations(
            expectations: dict[str, CriterionExpectation],
        ) -> bool | None:
            if criterion_id in expectations:
                return expectations[criterion_id].met

            for exp in expectations.values():
                if exp.sub_results:
                    found = find_in_expectations(exp.sub_results)
                    if found is not None:
                        return found
            return None

        return find_in_expectations(result.expected.criterion_results)

    def _get_actual_met(
        self, result: TestCaseResult, criterion_id: str
    ) -> bool | None:
        """Get actual met status for a criterion."""
        if result.actual is None:
            return None

        def find_in_results(clause_results: list) -> bool | None:
            for clause in clause_results:
                if clause.clause_id == criterion_id:
                    return clause.met
                if clause.sub_results:
                    found = find_in_results(clause.sub_results)
                    if found is not None:
                        return found
            return None

        return find_in_results(result.actual.clause_results)

    def _calculate_category_accuracy(
        self,
        results: list[TestCaseResult],
        test_categories: dict[str, str] | None,
    ) -> dict[str, float]:
        """Calculate accuracy per test category."""
        if not test_categories:
            return {}

        # Group results by category
        by_category: dict[str, list[TestCaseResult]] = defaultdict(list)
        for result in results:
            category = test_categories.get(result.test_id)
            if category:
                by_category[category].append(result)

        # Calculate accuracy per category
        category_accuracy: dict[str, float] = {}
        for category, cat_results in by_category.items():
            if cat_results:
                correct = sum(1 for r in cat_results if r.passed)
                category_accuracy[category] = correct / len(cat_results)

        return category_accuracy

    def _calculate_confidence_calibration(
        self, results: list[TestCaseResult]
    ) -> ConfidenceCalibration:
        """Calculate confidence calibration metrics."""
        high_correct = high_total = 0
        medium_correct = medium_total = 0
        low_correct = low_total = 0

        for result in results:
            if result.actual is None:
                continue

            confidence = result.actual.overall_confidence
            is_correct = result.passed

            if confidence >= self.high_threshold:
                high_total += 1
                if is_correct:
                    high_correct += 1
            elif confidence >= self.low_threshold:
                medium_total += 1
                if is_correct:
                    medium_correct += 1
            else:
                low_total += 1
                if is_correct:
                    low_correct += 1

        return ConfidenceCalibration(
            high_confidence_accuracy=(
                high_correct / high_total if high_total > 0 else 0.0
            ),
            medium_confidence_accuracy=(
                medium_correct / medium_total if medium_total > 0 else 0.0
            ),
            low_confidence_accuracy=(
                low_correct / low_total if low_total > 0 else 0.0
            ),
        )
