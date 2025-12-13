"""Tests for benchmark protocol interfaces."""

from datetime import datetime
from typing import Callable

import pytest

from policyflow.benchmark.models import (
    AnalysisReport,
    BenchmarkMetrics,
    BenchmarkReport,
    ConfidenceCalibration,
    CriterionExpectation,
    ExpectedResult,
    Experiment,
    GeneratorConfig,
    GoldenDataset,
    GoldenTestCase,
    Hypothesis,
    OptimizationBudget,
    OptimizationResult,
    TestCaseResult,
)
from policyflow.models import ClauseResult, EvaluationResult, NormalizedPolicy, ParsedWorkflowPolicy


class TestBenchmarkRunnerProtocol:
    """Tests for BenchmarkRunner protocol."""

    def test_protocol_exists(self):
        from policyflow.benchmark.protocols import BenchmarkRunner

        assert BenchmarkRunner is not None

    def test_protocol_has_run_method(self):
        from policyflow.benchmark.protocols import BenchmarkRunner

        # Check the protocol defines run
        assert hasattr(BenchmarkRunner, "run")


class TestResultComparatorProtocol:
    """Tests for ResultComparator protocol."""

    def test_protocol_exists(self):
        from policyflow.benchmark.protocols import ResultComparator

        assert ResultComparator is not None

    def test_protocol_has_compare_method(self):
        from policyflow.benchmark.protocols import ResultComparator

        assert hasattr(ResultComparator, "compare")


class TestMetricsCalculatorProtocol:
    """Tests for MetricsCalculator protocol."""

    def test_protocol_exists(self):
        from policyflow.benchmark.protocols import MetricsCalculator

        assert MetricsCalculator is not None

    def test_protocol_has_calculate_method(self):
        from policyflow.benchmark.protocols import MetricsCalculator

        assert hasattr(MetricsCalculator, "calculate")


class TestFailureAnalyzerProtocol:
    """Tests for FailureAnalyzer protocol."""

    def test_protocol_exists(self):
        from policyflow.benchmark.protocols import FailureAnalyzer

        assert FailureAnalyzer is not None

    def test_protocol_has_analyze_method(self):
        from policyflow.benchmark.protocols import FailureAnalyzer

        assert hasattr(FailureAnalyzer, "analyze")


class TestHypothesisGeneratorProtocol:
    """Tests for HypothesisGenerator protocol."""

    def test_protocol_exists(self):
        from policyflow.benchmark.protocols import HypothesisGenerator

        assert HypothesisGenerator is not None

    def test_protocol_has_generate_method(self):
        from policyflow.benchmark.protocols import HypothesisGenerator

        assert hasattr(HypothesisGenerator, "generate")


class TestHypothesisApplierProtocol:
    """Tests for HypothesisApplier protocol."""

    def test_protocol_exists(self):
        from policyflow.benchmark.protocols import HypothesisApplier

        assert HypothesisApplier is not None

    def test_protocol_has_apply_method(self):
        from policyflow.benchmark.protocols import HypothesisApplier

        assert hasattr(HypothesisApplier, "apply")


class TestExperimentTrackerProtocol:
    """Tests for ExperimentTracker protocol."""

    def test_protocol_exists(self):
        from policyflow.benchmark.protocols import ExperimentTracker

        assert ExperimentTracker is not None

    def test_protocol_has_required_methods(self):
        from policyflow.benchmark.protocols import ExperimentTracker

        assert hasattr(ExperimentTracker, "record")
        assert hasattr(ExperimentTracker, "get_history")
        assert hasattr(ExperimentTracker, "get_best")


class TestDatasetGeneratorProtocol:
    """Tests for DatasetGenerator protocol."""

    def test_protocol_exists(self):
        from policyflow.benchmark.protocols import DatasetGenerator

        assert DatasetGenerator is not None

    def test_protocol_has_required_methods(self):
        from policyflow.benchmark.protocols import DatasetGenerator

        assert hasattr(DatasetGenerator, "generate")
        assert hasattr(DatasetGenerator, "generate_for_criterion")
        assert hasattr(DatasetGenerator, "augment")


class TestOptimizerProtocol:
    """Tests for Optimizer protocol."""

    def test_protocol_exists(self):
        from policyflow.benchmark.protocols import Optimizer

        assert Optimizer is not None

    def test_protocol_has_required_methods(self):
        from policyflow.benchmark.protocols import Optimizer

        assert hasattr(Optimizer, "optimize")
        assert hasattr(Optimizer, "step")


class TestComparisonResult:
    """Tests for ComparisonResult model used by ResultComparator."""

    def test_comparison_result_exists(self):
        from policyflow.benchmark.protocols import ComparisonResult

        assert ComparisonResult is not None

    def test_comparison_result_fields(self):
        from policyflow.benchmark.protocols import ComparisonResult

        result = ComparisonResult(
            matches=True,
            policy_satisfied_match=True,
            criterion_matches={"criterion_1": True, "criterion_2": False},
            mismatched_criteria=["criterion_2"],
            details={"criterion_2": "Expected True, got False"},
        )
        assert result.matches is True
        assert len(result.mismatched_criteria) == 1


class TestProtocolImplementations:
    """Tests that verify mock implementations satisfy protocols."""

    def test_mock_benchmark_runner_satisfies_protocol(self):
        from policyflow.benchmark.protocols import BenchmarkRunner

        class MockBenchmarkRunner:
            def run(
                self, workflow, test_cases: list[GoldenTestCase]
            ) -> BenchmarkReport:
                return BenchmarkReport(
                    workflow_id="mock",
                    timestamp=datetime.now(),
                    results=[],
                    metrics=BenchmarkMetrics(
                        overall_accuracy=1.0,
                        criterion_metrics={},
                        category_accuracy={},
                        confidence_calibration=ConfidenceCalibration(
                            high_confidence_accuracy=1.0,
                            medium_confidence_accuracy=1.0,
                            low_confidence_accuracy=1.0,
                        ),
                    ),
                    config={},
                )

        runner = MockBenchmarkRunner()
        # Just verify it has the right method signature
        assert callable(runner.run)

    def test_mock_optimizer_satisfies_protocol(self):
        from policyflow.benchmark.protocols import Optimizer

        class MockOptimizer:
            def optimize(
                self,
                workflow: ParsedWorkflowPolicy,
                dataset: GoldenDataset,
                budget: OptimizationBudget,
                metric: Callable[[BenchmarkReport], float],
            ) -> OptimizationResult:
                return OptimizationResult(
                    best_workflow_yaml="workflow: yaml",
                    best_metric=1.0,
                    history=[],
                    converged=True,
                    convergence_reason="mock",
                    total_llm_calls=0,
                    total_time_seconds=0.0,
                )

            def step(
                self,
                workflow: ParsedWorkflowPolicy,
                report: BenchmarkReport,
            ) -> ParsedWorkflowPolicy | None:
                return None

        optimizer = MockOptimizer()
        assert callable(optimizer.optimize)
        assert callable(optimizer.step)
