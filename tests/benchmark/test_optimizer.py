"""Tests for the optimizer and convergence tester."""

from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from policyflow.benchmark.models import (
    AnalysisReport,
    BenchmarkMetrics,
    BenchmarkReport,
    ConfidenceCalibration,
    ExpectedResult,
    FailurePattern,
    GoldenDataset,
    GoldenTestCase,
    Hypothesis,
    OptimizationBudget,
    TestCaseResult,
)
from policyflow.models import (
    EvaluationResult,
    HierarchicalWorkflowDefinition,
    NodeConfig,
    NodeGroup,
    ParsedWorkflowPolicy,
)


def create_test_workflow() -> ParsedWorkflowPolicy:
    """Create a minimal test workflow."""
    return ParsedWorkflowPolicy(
        title="Test Policy",
        description="Test workflow",
        workflow=HierarchicalWorkflowDefinition(
            nodes=[
                NodeConfig(
                    id="node_1",
                    type="LLMEvaluatorNode",
                    params={"confidence_threshold": 0.7},
                    routes={"complete": "end"},
                ),
            ],
            start_node="node_1",
            hierarchy=[
                NodeGroup(
                    clause_number="1",
                    clause_text="Test clause",
                    nodes=["node_1"],
                )
            ],
        ),
    )


def create_test_dataset() -> GoldenDataset:
    """Create a minimal test dataset."""
    return GoldenDataset(
        policy_file="test_policy.yaml",
        description="Test dataset",
        test_cases=[
            GoldenTestCase(
                id="test_001",
                name="Test case 1",
                input_text="Test input",
                expected=ExpectedResult(policy_satisfied=True),
                category="clear_pass",
            ),
            GoldenTestCase(
                id="test_002",
                name="Test case 2",
                input_text="Test input 2",
                expected=ExpectedResult(policy_satisfied=False),
                category="clear_fail",
            ),
        ],
    )


def create_test_report(accuracy: float = 0.8) -> BenchmarkReport:
    """Create a test benchmark report with given accuracy."""
    return BenchmarkReport(
        workflow_id="test_workflow",
        timestamp=datetime.now(),
        results=[
            TestCaseResult(
                test_id="test_001",
                expected=ExpectedResult(policy_satisfied=True),
                actual=EvaluationResult(
                    policy_satisfied=True,
                    policy_title="Test",
                    overall_reasoning="OK",
                    overall_confidence=0.9,
                ),
                duration_ms=100,
            ),
        ],
        metrics=BenchmarkMetrics(
            overall_accuracy=accuracy,
            confidence_calibration=ConfidenceCalibration(
                high_confidence_accuracy=accuracy,
                medium_confidence_accuracy=accuracy,
                low_confidence_accuracy=accuracy,
            ),
        ),
        llm_calls=1,
    )


class TestConvergenceTester:
    """Tests for the ConvergenceTester class."""

    def test_convergence_tester_initialization(self):
        """Test that ConvergenceTester can be initialized."""
        from policyflow.benchmark.optimizer import ConvergenceTester

        budget = OptimizationBudget(max_iterations=5)
        tester = ConvergenceTester(budget)

        assert tester is not None
        assert tester.budget == budget
        assert tester.iterations == 0
        assert tester.llm_calls == 0

    def test_record_step_updates_state(self):
        """Test that record_step updates internal state."""
        from policyflow.benchmark.optimizer import ConvergenceTester

        budget = OptimizationBudget(max_iterations=10)
        tester = ConvergenceTester(budget)

        tester.record_step(metric=0.75, llm_calls=5)

        assert tester.iterations == 1
        assert tester.llm_calls == 5
        assert tester.best_metric == 0.75
        assert len(tester.history) == 1

    def test_should_stop_max_iterations(self):
        """Test stopping when max iterations reached."""
        from policyflow.benchmark.optimizer import ConvergenceTester

        budget = OptimizationBudget(max_iterations=2)
        tester = ConvergenceTester(budget)

        tester.record_step(0.7, 1)
        should_stop, reason = tester.should_stop()
        assert not should_stop

        tester.record_step(0.8, 1)
        should_stop, reason = tester.should_stop()
        assert should_stop
        assert reason == "max_iterations"

    def test_should_stop_target_reached(self):
        """Test stopping when target metric reached."""
        from policyflow.benchmark.optimizer import ConvergenceTester

        budget = OptimizationBudget(
            max_iterations=10,
            target_metric=0.9,
        )
        tester = ConvergenceTester(budget)

        tester.record_step(0.85, 1)
        should_stop, reason = tester.should_stop()
        assert not should_stop

        tester.record_step(0.92, 1)
        should_stop, reason = tester.should_stop()
        assert should_stop
        assert reason == "target_reached"

    def test_should_stop_max_llm_calls(self):
        """Test stopping when max LLM calls reached."""
        from policyflow.benchmark.optimizer import ConvergenceTester

        budget = OptimizationBudget(
            max_iterations=100,
            max_llm_calls=10,
        )
        tester = ConvergenceTester(budget)

        tester.record_step(0.7, 5)
        should_stop, _ = tester.should_stop()
        assert not should_stop

        tester.record_step(0.75, 6)  # Total: 11
        should_stop, reason = tester.should_stop()
        assert should_stop
        assert reason == "max_llm_calls"

    def test_should_stop_patience(self):
        """Test stopping after N iterations without improvement."""
        from policyflow.benchmark.optimizer import ConvergenceTester

        budget = OptimizationBudget(
            max_iterations=100,
            patience=3,
            min_improvement=0.01,
        )
        tester = ConvergenceTester(budget)

        tester.record_step(0.80, 1)  # Initial - best=0.80, no_improve=0
        tester.record_step(0.82, 1)  # Improvement > 0.01 - best=0.82, no_improve=0
        tester.record_step(0.821, 1)  # No significant improvement - no_improve=1
        tester.record_step(0.822, 1)  # No significant improvement - no_improve=2
        tester.record_step(0.823, 1)  # No significant improvement - no_improve=3 = patience

        should_stop, reason = tester.should_stop()
        assert should_stop
        assert reason == "no_improvement"

    def test_should_stop_timeout(self):
        """Test stopping on timeout."""
        from policyflow.benchmark.optimizer import ConvergenceTester

        budget = OptimizationBudget(
            max_iterations=100,
            max_time_seconds=0.1,  # Very short timeout
        )
        tester = ConvergenceTester(budget)

        tester.record_step(0.7, 1)
        time.sleep(0.15)  # Exceed timeout

        should_stop, reason = tester.should_stop()
        assert should_stop
        assert reason == "timeout"

    def test_get_summary(self):
        """Test getting summary statistics."""
        from policyflow.benchmark.optimizer import ConvergenceTester

        budget = OptimizationBudget()
        tester = ConvergenceTester(budget)

        tester.record_step(0.7, 5)
        tester.record_step(0.8, 3)

        summary = tester.get_summary()

        assert summary["iterations"] == 2
        assert summary["llm_calls"] == 8
        assert summary["best_metric"] == 0.8
        assert summary["improvement_curve"] == [0.7, 0.8]
        assert "elapsed_seconds" in summary


class TestHillClimbingOptimizer:
    """Tests for the HillClimbingOptimizer."""

    def test_optimizer_initialization(self):
        """Test optimizer can be initialized."""
        from policyflow.benchmark.optimizer import HillClimbingOptimizer

        optimizer = HillClimbingOptimizer()
        assert optimizer is not None

    def test_optimizer_with_dependencies(self):
        """Test optimizer can be initialized with dependencies."""
        from policyflow.benchmark.analyzer import RuleBasedAnalyzer
        from policyflow.benchmark.applier import BasicHypothesisApplier
        from policyflow.benchmark.hypothesis import TemplateBasedHypothesisGenerator
        from policyflow.benchmark.optimizer import HillClimbingOptimizer

        optimizer = HillClimbingOptimizer(
            analyzer=RuleBasedAnalyzer(),
            hypothesis_generator=TemplateBasedHypothesisGenerator(),
            hypothesis_applier=BasicHypothesisApplier(),
        )
        assert optimizer is not None

    def test_step_returns_modified_or_none(self):
        """Test that step returns modified workflow or None."""
        from policyflow.benchmark.optimizer import HillClimbingOptimizer

        optimizer = HillClimbingOptimizer()
        workflow = create_test_workflow()
        report = create_test_report(accuracy=0.8)

        # Step might return modified workflow or None
        result = optimizer.step(workflow, report)

        # Result should be either ParsedWorkflowPolicy or None
        assert result is None or isinstance(result, ParsedWorkflowPolicy)

    def test_optimize_respects_max_iterations(self):
        """Test that optimize respects max_iterations budget."""
        from policyflow.benchmark.optimizer import HillClimbingOptimizer

        optimizer = HillClimbingOptimizer()
        workflow = create_test_workflow()
        dataset = create_test_dataset()

        budget = OptimizationBudget(max_iterations=2)

        # Mock the runner to avoid actual execution
        with patch.object(optimizer, "_run_benchmark") as mock_run:
            mock_run.return_value = create_test_report(accuracy=0.8)

            result = optimizer.optimize(
                workflow=workflow,
                dataset=dataset,
                budget=budget,
            )

        assert result is not None
        assert result.convergence_reason in ["max_iterations", "no_hypotheses", "target_reached"]
        assert len(result.history) <= budget.max_iterations + 1

    def test_optimize_returns_best_workflow(self):
        """Test that optimize returns the best workflow found."""
        from policyflow.benchmark.optimizer import HillClimbingOptimizer

        optimizer = HillClimbingOptimizer()
        workflow = create_test_workflow()
        dataset = create_test_dataset()

        budget = OptimizationBudget(max_iterations=1)

        with patch.object(optimizer, "_run_benchmark") as mock_run:
            mock_run.return_value = create_test_report(accuracy=0.85)

            result = optimizer.optimize(
                workflow=workflow,
                dataset=dataset,
                budget=budget,
            )

        assert result.best_metric >= 0.0
        assert result.best_workflow_yaml is not None

    def test_optimize_stops_on_target_reached(self):
        """Test that optimize stops when target metric is reached."""
        from policyflow.benchmark.optimizer import HillClimbingOptimizer

        optimizer = HillClimbingOptimizer()
        workflow = create_test_workflow()
        dataset = create_test_dataset()

        budget = OptimizationBudget(
            max_iterations=10,
            target_metric=0.9,
        )

        with patch.object(optimizer, "_run_benchmark") as mock_run:
            # Return 95% accuracy - should hit target
            mock_run.return_value = create_test_report(accuracy=0.95)

            result = optimizer.optimize(
                workflow=workflow,
                dataset=dataset,
                budget=budget,
            )

        assert result.converged is True
        assert result.convergence_reason == "target_reached"

    def test_optimize_with_custom_metric(self):
        """Test that optimize works with custom metric function."""
        from policyflow.benchmark.optimizer import HillClimbingOptimizer

        optimizer = HillClimbingOptimizer()
        workflow = create_test_workflow()
        dataset = create_test_dataset()

        budget = OptimizationBudget(max_iterations=1)

        # Custom metric that returns a different value
        def custom_metric(report: BenchmarkReport) -> float:
            return report.metrics.overall_accuracy * 0.5

        with patch.object(optimizer, "_run_benchmark") as mock_run:
            mock_run.return_value = create_test_report(accuracy=0.8)

            result = optimizer.optimize(
                workflow=workflow,
                dataset=dataset,
                budget=budget,
                metric=custom_metric,
            )

        # Best metric should be the custom metric result
        assert result.best_metric == 0.4  # 0.8 * 0.5


class TestOptimizerFactory:
    """Tests for optimizer factory functions."""

    def test_create_optimizer(self):
        """Test creating optimizer via factory."""
        from policyflow.benchmark.optimizer import HillClimbingOptimizer, create_optimizer

        optimizer = create_optimizer()
        assert isinstance(optimizer, HillClimbingOptimizer)

    def test_create_optimizer_with_mode(self):
        """Test creating optimizer with specific mode."""
        from policyflow.benchmark.optimizer import HillClimbingOptimizer, create_optimizer

        optimizer = create_optimizer(mode="hill_climbing")
        assert isinstance(optimizer, HillClimbingOptimizer)

    def test_create_optimizer_invalid_mode(self):
        """Test that invalid mode raises error."""
        from policyflow.benchmark.optimizer import create_optimizer

        with pytest.raises(ValueError, match="Invalid optimizer mode"):
            create_optimizer(mode="invalid_mode")


class TestOptimizerProtocolConformance:
    """Test that optimizer conforms to the Optimizer protocol."""

    def test_satisfies_protocol(self):
        """Test that HillClimbingOptimizer satisfies the protocol."""
        from policyflow.benchmark.optimizer import HillClimbingOptimizer
        from policyflow.benchmark.protocols import Optimizer

        optimizer = HillClimbingOptimizer()

        # Should have required methods
        assert hasattr(optimizer, "optimize")
        assert hasattr(optimizer, "step")
        assert callable(optimizer.optimize)
        assert callable(optimizer.step)


class TestOptimizationResult:
    """Tests for OptimizationResult structure."""

    def test_result_has_required_fields(self):
        """Test that optimization result has all required fields."""
        from policyflow.benchmark.optimizer import HillClimbingOptimizer

        optimizer = HillClimbingOptimizer()
        workflow = create_test_workflow()
        dataset = create_test_dataset()

        budget = OptimizationBudget(max_iterations=1)

        with patch.object(optimizer, "_run_benchmark") as mock_run:
            mock_run.return_value = create_test_report()

            result = optimizer.optimize(
                workflow=workflow,
                dataset=dataset,
                budget=budget,
            )

        assert hasattr(result, "best_workflow_yaml")
        assert hasattr(result, "best_metric")
        assert hasattr(result, "history")
        assert hasattr(result, "converged")
        assert hasattr(result, "convergence_reason")
        assert hasattr(result, "total_llm_calls")
        assert hasattr(result, "total_time_seconds")


class TestRunBenchmarkIntegration:
    """Tests for _run_benchmark integration with SimpleBenchmarkRunner."""

    def test_run_benchmark_uses_simple_runner(self):
        """Test that _run_benchmark uses SimpleBenchmarkRunner internally."""
        from policyflow.benchmark.optimizer import HillClimbingOptimizer
        from policyflow.benchmark.runner import SimpleBenchmarkRunner

        optimizer = HillClimbingOptimizer()
        workflow = create_test_workflow()
        dataset = create_test_dataset()

        # Mock SimpleBenchmarkRunner.run to avoid actual LLM calls
        with patch.object(SimpleBenchmarkRunner, "run") as mock_run:
            mock_run.return_value = create_test_report(accuracy=0.75)

            report = optimizer._run_benchmark(workflow, dataset)

        # Verify SimpleBenchmarkRunner.run was called
        mock_run.assert_called_once()
        # Verify it was called with the workflow and test_cases
        call_args = mock_run.call_args
        assert call_args[0][0] == workflow  # First arg is workflow
        assert call_args[0][1] == dataset.test_cases  # Second arg is test_cases

    def test_run_benchmark_returns_actual_report(self):
        """Test that _run_benchmark returns the report from SimpleBenchmarkRunner."""
        from policyflow.benchmark.optimizer import HillClimbingOptimizer
        from policyflow.benchmark.runner import SimpleBenchmarkRunner

        optimizer = HillClimbingOptimizer()
        workflow = create_test_workflow()
        dataset = create_test_dataset()

        expected_report = create_test_report(accuracy=0.85)

        with patch.object(SimpleBenchmarkRunner, "run") as mock_run:
            mock_run.return_value = expected_report

            report = optimizer._run_benchmark(workflow, dataset)

        assert report.metrics.overall_accuracy == 0.85
        assert report == expected_report
