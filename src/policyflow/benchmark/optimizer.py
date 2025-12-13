"""Optimization loop for workflow improvement.

Provides a hill-climbing optimizer that iteratively improves workflows
based on benchmark results and hypothesis generation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Literal

from policyflow.benchmark.analyzer import RuleBasedAnalyzer, create_analyzer
from policyflow.benchmark.applier import BasicHypothesisApplier, create_applier
from policyflow.benchmark.hypothesis import (
    TemplateBasedHypothesisGenerator,
    create_hypothesis_generator,
)
from policyflow.benchmark.models import (
    BenchmarkReport,
    GoldenDataset,
    OptimizationBudget,
    OptimizationResult,
    OptimizationStep,
)

if TYPE_CHECKING:
    from policyflow.benchmark.protocols import (
        FailureAnalyzer,
        HypothesisApplier,
        HypothesisGenerator,
    )
    from policyflow.models import ParsedWorkflowPolicy


class ConvergenceTester:
    """Tracks optimization progress and determines when to stop.

    Monitors metrics across iterations and checks various stopping conditions:
    - Maximum iterations reached
    - Target metric achieved
    - Maximum LLM calls exceeded
    - Timeout exceeded
    - No improvement for N iterations (patience)
    """

    def __init__(self, budget: OptimizationBudget):
        """Initialize with optimization budget.

        Args:
            budget: The optimization budget constraints
        """
        self.budget = budget
        self.history: list[float] = []
        self.iterations = 0
        self.llm_calls = 0
        self.start_time = time.time()
        self.best_metric = float("-inf")
        self.steps_without_improvement = 0

    def record_step(self, metric: float, llm_calls: int) -> None:
        """Record a step in the optimization process.

        Args:
            metric: The metric value for this step
            llm_calls: Number of LLM calls made in this step
        """
        self.history.append(metric)
        self.iterations += 1
        self.llm_calls += llm_calls

        # Check for significant improvement
        if metric > self.best_metric + self.budget.min_improvement:
            self.best_metric = metric
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

    def should_stop(self) -> tuple[bool, str]:
        """Check if optimization should stop.

        Returns:
            Tuple of (should_stop, reason)
        """
        elapsed = time.time() - self.start_time

        # Check target metric reached
        if (
            self.budget.target_metric is not None
            and self.best_metric >= self.budget.target_metric
        ):
            return True, "target_reached"

        # Check max iterations
        if self.iterations >= self.budget.max_iterations:
            return True, "max_iterations"

        # Check max LLM calls
        if self.llm_calls >= self.budget.max_llm_calls:
            return True, "max_llm_calls"

        # Check timeout
        if elapsed >= self.budget.max_time_seconds:
            return True, "timeout"

        # Check patience (no improvement)
        if self.steps_without_improvement >= self.budget.patience:
            return True, "no_improvement"

        return False, ""

    def get_summary(self) -> dict:
        """Get summary statistics of the optimization run.

        Returns:
            Dictionary with summary statistics
        """
        return {
            "iterations": self.iterations,
            "llm_calls": self.llm_calls,
            "elapsed_seconds": time.time() - self.start_time,
            "best_metric": self.best_metric,
            "improvement_curve": self.history.copy(),
        }


class HillClimbingOptimizer:
    """Simple optimizer that applies hypotheses one at a time.

    Uses a greedy hill-climbing approach:
    1. Benchmark current workflow
    2. Analyze failures
    3. Generate improvement hypotheses
    4. Apply top hypothesis
    5. Repeat until convergence or budget exhausted
    """

    def __init__(
        self,
        analyzer: "FailureAnalyzer | None" = None,
        hypothesis_generator: "HypothesisGenerator | None" = None,
        hypothesis_applier: "HypothesisApplier | None" = None,
    ):
        """Initialize the optimizer with optional components.

        Args:
            analyzer: Failure analyzer (defaults to RuleBasedAnalyzer)
            hypothesis_generator: Hypothesis generator (defaults to TemplateBasedHypothesisGenerator)
            hypothesis_applier: Hypothesis applier (defaults to BasicHypothesisApplier)
        """
        self.analyzer = analyzer or RuleBasedAnalyzer()
        self.generator = hypothesis_generator or TemplateBasedHypothesisGenerator()
        self.applier = hypothesis_applier or BasicHypothesisApplier()

    def optimize(
        self,
        workflow: "ParsedWorkflowPolicy",
        dataset: GoldenDataset,
        budget: OptimizationBudget,
        metric: Callable[[BenchmarkReport], float] | None = None,
    ) -> OptimizationResult:
        """Run full optimization loop.

        Args:
            workflow: Initial workflow to optimize
            dataset: Golden dataset for benchmarking
            budget: Optimization budget constraints
            metric: Function to compute optimization metric from benchmark
                    (defaults to overall_accuracy)

        Returns:
            Optimization result with best workflow and history
        """
        if metric is None:
            metric = lambda r: r.metrics.overall_accuracy

        tester = ConvergenceTester(budget)
        history: list[OptimizationStep] = []

        current_workflow = workflow
        best_workflow = workflow
        best_metric = float("-inf")

        while True:
            # 1. Benchmark current workflow
            report = self._run_benchmark(current_workflow, dataset)
            current_metric = metric(report)
            tester.record_step(current_metric, report.llm_calls)

            # Track best
            if current_metric > best_metric:
                best_metric = current_metric
                best_workflow = current_workflow

            # Record history
            history.append(
                OptimizationStep(
                    iteration=tester.iterations,
                    workflow_snapshot=current_workflow.to_yaml(),
                    metric=current_metric,
                    changes_made=[],
                    llm_calls=report.llm_calls,
                )
            )

            # 2. Check convergence
            should_stop, reason = tester.should_stop()
            if should_stop:
                summary = tester.get_summary()
                return OptimizationResult(
                    best_workflow_yaml=best_workflow.to_yaml(),
                    best_metric=best_metric,
                    history=history,
                    converged=(reason == "target_reached"),
                    convergence_reason=reason,
                    total_llm_calls=summary["llm_calls"],
                    total_time_seconds=summary["elapsed_seconds"],
                )

            # 3. Analyze failures and generate hypotheses
            analysis = self.analyzer.analyze(report, current_workflow)
            hypotheses = self.generator.generate(analysis, current_workflow)

            if not hypotheses:
                summary = tester.get_summary()
                return OptimizationResult(
                    best_workflow_yaml=best_workflow.to_yaml(),
                    best_metric=best_metric,
                    history=history,
                    converged=True,
                    convergence_reason="no_hypotheses",
                    total_llm_calls=summary["llm_calls"],
                    total_time_seconds=summary["elapsed_seconds"],
                )

            # 4. Try top hypothesis (greedy)
            best_hypothesis = hypotheses[0]
            try:
                current_workflow = self.applier.apply(current_workflow, best_hypothesis)
                history[-1].changes_made = [best_hypothesis.description]
            except ValueError as e:
                # Log rejected hypothesis with reason
                history[-1].changes_made = [f"REJECTED: {best_hypothesis.description} - {e}"]
                # Continue with current workflow unchanged

    def step(
        self,
        workflow: "ParsedWorkflowPolicy",
        report: BenchmarkReport,
    ) -> "ParsedWorkflowPolicy | None":
        """Single optimization step.

        Args:
            workflow: Current workflow
            report: Benchmark report from current workflow

        Returns:
            Modified workflow, or None if no improvement found
        """
        # Analyze failures
        analysis = self.analyzer.analyze(report, workflow)

        # Generate hypotheses
        hypotheses = self.generator.generate(analysis, workflow)

        if not hypotheses:
            return None

        # Apply top hypothesis
        try:
            modified = self.applier.apply(workflow, hypotheses[0])
            return modified
        except ValueError:
            # Hypothesis could not be applied (e.g., target node doesn't exist)
            # Try remaining hypotheses
            for hyp in hypotheses[1:]:
                try:
                    modified = self.applier.apply(workflow, hyp)
                    return modified
                except ValueError:
                    continue
            return None

    def _run_benchmark(
        self,
        workflow: "ParsedWorkflowPolicy",
        dataset: GoldenDataset,
    ) -> BenchmarkReport:
        """Run benchmark on workflow with dataset.

        Uses SimpleBenchmarkRunner to execute the workflow against all test cases
        in the dataset and compute metrics.

        Args:
            workflow: The workflow to benchmark
            dataset: Golden dataset with test cases

        Returns:
            Complete benchmark report with results and metrics
        """
        from policyflow.benchmark.runner import BenchmarkConfig, SimpleBenchmarkRunner

        config = BenchmarkConfig(workflow_id=workflow.title)
        runner = SimpleBenchmarkRunner(config)
        return runner.run(workflow, dataset.test_cases)


def create_optimizer(
    mode: Literal["hill_climbing"] = "hill_climbing",
    analyzer_mode: Literal["rule_based", "llm", "hybrid"] = "hybrid",
    hypothesis_mode: Literal["template", "llm", "hybrid"] = "hybrid",
) -> HillClimbingOptimizer:
    """Factory function to create an optimizer.

    Args:
        mode: Optimizer mode (currently only "hill_climbing" supported)
        analyzer_mode: Mode for the failure analyzer
        hypothesis_mode: Mode for the hypothesis generator

    Returns:
        Configured optimizer instance

    Raises:
        ValueError: If mode is invalid
    """
    if mode == "hill_climbing":
        return HillClimbingOptimizer(
            analyzer=create_analyzer(analyzer_mode),
            hypothesis_generator=create_hypothesis_generator(hypothesis_mode),
            hypothesis_applier=create_applier(),
        )
    else:
        raise ValueError(f"Invalid optimizer mode: {mode}")
