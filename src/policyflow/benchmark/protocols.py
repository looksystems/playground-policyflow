"""Protocol interfaces for the benchmark system.

These protocols define the contracts that implementations must follow,
allowing for easy swapping of implementations (rule-based, LLM-powered, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from policyflow.benchmark.models import (
        AnalysisReport,
        BenchmarkMetrics,
        BenchmarkReport,
        Experiment,
        GeneratorConfig,
        GoldenDataset,
        GoldenTestCase,
        Hypothesis,
        OptimizationBudget,
        OptimizationResult,
        TestCaseResult,
    )
    from policyflow.models import (
        Clause,
        EvaluationResult,
        NormalizedPolicy,
        ParsedWorkflowPolicy,
    )


# ============================================================================
# Comparison Models
# ============================================================================


class ComparisonResult(BaseModel):
    """Result of comparing actual vs expected evaluation results."""

    matches: bool = Field(description="Whether the overall result matches")
    policy_satisfied_match: bool = Field(
        description="Whether policy_satisfied matches"
    )
    criterion_matches: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-criterion match status",
    )
    mismatched_criteria: list[str] = Field(
        default_factory=list,
        description="IDs of criteria that didn't match",
    )
    details: dict[str, str] = Field(
        default_factory=dict,
        description="Detailed mismatch information",
    )


# ============================================================================
# Core Protocols
# ============================================================================


class BenchmarkRunner(Protocol):
    """Runs workflow against test cases.

    Implementations should execute the workflow for each test case
    and collect results including timing and any errors.
    """

    def run(
        self,
        workflow: "ParsedWorkflowPolicy",
        test_cases: list["GoldenTestCase"],
    ) -> "BenchmarkReport":
        """Run benchmark against all test cases.

        Args:
            workflow: The workflow to benchmark
            test_cases: Test cases to run

        Returns:
            Complete benchmark report with results and metrics
        """
        ...


class ResultComparator(Protocol):
    """Compares actual vs expected results.

    Implementations should perform deep comparison of evaluation results,
    including nested criterion and sub-criterion results.
    """

    def compare(
        self,
        actual: "EvaluationResult",
        expected: "ExpectedResult",
    ) -> ComparisonResult:
        """Compare actual evaluation result against expected.

        Args:
            actual: The actual result from workflow execution
            expected: The expected result from the golden dataset

        Returns:
            Detailed comparison result
        """
        ...


class MetricsCalculator(Protocol):
    """Calculates metrics from benchmark results.

    Implementations should compute overall accuracy, per-criterion
    precision/recall/F1, and confidence calibration metrics.
    """

    def calculate(self, results: list["TestCaseResult"]) -> "BenchmarkMetrics":
        """Calculate aggregate metrics from test results.

        Args:
            results: List of individual test case results

        Returns:
            Aggregate benchmark metrics
        """
        ...


class FailureAnalyzer(Protocol):
    """Analyzes failure patterns.

    Implementations can be rule-based, LLM-powered, or hybrid.
    They should identify patterns in failures to guide improvement.
    """

    def analyze(
        self,
        report: "BenchmarkReport",
        workflow: "ParsedWorkflowPolicy",
    ) -> "AnalysisReport":
        """Analyze benchmark failures to identify patterns.

        Args:
            report: The benchmark report to analyze
            workflow: The workflow that was benchmarked

        Returns:
            Analysis report with patterns and recommendations
        """
        ...


class HypothesisGenerator(Protocol):
    """Generates improvement hypotheses.

    Implementations can use templates, LLM generation, or hybrid approaches
    to suggest workflow improvements based on failure analysis.
    """

    def generate(
        self,
        analysis: "AnalysisReport",
        workflow: "ParsedWorkflowPolicy",
    ) -> list["Hypothesis"]:
        """Generate hypotheses for workflow improvement.

        Args:
            analysis: Analysis report with identified patterns
            workflow: The current workflow

        Returns:
            List of improvement hypotheses
        """
        ...


class HypothesisApplier(Protocol):
    """Applies a hypothesis to modify a workflow.

    Implementations should handle different change types:
    - node_param: Update node parameters
    - prompt_tuning: Update prompt templates
    - workflow_structure: Add/remove/rewire nodes
    - threshold: Update confidence/gate thresholds
    """

    def apply(
        self,
        workflow: "ParsedWorkflowPolicy",
        hypothesis: "Hypothesis",
    ) -> "ParsedWorkflowPolicy":
        """Apply a hypothesis to create a modified workflow.

        Args:
            workflow: The workflow to modify
            hypothesis: The hypothesis to apply

        Returns:
            Modified workflow
        """
        ...


class ExperimentTracker(Protocol):
    """Tracks experiments over time.

    Implementations should persist experiment data (YAML files recommended)
    and provide querying capabilities for history and comparisons.
    """

    def record(self, experiment: "Experiment") -> None:
        """Record an experiment.

        Args:
            experiment: The experiment to record
        """
        ...

    def get_history(self) -> list["Experiment"]:
        """Get all recorded experiments.

        Returns:
            List of all experiments, sorted by timestamp
        """
        ...

    def get_best(self) -> "Experiment | None":
        """Get the best experiment by accuracy.

        Returns:
            The experiment with highest accuracy, or None if no experiments
        """
        ...

    def get_by_id(self, experiment_id: str) -> "Experiment | None":
        """Get an experiment by ID.

        Args:
            experiment_id: The experiment ID to find

        Returns:
            The experiment if found, None otherwise
        """
        ...


class DatasetGenerator(Protocol):
    """Generates golden dataset from normalized policy.

    Implementations should generate comprehensive test cases including
    clear passes, clear fails, edge cases, and partial matches.
    """

    def generate(
        self,
        policy: "NormalizedPolicy",
        config: "GeneratorConfig",
    ) -> "GoldenDataset":
        """Generate a complete golden dataset.

        Args:
            policy: The normalized policy to generate tests for
            config: Generation configuration

        Returns:
            Complete golden dataset
        """
        ...

    def generate_for_criterion(
        self,
        criterion: "Clause",
        policy: "NormalizedPolicy",
        count: int,
    ) -> list["GoldenTestCase"]:
        """Generate test cases for a specific criterion.

        Args:
            criterion: The criterion to generate tests for
            policy: The full policy for context
            count: Number of test cases to generate

        Returns:
            List of generated test cases
        """
        ...

    def augment(
        self,
        existing: "GoldenDataset",
        policy: "NormalizedPolicy",
        config: "GeneratorConfig",
    ) -> "GoldenDataset":
        """Augment an existing dataset with more test cases.

        Args:
            existing: The existing dataset to augment
            policy: The normalized policy
            config: Generation configuration

        Returns:
            Augmented dataset
        """
        ...


class Optimizer(Protocol):
    """Base protocol for all optimizers.

    Implementations can use various strategies:
    - Hill climbing with hypothesis-based changes
    - Bayesian optimization for continuous parameters
    - Evolutionary/genetic approaches
    - LLM-powered meta-optimization
    """

    def optimize(
        self,
        workflow: "ParsedWorkflowPolicy",
        dataset: "GoldenDataset",
        budget: "OptimizationBudget",
        metric: Callable[["BenchmarkReport"], float],
    ) -> "OptimizationResult":
        """Run full optimization loop.

        Args:
            workflow: Initial workflow to optimize
            dataset: Golden dataset for benchmarking
            budget: Optimization budget constraints
            metric: Function to compute optimization metric from benchmark

        Returns:
            Optimization result with best workflow and history
        """
        ...

    def step(
        self,
        workflow: "ParsedWorkflowPolicy",
        report: "BenchmarkReport",
    ) -> "ParsedWorkflowPolicy | None":
        """Single optimization step.

        Args:
            workflow: Current workflow
            report: Benchmark report from current workflow

        Returns:
            Modified workflow, or None if converged
        """
        ...
