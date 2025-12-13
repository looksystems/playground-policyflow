"""Data models for the benchmark system."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from policyflow.models import EvaluationResult, YAMLMixin


# ============================================================================
# Golden Dataset Models
# ============================================================================


class CriterionExpectation(BaseModel):
    """Expected result for a single criterion."""

    met: bool = Field(description="Whether the criterion should be met")
    sub_results: dict[str, "CriterionExpectation"] | None = Field(
        default=None,
        description="Expected sub-criterion results",
    )


class ExpectedResult(BaseModel):
    """Expected evaluation result for a test case."""

    policy_satisfied: bool = Field(description="Whether the policy should be satisfied")
    criterion_results: dict[str, CriterionExpectation] = Field(
        default_factory=dict,
        description="Expected results per criterion",
    )


class IntermediateState(BaseModel):
    """Expected state at a specific clause/subclause for debugging."""

    clause_id: str = Field(description="Clause identifier")
    expected_met: bool = Field(description="Whether this clause should be met")
    key_signals: list[str] = Field(
        default_factory=list,
        description="Text signals that indicate this result",
    )
    reasoning: str = Field(
        default="",
        description="Explanation for the expected result",
    )


class GoldenTestCase(BaseModel):
    """A single test case in the golden dataset."""

    id: str = Field(description="Unique test case identifier")
    name: str = Field(description="Human-readable test name")
    input_text: str = Field(description="Text to evaluate")
    expected: ExpectedResult = Field(description="Expected evaluation result")
    category: str = Field(description="Test category (clear_pass, clear_fail, edge_case, etc.)")
    notes: str = Field(default="", description="Additional notes about the test case")
    intermediate_expectations: dict[str, IntermediateState] | None = Field(
        default=None,
        description="Intermediate state expectations for debugging",
    )


class GeneratorConfig(BaseModel):
    """Configuration for golden dataset generation."""

    # Scope control
    cases_per_criterion: int = Field(
        default=3,
        description="Number of test cases to generate per criterion",
    )
    include_edge_cases: bool = Field(
        default=True,
        description="Whether to generate edge cases",
    )
    include_partial_matches: bool = Field(
        default=True,
        description="Whether to generate partial match cases",
    )

    # Category distribution
    categories: list[str] = Field(
        default_factory=lambda: ["clear_pass", "clear_fail", "edge_case", "partial_match"],
        description="Categories of test cases to generate",
    )

    # Edge case strategies
    edge_case_strategies: list[str] = Field(
        default_factory=lambda: ["boundary", "negation", "ambiguous", "implicit", "missing_element"],
        description="Strategies for edge case generation",
    )

    # Output control
    include_reasoning: bool = Field(
        default=True,
        description="Whether to include expected reasoning",
    )
    include_intermediate_states: bool = Field(
        default=True,
        description="Whether to include clause-by-clause expectations",
    )

    # LLM settings
    mode: Literal["llm", "template", "hybrid"] = Field(
        default="hybrid",
        description="Generation mode",
    )
    temperature: float = Field(
        default=0.7,
        description="LLM temperature for generation diversity",
    )


class GenerationMetadata(BaseModel):
    """Metadata about dataset generation."""

    generator_version: str = Field(description="Version of the generator used")
    config_used: GeneratorConfig = Field(description="Configuration used for generation")
    timestamp: datetime = Field(description="When the dataset was generated")
    policy_hash: str = Field(description="Hash of the source policy for tracking changes")


class GoldenDataset(YAMLMixin, BaseModel):
    """A complete golden dataset for benchmarking."""

    policy_file: str = Field(description="Path to the policy file")
    description: str = Field(description="Description of the dataset")
    test_cases: list[GoldenTestCase] = Field(
        default_factory=list,
        description="All test cases in the dataset",
    )
    generation_metadata: GenerationMetadata | None = Field(
        default=None,
        description="Metadata about how the dataset was generated",
    )

    def filter_by_category(self, category: str) -> list[GoldenTestCase]:
        """Filter test cases by category."""
        return [tc for tc in self.test_cases if tc.category == category]

    def filter_by_ids(self, ids: list[str]) -> list[GoldenTestCase]:
        """Filter test cases by IDs."""
        return [tc for tc in self.test_cases if tc.id in ids]


# ============================================================================
# Benchmark Result Models
# ============================================================================


class ConfusionMatrix(BaseModel):
    """Confusion matrix for binary classification."""

    tp: int = Field(description="True positives")
    tn: int = Field(description="True negatives")
    fp: int = Field(description="False positives")
    fn: int = Field(description="False negatives")

    @property
    def total(self) -> int:
        """Total number of samples."""
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        if self.total == 0:
            return 0.0
        return (self.tp + self.tn) / self.total

    @property
    def precision(self) -> float:
        """Calculate precision."""
        denominator = self.tp + self.fp
        if denominator == 0:
            return 0.0
        return self.tp / denominator

    @property
    def recall(self) -> float:
        """Calculate recall."""
        denominator = self.tp + self.fn
        if denominator == 0:
            return 0.0
        return self.tp / denominator

    @property
    def f1(self) -> float:
        """Calculate F1 score."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)


class CriterionMetrics(BaseModel):
    """Metrics for a single criterion."""

    accuracy: float = Field(ge=0.0, le=1.0, description="Accuracy score")
    precision: float = Field(ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(ge=0.0, le=1.0, description="Recall score")
    f1: float = Field(ge=0.0, le=1.0, description="F1 score")
    confusion: ConfusionMatrix = Field(description="Confusion matrix")


class ConfidenceCalibration(BaseModel):
    """Calibration metrics for confidence scores."""

    high_confidence_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Accuracy when confidence is high",
    )
    medium_confidence_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Accuracy when confidence is medium",
    )
    low_confidence_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Accuracy when confidence is low",
    )


class BenchmarkMetrics(BaseModel):
    """Complete metrics from a benchmark run."""

    overall_accuracy: float = Field(ge=0.0, le=1.0, description="Overall accuracy")
    criterion_metrics: dict[str, CriterionMetrics] = Field(
        default_factory=dict,
        description="Metrics per criterion",
    )
    category_accuracy: dict[str, float] = Field(
        default_factory=dict,
        description="Accuracy per test category",
    )
    confidence_calibration: ConfidenceCalibration = Field(
        description="Confidence calibration metrics"
    )


class TestCaseResult(BaseModel):
    """Result of running a single test case."""

    test_id: str = Field(description="ID of the test case")
    category: str = Field(default="unknown", description="Test category (clear_pass, clear_fail, etc.)")
    expected: ExpectedResult = Field(description="Expected result")
    actual: EvaluationResult | None = Field(
        default=None,
        description="Actual result (None if error occurred)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if execution failed",
    )
    duration_ms: float = Field(description="Execution time in milliseconds")

    @property
    def passed(self) -> bool:
        """Check if the test passed (policy_satisfied matches)."""
        if self.actual is None:
            return False
        return self.actual.policy_satisfied == self.expected.policy_satisfied

    @property
    def is_error(self) -> bool:
        """Check if the test had an error."""
        return self.error is not None


class BenchmarkReport(YAMLMixin, BaseModel):
    """Complete benchmark report."""

    workflow_id: str = Field(description="Identifier for the workflow version")
    timestamp: datetime = Field(description="When the benchmark was run")
    results: list[TestCaseResult] = Field(
        default_factory=list,
        description="Results for each test case",
    )
    metrics: BenchmarkMetrics = Field(description="Aggregate metrics")
    config: dict = Field(
        default_factory=dict,
        description="Snapshot of workflow configuration",
    )
    llm_calls: int = Field(
        default=0,
        description="Total LLM calls made during benchmark",
    )

    @property
    def failures(self) -> list[TestCaseResult]:
        """Get all failed test cases."""
        return [r for r in self.results if not r.passed]

    @property
    def errors(self) -> list[TestCaseResult]:
        """Get all test cases with errors."""
        return [r for r in self.results if r.is_error]


# ============================================================================
# Analysis Models
# ============================================================================


class FailurePattern(BaseModel):
    """A pattern identified in benchmark failures."""

    pattern_type: str = Field(
        description="Type of pattern (category_cluster, criterion_systematic, etc.)"
    )
    description: str = Field(description="Human-readable description")
    affected_tests: list[str] = Field(
        default_factory=list,
        description="IDs of affected test cases",
    )
    severity: Literal["high", "medium", "low"] = Field(description="Pattern severity")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Structured metadata (criterion, category, etc.) to avoid regex parsing",
    )


class ProblematicCriterion(BaseModel):
    """A criterion that shows problematic behavior."""

    criterion_id: str = Field(description="Criterion identifier")
    failure_rate: float = Field(ge=0.0, le=1.0, description="Overall failure rate")
    false_positive_rate: float = Field(ge=0.0, le=1.0, description="False positive rate")
    false_negative_rate: float = Field(ge=0.0, le=1.0, description="False negative rate")
    common_failure_patterns: list[str] = Field(
        default_factory=list,
        description="Common patterns in failures",
    )


class AnalysisReport(YAMLMixin, BaseModel):
    """Report from analyzing benchmark failures."""

    patterns: list[FailurePattern] = Field(
        default_factory=list,
        description="Identified failure patterns",
    )
    problematic_criteria: list[ProblematicCriterion] = Field(
        default_factory=list,
        description="Criteria with high failure rates",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations",
    )

    def has_systematic_failures(self) -> bool:
        """Check if there are systematic failure patterns."""
        return any(p.severity == "high" for p in self.patterns)

    def get_weak_strategies(self) -> list[str]:
        """Get strategies that need more test coverage."""
        weak = []
        for pattern in self.patterns:
            if "edge_case" in pattern.pattern_type:
                weak.append(pattern.pattern_type.replace("edge_case_", ""))
        return weak


class Hypothesis(YAMLMixin, BaseModel):
    """A hypothesis for workflow improvement."""

    id: str = Field(description="Unique hypothesis identifier")
    description: str = Field(description="Human-readable description")
    change_type: Literal["node_param", "workflow_structure", "prompt_tuning", "threshold"] = Field(
        description="Type of change proposed"
    )
    target: str = Field(description="Target node ID or 'workflow'")
    suggested_change: dict = Field(description="The proposed change")
    rationale: str = Field(description="Why this change might help")
    expected_impact: str = Field(description="Expected impact on metrics")


# ============================================================================
# Optimization Models
# ============================================================================


class OptimizationBudget(BaseModel):
    """Controls when optimization stops."""

    max_iterations: int = Field(default=10, description="Maximum optimization iterations")
    max_llm_calls: int = Field(default=100, description="Maximum LLM calls")
    max_time_seconds: float = Field(default=3600.0, description="Maximum time in seconds")
    target_metric: float | None = Field(
        default=None,
        description="Stop if metric reaches this value",
    )
    min_improvement: float = Field(
        default=0.01,
        description="Minimum improvement to continue",
    )
    patience: int = Field(
        default=3,
        description="Stop after N iterations without improvement",
    )


class OptimizationStep(BaseModel):
    """A single step in the optimization process."""

    iteration: int = Field(description="Iteration number")
    workflow_snapshot: str = Field(description="YAML snapshot of workflow at this step")
    metric: float = Field(description="Metric value at this step")
    changes_made: list[str] = Field(
        default_factory=list,
        description="Changes applied in this step",
    )
    llm_calls: int = Field(description="LLM calls made in this step")


class OptimizationResult(YAMLMixin, BaseModel):
    """Result of an optimization run."""

    best_workflow_yaml: str = Field(description="YAML of the best workflow found")
    best_metric: float = Field(description="Best metric achieved")
    history: list[OptimizationStep] = Field(
        default_factory=list,
        description="History of optimization steps",
    )
    converged: bool = Field(description="Whether optimization converged")
    convergence_reason: str = Field(description="Reason for stopping")
    total_llm_calls: int = Field(description="Total LLM calls made")
    total_time_seconds: float = Field(description="Total time taken")


# ============================================================================
# Experiment Tracking Models
# ============================================================================


class Experiment(YAMLMixin, BaseModel):
    """A tracked experiment."""

    id: str = Field(description="Unique experiment identifier")
    timestamp: datetime = Field(description="When the experiment was run")
    workflow_snapshot: str = Field(description="YAML snapshot of the workflow")
    hypothesis_applied: Hypothesis | None = Field(
        default=None,
        description="Hypothesis that was applied (None for baseline)",
    )
    benchmark_report: BenchmarkReport = Field(description="Benchmark results")
    parent_experiment_id: str | None = Field(
        default=None,
        description="ID of parent experiment for lineage tracking",
    )

    @property
    def accuracy(self) -> float:
        """Get the overall accuracy from this experiment."""
        return self.benchmark_report.metrics.overall_accuracy
