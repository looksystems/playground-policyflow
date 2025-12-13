"""Benchmark system for evaluating and improving policy workflows.

This module provides a complete system for:
- Loading and managing golden test datasets
- Running benchmarks against workflows
- Analyzing failure patterns
- Generating improvement hypotheses
- Tracking experiments over time

Example usage:
    from policyflow.benchmark import (
        load_golden_dataset,
        SimpleBenchmarkRunner,
        BenchmarkConfig,
        create_analyzer,
        create_hypothesis_generator,
        FileBasedExperimentTracker,
    )

    # Load test data
    dataset = load_golden_dataset("golden_dataset.yaml")
    workflow = ParsedWorkflowPolicy.load_yaml("workflow.yaml")

    # Run benchmark
    runner = SimpleBenchmarkRunner(BenchmarkConfig(workflow_id="v1"))
    report = runner.run(workflow, dataset.test_cases)
    print(f"Accuracy: {report.metrics.overall_accuracy:.1%}")

    # Analyze failures
    analyzer = create_analyzer(mode="hybrid")
    analysis = analyzer.analyze(report, workflow)

    # Generate hypotheses
    generator = create_hypothesis_generator(mode="hybrid")
    hypotheses = generator.generate(analysis, workflow)

    # Track experiment
    tracker = FileBasedExperimentTracker("experiments/")
    tracker.record(Experiment(id="exp_001", ...))
"""

from policyflow.benchmark.models import (
    AnalysisReport,
    BenchmarkMetrics,
    BenchmarkReport,
    ConfidenceCalibration,
    ConfusionMatrix,
    CriterionExpectation,
    CriterionMetrics,
    Experiment,
    ExpectedResult,
    FailurePattern,
    GenerationMetadata,
    GeneratorConfig,
    GoldenDataset,
    GoldenTestCase,
    Hypothesis,
    IntermediateState,
    OptimizationBudget,
    OptimizationResult,
    OptimizationStep,
    ProblematicCriterion,
    TestCaseResult,
)
from policyflow.benchmark.protocols import (
    BenchmarkRunner,
    ComparisonResult,
    DatasetGenerator,
    ExperimentTracker,
    FailureAnalyzer,
    HypothesisApplier,
    HypothesisGenerator,
    MetricsCalculator,
    Optimizer,
    ResultComparator,
)

# Implementations
from policyflow.benchmark.loader import load_golden_dataset, load_test_cases
from policyflow.benchmark.runner import BenchmarkConfig, SimpleBenchmarkRunner
from policyflow.benchmark.comparator import SimpleResultComparator
from policyflow.benchmark.metrics import SimpleMetricsCalculator
from policyflow.benchmark.analyzer import (
    RuleBasedAnalyzer,
    LLMEnhancedAnalyzer,
    create_analyzer,
)
from policyflow.benchmark.hypothesis import (
    TemplateBasedHypothesisGenerator,
    LLMHypothesisGenerator,
    create_hypothesis_generator,
)
from policyflow.benchmark.tracker import FileBasedExperimentTracker
from policyflow.benchmark.generator import (
    TemplateBasedGenerator,
    HybridDatasetGenerator,
    create_generator,
)
from policyflow.benchmark.applier import BasicHypothesisApplier, create_applier
from policyflow.benchmark.optimizer import (
    ConvergenceTester,
    HillClimbingOptimizer,
    create_optimizer,
)

__all__ = [
    # Test case models
    "GoldenTestCase",
    "ExpectedResult",
    "CriterionExpectation",
    "IntermediateState",
    "GoldenDataset",
    "GeneratorConfig",
    "GenerationMetadata",
    # Benchmark result models
    "TestCaseResult",
    "BenchmarkReport",
    "BenchmarkMetrics",
    "CriterionMetrics",
    "ConfusionMatrix",
    "ConfidenceCalibration",
    # Analysis models
    "AnalysisReport",
    "FailurePattern",
    "ProblematicCriterion",
    "Hypothesis",
    # Optimization models
    "OptimizationBudget",
    "OptimizationResult",
    "OptimizationStep",
    # Experiment tracking
    "Experiment",
    # Protocols
    "BenchmarkRunner",
    "ResultComparator",
    "ComparisonResult",
    "MetricsCalculator",
    "FailureAnalyzer",
    "HypothesisGenerator",
    "HypothesisApplier",
    "ExperimentTracker",
    "DatasetGenerator",
    "Optimizer",
    # Loader functions
    "load_golden_dataset",
    "load_test_cases",
    # Runner implementations
    "BenchmarkConfig",
    "SimpleBenchmarkRunner",
    # Comparator implementations
    "SimpleResultComparator",
    # Metrics implementations
    "SimpleMetricsCalculator",
    # Analyzer implementations
    "RuleBasedAnalyzer",
    "LLMEnhancedAnalyzer",
    "create_analyzer",
    # Hypothesis implementations
    "TemplateBasedHypothesisGenerator",
    "LLMHypothesisGenerator",
    "create_hypothesis_generator",
    # Tracker implementations
    "FileBasedExperimentTracker",
    # Generator implementations
    "TemplateBasedGenerator",
    "HybridDatasetGenerator",
    "create_generator",
    # Applier implementations
    "BasicHypothesisApplier",
    "create_applier",
    # Optimizer implementations
    "ConvergenceTester",
    "HillClimbingOptimizer",
    "create_optimizer",
]
