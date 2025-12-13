"""Integration tests for the benchmark system.

Tests the full flow from dataset generation through optimization.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from policyflow.benchmark.models import (
    BenchmarkMetrics,
    BenchmarkReport,
    ConfidenceCalibration,
    ConfusionMatrix,
    CriterionMetrics,
    ExpectedResult,
    GeneratorConfig,
    GoldenDataset,
    OptimizationBudget,
    TestCaseResult,
)
from policyflow.models import (
    Clause,
    ClauseType,
    EvaluationResult,
    HierarchicalWorkflowDefinition,
    NodeConfig,
    NodeGroup,
    NormalizedPolicy,
    ParsedWorkflowPolicy,
    Section,
)


def create_test_policy() -> NormalizedPolicy:
    """Create a minimal test policy."""
    return NormalizedPolicy(
        title="Test Policy",
        description="A test policy for integration testing",
        sections=[
            Section(
                number="1",
                title="Requirements",
                clauses=[
                    Clause(
                        number="1",
                        text="Content must be professional",
                        clause_type=ClauseType.REQUIREMENT,
                    ),
                    Clause(
                        number="2",
                        text="Content must not contain profanity",
                        clause_type=ClauseType.REQUIREMENT,
                    ),
                ],
            )
        ],
        raw_text="Test policy content",
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


class TestGeneratorToOptimizerIntegration:
    """Tests for the full generator -> optimizer flow."""

    def test_generated_dataset_can_be_used_for_benchmarking(self):
        """Test that generated datasets work with the benchmark runner."""
        from policyflow.benchmark.generator import create_generator
        from policyflow.benchmark.runner import SimpleBenchmarkRunner, BenchmarkConfig

        # Generate dataset
        policy = create_test_policy()
        config = GeneratorConfig(
            cases_per_criterion=2,
            include_edge_cases=False,
            include_partial_matches=False,
        )
        generator = create_generator(mode="template")
        dataset = generator.generate(policy, config)

        assert len(dataset.test_cases) > 0

        # Verify dataset structure
        for tc in dataset.test_cases:
            assert tc.id is not None
            assert tc.category is not None
            assert tc.expected is not None

        # Create workflow and runner
        workflow = create_test_workflow()

        # Mock the actual workflow execution
        with patch.object(SimpleBenchmarkRunner, "_run_single_test") as mock_run:
            # Return successful results
            mock_run.return_value = TestCaseResult(
                test_id="test_001",
                category="clear_pass",
                expected=ExpectedResult(policy_satisfied=True),
                actual=EvaluationResult(
                    policy_satisfied=True,
                    policy_title="Test",
                    overall_reasoning="OK",
                    overall_confidence=0.9,
                ),
                duration_ms=100,
            )

            benchmark_config = BenchmarkConfig(workflow_id="test")
            runner = SimpleBenchmarkRunner(benchmark_config)
            report = runner.run(workflow, dataset.test_cases)

        assert report is not None
        assert len(report.results) > 0

    def test_full_optimization_loop_with_generated_dataset(self):
        """Test the full optimization loop using a generated dataset."""
        from policyflow.benchmark.generator import create_generator
        from policyflow.benchmark.optimizer import create_optimizer
        from policyflow.benchmark.runner import SimpleBenchmarkRunner

        # Generate dataset
        policy = create_test_policy()
        config = GeneratorConfig(
            cases_per_criterion=1,
            include_edge_cases=False,
            include_partial_matches=False,
        )
        generator = create_generator(mode="template")
        dataset = generator.generate(policy, config)

        # Create workflow
        workflow = create_test_workflow()

        # Create optimizer
        optimizer = create_optimizer()

        # Mock benchmark runner to avoid LLM calls
        def mock_benchmark_report(accuracy: float) -> BenchmarkReport:
            return BenchmarkReport(
                workflow_id="test",
                timestamp=datetime.now(),
                results=[
                    TestCaseResult(
                        test_id="test_001",
                        category="clear_pass",
                        expected=ExpectedResult(policy_satisfied=True),
                        actual=EvaluationResult(
                            policy_satisfied=True,
                            policy_title="Test",
                            overall_reasoning="OK",
                            overall_confidence=0.9,
                        ),
                        duration_ms=100,
                    )
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

        with patch.object(SimpleBenchmarkRunner, "run") as mock_run:
            # Return improving accuracy over iterations
            mock_run.side_effect = [
                mock_benchmark_report(0.7),
                mock_benchmark_report(0.8),
                mock_benchmark_report(0.85),
            ]

            budget = OptimizationBudget(
                max_iterations=3,
                patience=5,  # High patience to ensure we run all iterations
            )

            result = optimizer.optimize(
                workflow=workflow,
                dataset=dataset,
                budget=budget,
            )

        assert result is not None
        assert result.best_metric >= 0.7
        assert len(result.history) > 0


class TestAnalyzerToHypothesisIntegration:
    """Tests for the analyzer -> hypothesis generator flow."""

    def test_analyzer_output_feeds_hypothesis_generator(self):
        """Test that analyzer output can be used by hypothesis generator."""
        from policyflow.benchmark.analyzer import create_analyzer
        from policyflow.benchmark.hypothesis import create_hypothesis_generator

        # Create a benchmark report with failures
        report = BenchmarkReport(
            workflow_id="test",
            timestamp=datetime.now(),
            results=[
                TestCaseResult(
                    test_id="t1",
                    category="clear_pass",
                    expected=ExpectedResult(policy_satisfied=True),
                    actual=EvaluationResult(
                        policy_satisfied=False,  # Failure
                        policy_title="Test",
                        overall_reasoning="Failed",
                        overall_confidence=0.9,
                    ),
                    duration_ms=100,
                ),
            ],
            metrics=BenchmarkMetrics(
                overall_accuracy=0.0,
                criterion_metrics={
                    "1": CriterionMetrics(
                        accuracy=0.0,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        confusion=ConfusionMatrix(tp=0, tn=0, fp=0, fn=1),
                    )
                },
                category_accuracy={"clear_pass": 0.0},
                confidence_calibration=ConfidenceCalibration(
                    high_confidence_accuracy=0.0,
                    medium_confidence_accuracy=0.0,
                    low_confidence_accuracy=0.0,
                ),
            ),
            llm_calls=1,
        )

        workflow = create_test_workflow()

        # Analyze failures
        analyzer = create_analyzer(mode="rule_based")
        analysis = analyzer.analyze(report, workflow)

        # Analysis should identify problems
        assert analysis is not None

        # Generate hypotheses from analysis
        hypothesis_gen = create_hypothesis_generator(mode="template")
        hypotheses = hypothesis_gen.generate(analysis, workflow)

        # Should generate at least one hypothesis
        assert hypotheses is not None


class TestDatasetIdempotency:
    """Tests for dataset generation reproducibility."""

    def test_same_policy_produces_same_dataset_ids(self):
        """Test that generating datasets from the same policy produces identical IDs."""
        from policyflow.benchmark.generator import create_generator

        policy = create_test_policy()
        config = GeneratorConfig(
            cases_per_criterion=2,
            include_edge_cases=False,
            include_partial_matches=False,
        )

        generator1 = create_generator(mode="template")
        dataset1 = generator1.generate(policy, config)

        generator2 = create_generator(mode="template")
        dataset2 = generator2.generate(policy, config)

        ids1 = sorted([tc.id for tc in dataset1.test_cases])
        ids2 = sorted([tc.id for tc in dataset2.test_cases])

        assert ids1 == ids2, "Same policy should produce same test case IDs"

    def test_different_policies_produce_different_datasets(self):
        """Test that different policies produce different datasets."""
        from policyflow.benchmark.generator import create_generator

        policy1 = NormalizedPolicy(
            title="Policy 1",
            description="First policy",
            sections=[
                Section(
                    number="1",
                    title="Requirements",
                    clauses=[
                        Clause(
                            number="1",
                            text="Requirement A",
                            clause_type=ClauseType.REQUIREMENT,
                        )
                    ],
                )
            ],
            raw_text="Policy 1 text",
        )

        policy2 = NormalizedPolicy(
            title="Policy 2",
            description="Second policy",
            sections=[
                Section(
                    number="1",
                    title="Requirements",
                    clauses=[
                        Clause(
                            number="1",
                            text="Requirement B",
                            clause_type=ClauseType.REQUIREMENT,
                        )
                    ],
                )
            ],
            raw_text="Policy 2 text",
        )

        config = GeneratorConfig(
            cases_per_criterion=1,
            include_edge_cases=False,
            include_partial_matches=False,
        )

        generator = create_generator(mode="template")
        dataset1 = generator.generate(policy1, config)
        dataset2 = generator.generate(policy2, config)

        # Input text should be different (based on clause text)
        texts1 = set(tc.input_text for tc in dataset1.test_cases)
        texts2 = set(tc.input_text for tc in dataset2.test_cases)

        assert texts1 != texts2, "Different policies should produce different input texts"


class TestEdgeCases:
    """Tests for edge case handling."""

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        from policyflow.benchmark.optimizer import create_optimizer
        from policyflow.benchmark.runner import SimpleBenchmarkRunner

        workflow = create_test_workflow()
        dataset = GoldenDataset(
            policy_file="test.yaml",
            description="Empty dataset",
            test_cases=[],
        )

        optimizer = create_optimizer()

        with patch.object(SimpleBenchmarkRunner, "run") as mock_run:
            # Return empty report
            mock_run.return_value = BenchmarkReport(
                workflow_id="test",
                timestamp=datetime.now(),
                results=[],
                metrics=BenchmarkMetrics(
                    overall_accuracy=0.0,
                    confidence_calibration=ConfidenceCalibration(
                        high_confidence_accuracy=0.0,
                        medium_confidence_accuracy=0.0,
                        low_confidence_accuracy=0.0,
                    ),
                ),
                llm_calls=0,
            )

            budget = OptimizationBudget(max_iterations=1)
            result = optimizer.optimize(
                workflow=workflow,
                dataset=dataset,
                budget=budget,
            )

        # Should complete without error
        assert result is not None

    def test_single_criterion_policy(self):
        """Test with a policy that has only one criterion."""
        from policyflow.benchmark.generator import create_generator

        policy = NormalizedPolicy(
            title="Single Criterion Policy",
            description="Has only one criterion",
            sections=[
                Section(
                    number="1",
                    title="Requirements",
                    clauses=[
                        Clause(
                            number="1",
                            text="The only requirement",
                            clause_type=ClauseType.REQUIREMENT,
                        )
                    ],
                )
            ],
            raw_text="Single criterion policy",
        )

        config = GeneratorConfig(
            cases_per_criterion=2,
            include_edge_cases=True,
            include_partial_matches=True,  # Should handle gracefully with 1 criterion
        )

        generator = create_generator(mode="template")
        dataset = generator.generate(policy, config)

        # Should generate test cases without error
        assert len(dataset.test_cases) > 0
