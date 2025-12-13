"""Tests for benchmark data models."""

from datetime import datetime

import pytest


class TestGoldenTestCase:
    """Tests for GoldenTestCase and related models."""

    def test_create_criterion_expectation(self):
        from policyflow.benchmark.models import CriterionExpectation

        expectation = CriterionExpectation(met=True)
        assert expectation.met is True
        assert expectation.sub_results is None

    def test_create_criterion_expectation_with_sub_results(self):
        from policyflow.benchmark.models import CriterionExpectation

        sub_results = {
            "criterion_1a": CriterionExpectation(met=True),
            "criterion_1b": CriterionExpectation(met=False),
        }
        expectation = CriterionExpectation(met=True, sub_results=sub_results)
        assert expectation.met is True
        assert expectation.sub_results["criterion_1a"].met is True
        assert expectation.sub_results["criterion_1b"].met is False

    def test_create_expected_result(self):
        from policyflow.benchmark.models import CriterionExpectation, ExpectedResult

        result = ExpectedResult(
            policy_satisfied=True,
            criterion_results={
                "criterion_1": CriterionExpectation(met=True),
                "criterion_2": CriterionExpectation(met=True),
            },
        )
        assert result.policy_satisfied is True
        assert len(result.criterion_results) == 2

    def test_create_golden_test_case(self):
        from policyflow.benchmark.models import (
            CriterionExpectation,
            ExpectedResult,
            GoldenTestCase,
        )

        test_case = GoldenTestCase(
            id="test_001",
            name="Classic recommendation",
            input_text="I recommend you buy VTI.",
            expected=ExpectedResult(
                policy_satisfied=True,
                criterion_results={
                    "criterion_1": CriterionExpectation(met=True),
                },
            ),
            category="clear_pass",
            notes="Test case notes",
        )

        assert test_case.id == "test_001"
        assert test_case.name == "Classic recommendation"
        assert test_case.expected.policy_satisfied is True
        assert test_case.category == "clear_pass"

    def test_golden_test_case_with_intermediate_expectations(self):
        from policyflow.benchmark.models import (
            CriterionExpectation,
            ExpectedResult,
            GoldenTestCase,
            IntermediateState,
        )

        test_case = GoldenTestCase(
            id="test_001",
            name="Test with intermediates",
            input_text="Some text",
            expected=ExpectedResult(
                policy_satisfied=True,
                criterion_results={
                    "criterion_1": CriterionExpectation(met=True),
                },
            ),
            category="edge_case",
            notes="",
            intermediate_expectations={
                "criterion_1a": IntermediateState(
                    clause_id="criterion_1a",
                    expected_met=True,
                    key_signals=["signal1", "signal2"],
                    reasoning="Test reasoning",
                ),
            },
        )

        assert test_case.intermediate_expectations is not None
        assert test_case.intermediate_expectations["criterion_1a"].expected_met is True


class TestGoldenDataset:
    """Tests for GoldenDataset model."""

    def test_create_golden_dataset(self):
        from policyflow.benchmark.models import (
            CriterionExpectation,
            ExpectedResult,
            GenerationMetadata,
            GeneratorConfig,
            GoldenDataset,
            GoldenTestCase,
        )

        dataset = GoldenDataset(
            policy_file="policy.md",
            description="Test dataset",
            test_cases=[
                GoldenTestCase(
                    id="test_001",
                    name="Test",
                    input_text="Input",
                    expected=ExpectedResult(
                        policy_satisfied=True,
                        criterion_results={"criterion_1": CriterionExpectation(met=True)},
                    ),
                    category="clear_pass",
                    notes="",
                )
            ],
            generation_metadata=GenerationMetadata(
                generator_version="1.0",
                config_used=GeneratorConfig(),
                timestamp=datetime.now(),
                policy_hash="abc123",
            ),
        )

        assert dataset.policy_file == "policy.md"
        assert len(dataset.test_cases) == 1

    def test_golden_dataset_yaml_serialization(self):
        from policyflow.benchmark.models import (
            CriterionExpectation,
            ExpectedResult,
            GoldenDataset,
            GoldenTestCase,
        )

        dataset = GoldenDataset(
            policy_file="policy.md",
            description="Test dataset",
            test_cases=[
                GoldenTestCase(
                    id="test_001",
                    name="Test",
                    input_text="Input",
                    expected=ExpectedResult(
                        policy_satisfied=True,
                        criterion_results={"c1": CriterionExpectation(met=True)},
                    ),
                    category="clear_pass",
                    notes="",
                )
            ],
        )

        yaml_str = dataset.to_yaml()
        loaded = GoldenDataset.from_yaml(yaml_str)
        assert loaded.policy_file == "policy.md"
        assert len(loaded.test_cases) == 1


class TestBenchmarkResults:
    """Tests for benchmark result models."""

    def test_create_confusion_matrix(self):
        from policyflow.benchmark.models import ConfusionMatrix

        matrix = ConfusionMatrix(tp=10, tn=20, fp=2, fn=3)
        assert matrix.tp == 10
        assert matrix.tn == 20
        assert matrix.fp == 2
        assert matrix.fn == 3

    def test_confusion_matrix_metrics(self):
        from policyflow.benchmark.models import ConfusionMatrix

        matrix = ConfusionMatrix(tp=10, tn=20, fp=2, fn=3)
        assert matrix.accuracy == pytest.approx((10 + 20) / 35)
        assert matrix.precision == pytest.approx(10 / 12)
        assert matrix.recall == pytest.approx(10 / 13)

    def test_confusion_matrix_edge_cases(self):
        from policyflow.benchmark.models import ConfusionMatrix

        # All zeros - should handle division by zero
        matrix = ConfusionMatrix(tp=0, tn=0, fp=0, fn=0)
        assert matrix.accuracy == 0.0
        assert matrix.precision == 0.0
        assert matrix.recall == 0.0

    def test_create_criterion_metrics(self):
        from policyflow.benchmark.models import ConfusionMatrix, CriterionMetrics

        metrics = CriterionMetrics(
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1=0.865,
            confusion=ConfusionMatrix(tp=10, tn=20, fp=2, fn=3),
        )
        assert metrics.accuracy == 0.9
        assert metrics.f1 == 0.865

    def test_create_confidence_calibration(self):
        from policyflow.benchmark.models import ConfidenceCalibration

        calibration = ConfidenceCalibration(
            high_confidence_accuracy=0.95,
            medium_confidence_accuracy=0.80,
            low_confidence_accuracy=0.60,
        )
        assert calibration.high_confidence_accuracy == 0.95

    def test_create_benchmark_metrics(self):
        from policyflow.benchmark.models import (
            BenchmarkMetrics,
            ConfidenceCalibration,
            ConfusionMatrix,
            CriterionMetrics,
        )

        metrics = BenchmarkMetrics(
            overall_accuracy=0.85,
            criterion_metrics={
                "criterion_1": CriterionMetrics(
                    accuracy=0.9,
                    precision=0.85,
                    recall=0.88,
                    f1=0.865,
                    confusion=ConfusionMatrix(tp=10, tn=20, fp=2, fn=3),
                )
            },
            category_accuracy={"clear_pass": 0.95, "clear_fail": 0.90},
            confidence_calibration=ConfidenceCalibration(
                high_confidence_accuracy=0.95,
                medium_confidence_accuracy=0.80,
                low_confidence_accuracy=0.60,
            ),
        )
        assert metrics.overall_accuracy == 0.85
        assert "criterion_1" in metrics.criterion_metrics

    def test_create_test_case_result(self):
        from policyflow.benchmark.models import (
            CriterionExpectation,
            ExpectedResult,
            TestCaseResult,
        )
        from policyflow.models import ClauseResult, EvaluationResult

        expected = ExpectedResult(
            policy_satisfied=True,
            criterion_results={"criterion_1": CriterionExpectation(met=True)},
        )

        actual = EvaluationResult(
            policy_satisfied=True,
            policy_title="Test Policy",
            clause_results=[
                ClauseResult(
                    clause_id="criterion_1",
                    clause_name="Criterion 1",
                    met=True,
                    reasoning="Test",
                    confidence=0.9,
                )
            ],
            overall_reasoning="Pass",
            overall_confidence=0.9,
        )

        result = TestCaseResult(
            test_id="test_001",
            expected=expected,
            actual=actual,
            error=None,
            duration_ms=150.5,
        )

        assert result.test_id == "test_001"
        assert result.actual is not None
        assert result.duration_ms == 150.5

    def test_create_test_case_result_with_error(self):
        from policyflow.benchmark.models import (
            CriterionExpectation,
            ExpectedResult,
            TestCaseResult,
        )

        result = TestCaseResult(
            test_id="test_001",
            expected=ExpectedResult(
                policy_satisfied=True,
                criterion_results={"c1": CriterionExpectation(met=True)},
            ),
            actual=None,
            error="Connection failed",
            duration_ms=50.0,
        )

        assert result.actual is None
        assert result.error == "Connection failed"

    def test_create_benchmark_report(self):
        from datetime import datetime

        from policyflow.benchmark.models import (
            BenchmarkMetrics,
            BenchmarkReport,
            ConfidenceCalibration,
            CriterionExpectation,
            ExpectedResult,
            TestCaseResult,
        )
        from policyflow.models import ClauseResult, EvaluationResult

        report = BenchmarkReport(
            workflow_id="workflow_v1",
            timestamp=datetime.now(),
            results=[
                TestCaseResult(
                    test_id="test_001",
                    expected=ExpectedResult(
                        policy_satisfied=True,
                        criterion_results={"c1": CriterionExpectation(met=True)},
                    ),
                    actual=EvaluationResult(
                        policy_satisfied=True,
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
                        overall_reasoning="OK",
                        overall_confidence=0.9,
                    ),
                    error=None,
                    duration_ms=100.0,
                )
            ],
            metrics=BenchmarkMetrics(
                overall_accuracy=0.85,
                criterion_metrics={},
                category_accuracy={},
                confidence_calibration=ConfidenceCalibration(
                    high_confidence_accuracy=0.9,
                    medium_confidence_accuracy=0.8,
                    low_confidence_accuracy=0.6,
                ),
            ),
            config={"setting": "value"},
        )

        assert report.workflow_id == "workflow_v1"
        assert len(report.results) == 1


class TestAnalysisModels:
    """Tests for analysis and hypothesis models."""

    def test_create_failure_pattern(self):
        from policyflow.benchmark.models import FailurePattern

        pattern = FailurePattern(
            pattern_type="category_cluster",
            description="High failure rate in edge cases",
            affected_tests=["test_001", "test_002"],
            severity="high",
        )
        assert pattern.pattern_type == "category_cluster"
        assert pattern.severity == "high"

    def test_create_problematic_criterion(self):
        from policyflow.benchmark.models import ProblematicCriterion

        criterion = ProblematicCriterion(
            criterion_id="criterion_3",
            failure_rate=0.45,
            false_positive_rate=0.30,
            false_negative_rate=0.15,
            common_failure_patterns=["Implicit language causes misclassification"],
        )
        assert criterion.criterion_id == "criterion_3"
        assert criterion.failure_rate == 0.45

    def test_create_analysis_report(self):
        from policyflow.benchmark.models import (
            AnalysisReport,
            FailurePattern,
            ProblematicCriterion,
        )

        report = AnalysisReport(
            patterns=[
                FailurePattern(
                    pattern_type="category_cluster",
                    description="Test",
                    affected_tests=["test_001"],
                    severity="high",
                )
            ],
            problematic_criteria=[
                ProblematicCriterion(
                    criterion_id="c1",
                    failure_rate=0.3,
                    false_positive_rate=0.2,
                    false_negative_rate=0.1,
                    common_failure_patterns=[],
                )
            ],
            recommendations=["Improve prompt clarity"],
        )
        assert len(report.patterns) == 1
        assert len(report.recommendations) == 1

    def test_create_hypothesis(self):
        from policyflow.benchmark.models import Hypothesis

        hypothesis = Hypothesis(
            id="hyp_001",
            description="Clarify criterion 3 prompt",
            change_type="prompt_tuning",
            target="criterion_3",
            suggested_change={"prompt": "New prompt text"},
            rationale="Current prompt is ambiguous",
            expected_impact="Improve accuracy by 10%",
        )
        assert hypothesis.id == "hyp_001"
        assert hypothesis.change_type == "prompt_tuning"


class TestOptimizationModels:
    """Tests for optimization-related models."""

    def test_create_optimization_budget(self):
        from policyflow.benchmark.models import OptimizationBudget

        budget = OptimizationBudget(
            max_iterations=10,
            max_llm_calls=100,
            max_time_seconds=3600.0,
            target_metric=0.95,
            min_improvement=0.01,
            patience=3,
        )
        assert budget.max_iterations == 10
        assert budget.target_metric == 0.95

    def test_optimization_budget_defaults(self):
        from policyflow.benchmark.models import OptimizationBudget

        budget = OptimizationBudget()
        assert budget.max_iterations == 10
        assert budget.patience == 3

    def test_create_optimization_step(self):
        from policyflow.benchmark.models import OptimizationStep

        step = OptimizationStep(
            iteration=1,
            workflow_snapshot="workflow: yaml here",
            metric=0.85,
            changes_made=["Adjusted threshold"],
            llm_calls=5,
        )
        assert step.iteration == 1
        assert step.metric == 0.85

    def test_create_optimization_result(self):
        from policyflow.benchmark.models import OptimizationResult, OptimizationStep

        result = OptimizationResult(
            best_workflow_yaml="workflow: yaml",
            best_metric=0.92,
            history=[
                OptimizationStep(
                    iteration=1,
                    workflow_snapshot="v1",
                    metric=0.85,
                    changes_made=[],
                    llm_calls=5,
                ),
                OptimizationStep(
                    iteration=2,
                    workflow_snapshot="v2",
                    metric=0.92,
                    changes_made=["Change 1"],
                    llm_calls=8,
                ),
            ],
            converged=True,
            convergence_reason="target_reached",
            total_llm_calls=13,
            total_time_seconds=120.5,
        )
        assert result.best_metric == 0.92
        assert result.converged is True


class TestExperimentTracking:
    """Tests for experiment tracking models."""

    def test_create_experiment(self):
        from datetime import datetime

        from policyflow.benchmark.models import (
            BenchmarkMetrics,
            BenchmarkReport,
            ConfidenceCalibration,
            Experiment,
            Hypothesis,
        )

        experiment = Experiment(
            id="exp_001",
            timestamp=datetime.now(),
            workflow_snapshot="workflow: yaml",
            hypothesis_applied=Hypothesis(
                id="hyp_001",
                description="Test change",
                change_type="node_param",
                target="node_1",
                suggested_change={"threshold": 0.7},
                rationale="Improve precision",
                expected_impact="Better accuracy",
            ),
            benchmark_report=BenchmarkReport(
                workflow_id="v1",
                timestamp=datetime.now(),
                results=[],
                metrics=BenchmarkMetrics(
                    overall_accuracy=0.85,
                    criterion_metrics={},
                    category_accuracy={},
                    confidence_calibration=ConfidenceCalibration(
                        high_confidence_accuracy=0.9,
                        medium_confidence_accuracy=0.8,
                        low_confidence_accuracy=0.6,
                    ),
                ),
                config={},
            ),
            parent_experiment_id=None,
        )

        assert experiment.id == "exp_001"
        assert experiment.hypothesis_applied is not None

    def test_experiment_yaml_serialization(self):
        from datetime import datetime

        from policyflow.benchmark.models import (
            BenchmarkMetrics,
            BenchmarkReport,
            ConfidenceCalibration,
            Experiment,
        )

        experiment = Experiment(
            id="exp_001",
            timestamp=datetime.now(),
            workflow_snapshot="workflow: yaml",
            hypothesis_applied=None,
            benchmark_report=BenchmarkReport(
                workflow_id="v1",
                timestamp=datetime.now(),
                results=[],
                metrics=BenchmarkMetrics(
                    overall_accuracy=0.85,
                    criterion_metrics={},
                    category_accuracy={},
                    confidence_calibration=ConfidenceCalibration(
                        high_confidence_accuracy=0.9,
                        medium_confidence_accuracy=0.8,
                        low_confidence_accuracy=0.6,
                    ),
                ),
                config={},
            ),
            parent_experiment_id=None,
        )

        yaml_str = experiment.to_yaml()
        loaded = Experiment.from_yaml(yaml_str)
        assert loaded.id == "exp_001"


class TestGeneratorConfig:
    """Tests for generator configuration model."""

    def test_generator_config_defaults(self):
        from policyflow.benchmark.models import GeneratorConfig

        config = GeneratorConfig()
        assert config.cases_per_criterion == 3
        assert config.include_edge_cases is True
        assert config.mode == "hybrid"

    def test_generator_config_custom(self):
        from policyflow.benchmark.models import GeneratorConfig

        config = GeneratorConfig(
            cases_per_criterion=5,
            include_edge_cases=False,
            edge_case_strategies=["boundary", "negation"],
            mode="llm",
            temperature=0.9,
        )
        assert config.cases_per_criterion == 5
        assert config.mode == "llm"
        assert "boundary" in config.edge_case_strategies
