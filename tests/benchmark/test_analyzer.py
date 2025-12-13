"""Tests for failure analyzer."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from policyflow.benchmark.models import (
    AnalysisReport,
    BenchmarkMetrics,
    BenchmarkReport,
    ConfidenceCalibration,
    ConfusionMatrix,
    CriterionExpectation,
    CriterionMetrics,
    ExpectedResult,
    TestCaseResult,
)
from policyflow.models import ClauseResult, EvaluationResult, ParsedWorkflowPolicy


def make_test_result(
    test_id: str,
    category: str,
    expected_satisfied: bool,
    actual_satisfied: bool,
    expected_criteria: dict[str, bool] | None = None,
    actual_criteria: dict[str, bool] | None = None,
    confidence: float = 0.9,
) -> TestCaseResult:
    """Helper to create test results."""
    expected_criteria = expected_criteria or {}
    actual_criteria = actual_criteria or {}

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
        category=category,
        expected=expected,
        actual=actual,
        error=None,
        duration_ms=100.0,
    )


def make_benchmark_report(
    results: list[TestCaseResult],
    category_accuracy: dict[str, float] | None = None,
    criterion_metrics: dict[str, CriterionMetrics] | None = None,
) -> BenchmarkReport:
    """Helper to create a benchmark report."""
    return BenchmarkReport(
        workflow_id="test",
        timestamp=datetime.now(),
        results=results,
        metrics=BenchmarkMetrics(
            overall_accuracy=0.8,
            criterion_metrics=criterion_metrics or {},
            category_accuracy=category_accuracy or {},
            confidence_calibration=ConfidenceCalibration(
                high_confidence_accuracy=0.9,
                medium_confidence_accuracy=0.7,
                low_confidence_accuracy=0.5,
            ),
        ),
        config={},
    )


class TestRuleBasedAnalyzer:
    """Tests for RuleBasedAnalyzer."""

    def test_analyzer_initialization(self):
        from policyflow.benchmark.analyzer import RuleBasedAnalyzer

        analyzer = RuleBasedAnalyzer()
        assert analyzer is not None

    def test_detect_category_cluster_pattern(self):
        from policyflow.benchmark.analyzer import RuleBasedAnalyzer

        # Create report with high failure rate in edge_case category
        results = [
            make_test_result("t1", "clear_pass", True, True),
            make_test_result("t2", "clear_pass", True, True),
            make_test_result("t3", "edge_case", True, False),  # Fail
            make_test_result("t4", "edge_case", True, False),  # Fail
            make_test_result("t5", "edge_case", True, False),  # Fail
        ]

        report = make_benchmark_report(
            results,
            category_accuracy={"clear_pass": 1.0, "edge_case": 0.0},
        )

        # Mock workflow
        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)

        analyzer = RuleBasedAnalyzer()
        analysis = analyzer.analyze(report, mock_workflow)

        # Should detect category cluster pattern
        category_patterns = [
            p for p in analysis.patterns if p.pattern_type == "category_cluster"
        ]
        assert len(category_patterns) >= 1
        assert any("edge_case" in p.description for p in category_patterns)

    def test_detect_systematic_criterion_failure(self):
        from policyflow.benchmark.analyzer import RuleBasedAnalyzer

        results = [
            make_test_result(
                "t1", "clear_pass", True, True, {"c1": True, "c2": True}, {"c1": True, "c2": False}
            ),
            make_test_result(
                "t2", "clear_pass", True, True, {"c1": True, "c2": True}, {"c1": True, "c2": False}
            ),
            make_test_result(
                "t3", "clear_pass", True, True, {"c1": True, "c2": True}, {"c1": True, "c2": False}
            ),
        ]

        # c2 has 0% accuracy
        report = make_benchmark_report(
            results,
            criterion_metrics={
                "c1": CriterionMetrics(
                    accuracy=1.0,
                    precision=1.0,
                    recall=1.0,
                    f1=1.0,
                    confusion=ConfusionMatrix(tp=3, tn=0, fp=0, fn=0),
                ),
                "c2": CriterionMetrics(
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    confusion=ConfusionMatrix(tp=0, tn=0, fp=0, fn=3),
                ),
            },
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)

        analyzer = RuleBasedAnalyzer()
        analysis = analyzer.analyze(report, mock_workflow)

        # Should identify c2 as problematic
        assert len(analysis.problematic_criteria) >= 1
        c2_prob = next(
            (pc for pc in analysis.problematic_criteria if pc.criterion_id == "c2"),
            None,
        )
        assert c2_prob is not None

    def test_detect_false_positive_imbalance(self):
        from policyflow.benchmark.analyzer import RuleBasedAnalyzer

        results = []

        # High FP rate for c1
        report = make_benchmark_report(
            results,
            criterion_metrics={
                "c1": CriterionMetrics(
                    accuracy=0.6,
                    precision=0.3,  # Low precision = high FP
                    recall=0.9,
                    f1=0.45,
                    confusion=ConfusionMatrix(tp=9, tn=3, fp=7, fn=1),  # 7 FP out of 20
                ),
            },
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)

        analyzer = RuleBasedAnalyzer()
        analysis = analyzer.analyze(report, mock_workflow)

        # Should detect FP imbalance
        fp_patterns = [
            p for p in analysis.patterns if "false_positive" in p.pattern_type
        ]
        assert len(fp_patterns) >= 1

    def test_detect_confidence_miscalibration(self):
        from policyflow.benchmark.analyzer import RuleBasedAnalyzer

        results = [
            make_test_result("t1", "test", True, False, confidence=0.95),  # High conf, wrong
            make_test_result("t2", "test", True, False, confidence=0.92),  # High conf, wrong
        ]

        report = BenchmarkReport(
            workflow_id="test",
            timestamp=datetime.now(),
            results=results,
            metrics=BenchmarkMetrics(
                overall_accuracy=0.0,
                criterion_metrics={},
                category_accuracy={},
                confidence_calibration=ConfidenceCalibration(
                    high_confidence_accuracy=0.0,  # High conf but 0% accuracy!
                    medium_confidence_accuracy=0.8,
                    low_confidence_accuracy=0.5,
                ),
            ),
            config={},
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)

        analyzer = RuleBasedAnalyzer()
        analysis = analyzer.analyze(report, mock_workflow)

        # Should detect miscalibration
        miscal_patterns = [
            p for p in analysis.patterns if "miscalibration" in p.pattern_type
        ]
        assert len(miscal_patterns) >= 1

    def test_generate_recommendations(self):
        from policyflow.benchmark.analyzer import RuleBasedAnalyzer

        results = [
            make_test_result("t1", "edge_case", True, False),
        ]

        report = make_benchmark_report(
            results,
            category_accuracy={"edge_case": 0.0},
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)

        analyzer = RuleBasedAnalyzer()
        analysis = analyzer.analyze(report, mock_workflow)

        # Should have recommendations
        assert len(analysis.recommendations) >= 1

    def test_get_category_returns_correct_category(self):
        """Test that _get_category returns the category from TestCaseResult."""
        from policyflow.benchmark.analyzer import RuleBasedAnalyzer

        result = make_test_result("t1", "edge_case", True, False)

        analyzer = RuleBasedAnalyzer()
        category = analyzer._get_category(result)

        assert category == "edge_case"

    def test_category_cluster_finds_affected_tests(self):
        """Test that category cluster pattern correctly identifies affected tests."""
        from policyflow.benchmark.analyzer import RuleBasedAnalyzer

        # Create results with categories - 3 edge_case failures
        results = [
            make_test_result("t1", "clear_pass", True, True),
            make_test_result("t2", "clear_pass", True, True),
            make_test_result("t3", "edge_case", True, False),  # Fail
            make_test_result("t4", "edge_case", True, False),  # Fail
            make_test_result("t5", "edge_case", True, False),  # Fail
        ]

        report = make_benchmark_report(
            results,
            category_accuracy={"clear_pass": 1.0, "edge_case": 0.0},
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)

        analyzer = RuleBasedAnalyzer()
        analysis = analyzer.analyze(report, mock_workflow)

        # Find the edge_case category cluster pattern
        category_patterns = [
            p for p in analysis.patterns
            if p.pattern_type == "category_cluster" and "edge_case" in p.description
        ]
        assert len(category_patterns) >= 1

        # The affected_tests should include the failed edge_case tests
        affected = category_patterns[0].affected_tests
        assert "t3" in affected
        assert "t4" in affected
        assert "t5" in affected
        assert "t1" not in affected  # clear_pass should not be affected


class TestAnalyzerFactory:
    """Tests for analyzer factory function."""

    def test_create_rule_based_analyzer(self):
        from policyflow.benchmark.analyzer import create_analyzer

        analyzer = create_analyzer(mode="rule_based")
        assert analyzer is not None

    def test_create_hybrid_analyzer(self):
        from policyflow.benchmark.analyzer import create_analyzer

        analyzer = create_analyzer(mode="hybrid")
        assert analyzer is not None

    def test_invalid_mode_raises(self):
        from policyflow.benchmark.analyzer import create_analyzer

        with pytest.raises(ValueError):
            create_analyzer(mode="invalid_mode")
