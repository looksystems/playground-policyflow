"""Tests for benchmark runner."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from policyflow.benchmark.models import (
    BenchmarkMetrics,
    ConfidenceCalibration,
    CriterionExpectation,
    ExpectedResult,
    GoldenTestCase,
)
from policyflow.models import ClauseResult, EvaluationResult, ParsedWorkflowPolicy


class TestSimpleBenchmarkRunner:
    """Tests for SimpleBenchmarkRunner."""

    def test_runner_initialization(self):
        from policyflow.benchmark.runner import SimpleBenchmarkRunner, BenchmarkConfig

        config = BenchmarkConfig(workflow_id="test_v1")
        runner = SimpleBenchmarkRunner(config)
        assert runner.config.workflow_id == "test_v1"

    def test_run_single_test_case(self):
        from policyflow.benchmark.runner import SimpleBenchmarkRunner, BenchmarkConfig

        # Create a mock workflow
        mock_shared = {
            "policy_satisfied": True,
            "clause_results": [
                {
                    "clause_id": "c1",
                    "clause_name": "Criterion 1",
                    "met": True,
                    "reasoning": "Test reasoning",
                    "confidence": 0.9,
                }
            ],
            "overall_reasoning": "Pass",
            "overall_confidence": 0.9,
            "policy_title": "Test Policy",
        }

        mock_builder = MagicMock()
        mock_builder.run.return_value = mock_shared

        # Create test case
        test_case = GoldenTestCase(
            id="test_001",
            name="Test case",
            input_text="Test input",
            expected=ExpectedResult(
                policy_satisfied=True,
                criterion_results={"c1": CriterionExpectation(met=True)},
            ),
            category="clear_pass",
            notes="",
        )

        runner = SimpleBenchmarkRunner(BenchmarkConfig(workflow_id="test"))

        # Run single test
        with patch(
            "policyflow.benchmark.runner.DynamicWorkflowBuilder",
            return_value=mock_builder,
        ):
            result = runner._run_single_test(mock_builder, test_case)

        assert result.test_id == "test_001"
        assert result.actual is not None
        assert result.actual.policy_satisfied is True
        assert result.error is None

    def test_run_test_case_with_error(self):
        from policyflow.benchmark.runner import SimpleBenchmarkRunner, BenchmarkConfig

        mock_builder = MagicMock()
        mock_builder.run.side_effect = RuntimeError("Workflow failed")

        test_case = GoldenTestCase(
            id="test_001",
            name="Test case",
            input_text="Test input",
            expected=ExpectedResult(
                policy_satisfied=True,
                criterion_results={},
            ),
            category="clear_pass",
            notes="",
        )

        runner = SimpleBenchmarkRunner(BenchmarkConfig(workflow_id="test"))
        result = runner._run_single_test(mock_builder, test_case)

        assert result.test_id == "test_001"
        assert result.actual is None
        assert result.error is not None
        assert "Workflow failed" in result.error

    def test_run_full_benchmark(self):
        from policyflow.benchmark.runner import SimpleBenchmarkRunner, BenchmarkConfig

        # Mock workflow results
        mock_shared_1 = {
            "policy_satisfied": True,
            "clause_results": [
                {
                    "clause_id": "c1",
                    "clause_name": "C1",
                    "met": True,
                    "reasoning": "R",
                    "confidence": 0.9,
                }
            ],
            "overall_reasoning": "Pass",
            "overall_confidence": 0.9,
            "policy_title": "Test",
        }
        mock_shared_2 = {
            "policy_satisfied": False,
            "clause_results": [
                {
                    "clause_id": "c1",
                    "clause_name": "C1",
                    "met": False,
                    "reasoning": "R",
                    "confidence": 0.85,
                }
            ],
            "overall_reasoning": "Fail",
            "overall_confidence": 0.85,
            "policy_title": "Test",
        }

        mock_builder = MagicMock()
        mock_builder.run.side_effect = [mock_shared_1, mock_shared_2]

        test_cases = [
            GoldenTestCase(
                id="test_001",
                name="Pass case",
                input_text="Input 1",
                expected=ExpectedResult(
                    policy_satisfied=True,
                    criterion_results={"c1": CriterionExpectation(met=True)},
                ),
                category="clear_pass",
                notes="",
            ),
            GoldenTestCase(
                id="test_002",
                name="Fail case",
                input_text="Input 2",
                expected=ExpectedResult(
                    policy_satisfied=False,
                    criterion_results={"c1": CriterionExpectation(met=False)},
                ),
                category="clear_fail",
                notes="",
            ),
        ]

        # Create mock workflow policy
        mock_policy = MagicMock(spec=ParsedWorkflowPolicy)

        runner = SimpleBenchmarkRunner(BenchmarkConfig(workflow_id="test_v1"))

        with patch(
            "policyflow.benchmark.runner.DynamicWorkflowBuilder",
            return_value=mock_builder,
        ):
            report = runner.run(mock_policy, test_cases)

        assert report.workflow_id == "test_v1"
        assert len(report.results) == 2
        assert report.metrics.overall_accuracy == 1.0  # Both match expectations


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_config(self):
        from policyflow.benchmark.runner import BenchmarkConfig

        config = BenchmarkConfig()
        assert config.workflow_id == "default"
        assert config.max_iterations == 100

    def test_custom_config(self):
        from policyflow.benchmark.runner import BenchmarkConfig

        config = BenchmarkConfig(
            workflow_id="my_workflow",
            max_iterations=50,
        )
        assert config.workflow_id == "my_workflow"
        assert config.max_iterations == 50


class TestResultExtraction:
    """Tests for extracting EvaluationResult from shared store."""

    def test_extract_evaluation_result(self):
        from policyflow.benchmark.runner import _extract_evaluation_result

        shared = {
            "policy_satisfied": True,
            "clause_results": [
                {
                    "clause_id": "c1",
                    "clause_name": "Criterion 1",
                    "met": True,
                    "reasoning": "Test",
                    "confidence": 0.9,
                    "sub_results": [
                        {
                            "clause_id": "c1a",
                            "clause_name": "Sub 1a",
                            "met": True,
                            "reasoning": "Sub test",
                            "confidence": 0.85,
                        }
                    ],
                }
            ],
            "overall_reasoning": "Pass",
            "overall_confidence": 0.9,
            "policy_title": "Test Policy",
        }

        result = _extract_evaluation_result(shared)

        assert result.policy_satisfied is True
        assert len(result.clause_results) == 1
        assert result.clause_results[0].clause_id == "c1"
        assert len(result.clause_results[0].sub_results) == 1
        assert result.clause_results[0].sub_results[0].clause_id == "c1a"

    def test_extract_result_with_missing_fields(self):
        from policyflow.benchmark.runner import _extract_evaluation_result

        # Minimal shared dict
        shared = {
            "policy_satisfied": False,
            "overall_reasoning": "Minimal",
            "overall_confidence": 0.5,
        }

        result = _extract_evaluation_result(shared)

        assert result.policy_satisfied is False
        assert len(result.clause_results) == 0
        assert result.policy_title == ""
