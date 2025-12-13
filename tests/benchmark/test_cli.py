"""Tests for benchmark CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from policyflow.benchmark.models import (
    BenchmarkMetrics,
    BenchmarkReport,
    ConfidenceCalibration,
    CriterionExpectation,
    Experiment,
    ExpectedResult,
    GoldenTestCase,
    TestCaseResult,
)
from policyflow.models import ClauseResult, EvaluationResult


runner = CliRunner()


class TestBenchmarkCLI:
    """Tests for the benchmark command."""

    def test_benchmark_command_exists(self):
        from policyflow.benchmark.cli import benchmark_app

        result = runner.invoke(benchmark_app, ["--help"])
        assert result.exit_code == 0
        assert "benchmark" in result.output.lower() or "workflow" in result.output.lower()

    def test_benchmark_requires_args(self):
        from policyflow.benchmark.cli import benchmark_app

        result = runner.invoke(benchmark_app, [])
        # Should show help or error about missing args
        assert result.exit_code != 0 or "Usage" in result.output or "Missing" in result.output


class TestAnalyzeCLI:
    """Tests for the analyze command."""

    def test_analyze_command_exists(self):
        from policyflow.benchmark.cli import analyze_app

        result = runner.invoke(analyze_app, ["--help"])
        assert result.exit_code == 0


class TestExperimentsCLI:
    """Tests for experiments commands."""

    def test_experiments_list_empty(self, tmp_path):
        from policyflow.benchmark.cli import experiments_app

        result = runner.invoke(
            experiments_app,
            ["list", "--dir", str(tmp_path)],
        )
        assert result.exit_code == 0

    def test_experiments_best_empty(self, tmp_path):
        from policyflow.benchmark.cli import experiments_app

        result = runner.invoke(
            experiments_app,
            ["best", "--dir", str(tmp_path)],
        )
        # Should handle empty gracefully
        assert result.exit_code == 0 or "No experiments" in result.output


class TestGenerateDatasetCLI:
    """Tests for the generate-dataset command."""

    def test_generate_dataset_command_exists(self):
        from policyflow.benchmark.cli import generate_dataset_app

        result = runner.invoke(generate_dataset_app, ["--help"])
        assert result.exit_code == 0
        assert "dataset" in result.output.lower() or "policy" in result.output.lower()

    def test_generate_dataset_requires_args(self):
        from policyflow.benchmark.cli import generate_dataset_app

        result = runner.invoke(generate_dataset_app, [])
        # Should show help or error about missing args
        assert result.exit_code != 0 or "Usage" in result.output or "Missing" in result.output

    def test_generate_dataset_creates_file(self, tmp_path):
        from policyflow.benchmark.cli import generate_dataset_app

        # Create a minimal normalized policy file
        policy_data = {
            "title": "Test Policy",
            "description": "Test description",
            "sections": [
                {
                    "number": "1",
                    "title": "Requirements",
                    "clauses": [
                        {
                            "number": "1",
                            "text": "Content must be professional",
                            "clause_type": "requirement",
                        }
                    ],
                }
            ],
            "raw_text": "Test policy",
        }
        policy_path = tmp_path / "policy.yaml"
        with policy_path.open("w") as f:
            yaml.dump(policy_data, f)

        output_path = tmp_path / "dataset.yaml"

        result = runner.invoke(
            generate_dataset_app,
            [
                "--policy", str(policy_path),
                "--output", str(output_path),
                "--cases-per-criterion", "1",
                "--no-edge-cases",
                "--no-partial-matches",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_path.exists(), "Dataset file was not created"


class TestOptimizeCLI:
    """Tests for the optimize command."""

    def test_optimize_command_exists(self):
        from policyflow.benchmark.cli import optimize_app

        result = runner.invoke(optimize_app, ["--help"])
        assert result.exit_code == 0
        assert "optimize" in result.output.lower() or "workflow" in result.output.lower()

    def test_optimize_requires_args(self):
        from policyflow.benchmark.cli import optimize_app

        result = runner.invoke(optimize_app, [])
        # Should show help or error about missing args
        assert result.exit_code != 0 or "Usage" in result.output or "Missing" in result.output


class TestImproveCLI:
    """Tests for the improve command."""

    def test_improve_command_exists(self):
        from policyflow.benchmark.cli import improve_app

        result = runner.invoke(improve_app, ["--help"])
        assert result.exit_code == 0
        assert "improve" in result.output.lower() or "workflow" in result.output.lower()
