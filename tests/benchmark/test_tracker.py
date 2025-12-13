"""Tests for experiment tracker."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from policyflow.benchmark.models import (
    BenchmarkMetrics,
    BenchmarkReport,
    ConfidenceCalibration,
    Experiment,
    Hypothesis,
)


def make_experiment(
    exp_id: str,
    accuracy: float,
    hypothesis: Hypothesis | None = None,
    parent_id: str | None = None,
) -> Experiment:
    """Helper to create experiments."""
    return Experiment(
        id=exp_id,
        timestamp=datetime.now(),
        workflow_snapshot="workflow: yaml",
        hypothesis_applied=hypothesis,
        benchmark_report=BenchmarkReport(
            workflow_id="test",
            timestamp=datetime.now(),
            results=[],
            metrics=BenchmarkMetrics(
                overall_accuracy=accuracy,
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
        parent_experiment_id=parent_id,
    )


class TestFileBasedExperimentTracker:
    """Tests for FileBasedExperimentTracker."""

    def test_tracker_initialization(self, tmp_path):
        from policyflow.benchmark.tracker import FileBasedExperimentTracker

        tracker = FileBasedExperimentTracker(tmp_path)
        assert tracker.experiments_dir == tmp_path

    def test_record_experiment(self, tmp_path):
        from policyflow.benchmark.tracker import FileBasedExperimentTracker

        tracker = FileBasedExperimentTracker(tmp_path)
        experiment = make_experiment("exp_001", 0.85)

        tracker.record(experiment)

        # Check file was created
        exp_file = tmp_path / "exp_001.yaml"
        assert exp_file.exists()

    def test_get_history(self, tmp_path):
        from policyflow.benchmark.tracker import FileBasedExperimentTracker

        tracker = FileBasedExperimentTracker(tmp_path)

        # Record multiple experiments
        tracker.record(make_experiment("exp_001", 0.85))
        tracker.record(make_experiment("exp_002", 0.90))
        tracker.record(make_experiment("exp_003", 0.88))

        history = tracker.get_history()

        assert len(history) == 3
        # Should be sorted by timestamp
        ids = [e.id for e in history]
        assert "exp_001" in ids
        assert "exp_002" in ids
        assert "exp_003" in ids

    def test_get_best(self, tmp_path):
        from policyflow.benchmark.tracker import FileBasedExperimentTracker

        tracker = FileBasedExperimentTracker(tmp_path)

        tracker.record(make_experiment("exp_001", 0.85))
        tracker.record(make_experiment("exp_002", 0.95))  # Best
        tracker.record(make_experiment("exp_003", 0.88))

        best = tracker.get_best()

        assert best is not None
        assert best.id == "exp_002"
        assert best.accuracy == 0.95

    def test_get_best_empty(self, tmp_path):
        from policyflow.benchmark.tracker import FileBasedExperimentTracker

        tracker = FileBasedExperimentTracker(tmp_path)

        best = tracker.get_best()
        assert best is None

    def test_get_by_id(self, tmp_path):
        from policyflow.benchmark.tracker import FileBasedExperimentTracker

        tracker = FileBasedExperimentTracker(tmp_path)
        tracker.record(make_experiment("exp_001", 0.85))
        tracker.record(make_experiment("exp_002", 0.90))

        exp = tracker.get_by_id("exp_001")
        assert exp is not None
        assert exp.id == "exp_001"

        missing = tracker.get_by_id("nonexistent")
        assert missing is None

    def test_creates_directory_if_missing(self, tmp_path):
        from policyflow.benchmark.tracker import FileBasedExperimentTracker

        new_dir = tmp_path / "experiments" / "nested"
        tracker = FileBasedExperimentTracker(new_dir)

        tracker.record(make_experiment("exp_001", 0.85))

        assert new_dir.exists()
        assert (new_dir / "exp_001.yaml").exists()

    def test_experiment_with_hypothesis(self, tmp_path):
        from policyflow.benchmark.tracker import FileBasedExperimentTracker

        tracker = FileBasedExperimentTracker(tmp_path)

        hypothesis = Hypothesis(
            id="hyp_001",
            description="Test hypothesis",
            change_type="prompt_tuning",
            target="criterion_1",
            suggested_change={"prompt": "new prompt"},
            rationale="Testing",
            expected_impact="Better accuracy",
        )

        experiment = make_experiment("exp_001", 0.90, hypothesis=hypothesis)
        tracker.record(experiment)

        # Reload and verify
        loaded = tracker.get_by_id("exp_001")
        assert loaded is not None
        assert loaded.hypothesis_applied is not None
        assert loaded.hypothesis_applied.id == "hyp_001"

    def test_experiment_with_parent(self, tmp_path):
        from policyflow.benchmark.tracker import FileBasedExperimentTracker

        tracker = FileBasedExperimentTracker(tmp_path)

        # Record baseline
        tracker.record(make_experiment("baseline", 0.80))

        # Record child experiment
        child = make_experiment("exp_001", 0.85, parent_id="baseline")
        tracker.record(child)

        loaded = tracker.get_by_id("exp_001")
        assert loaded is not None
        assert loaded.parent_experiment_id == "baseline"


class TestExperimentComparison:
    """Tests for comparing experiments."""

    def test_compare_experiments(self, tmp_path):
        from policyflow.benchmark.tracker import FileBasedExperimentTracker

        tracker = FileBasedExperimentTracker(tmp_path)

        tracker.record(make_experiment("exp_001", 0.85))
        tracker.record(make_experiment("exp_002", 0.90))

        comparison = tracker.compare("exp_001", "exp_002")

        assert comparison is not None
        assert comparison["accuracy_diff"] == pytest.approx(0.05)
        assert comparison["exp_001_accuracy"] == 0.85
        assert comparison["exp_002_accuracy"] == 0.90
        assert comparison["improved"] is True


class TestMalformedFileHandling:
    """Tests for handling malformed experiment files."""

    def test_get_history_skips_malformed_files_with_warning(self, tmp_path, caplog):
        """Test that malformed files are skipped and logged."""
        import logging

        from policyflow.benchmark.tracker import FileBasedExperimentTracker

        tracker = FileBasedExperimentTracker(tmp_path)

        # Create a valid experiment
        tracker.record(make_experiment("exp_001", 0.85))

        # Create a malformed YAML file
        malformed_file = tmp_path / "malformed.yaml"
        malformed_file.write_text("not: valid: yaml: :::!")

        # Get history should skip the malformed file and log a warning
        with caplog.at_level(logging.WARNING):
            history = tracker.get_history()

        # Should only have the valid experiment
        assert len(history) == 1
        assert history[0].id == "exp_001"

        # Should have logged a warning about the malformed file
        assert "malformed.yaml" in caplog.text or "Failed to load" in caplog.text

    def test_get_by_id_returns_none_for_malformed_file(self, tmp_path, caplog):
        """Test that get_by_id returns None and logs for malformed files."""
        import logging

        from policyflow.benchmark.tracker import FileBasedExperimentTracker

        tracker = FileBasedExperimentTracker(tmp_path)

        # Create a malformed YAML file
        malformed_file = tmp_path / "exp_001.yaml"
        malformed_file.write_text("invalid yaml content :::!")

        with caplog.at_level(logging.WARNING):
            result = tracker.get_by_id("exp_001")

        assert result is None
        # Should log a warning
        assert "exp_001" in caplog.text or "Failed to load" in caplog.text
