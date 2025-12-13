"""Experiment tracker for recording and comparing benchmark runs."""

from __future__ import annotations

import logging
from pathlib import Path

from policyflow.benchmark.models import Experiment

logger = logging.getLogger(__name__)


class FileBasedExperimentTracker:
    """File-based experiment tracker using YAML persistence.

    Stores each experiment as a separate YAML file for easy
    inspection and version control.
    """

    def __init__(self, experiments_dir: str | Path):
        """Initialize the tracker.

        Args:
            experiments_dir: Directory to store experiment YAML files
        """
        self.experiments_dir = Path(experiments_dir)
        self._ensure_dir_exists()

    def _ensure_dir_exists(self) -> None:
        """Create experiments directory if it doesn't exist."""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def record(self, experiment: Experiment) -> None:
        """Record an experiment to disk.

        Args:
            experiment: The experiment to record
        """
        path = self.experiments_dir / f"{experiment.id}.yaml"
        experiment.save_yaml(path)

    def get_history(self) -> list[Experiment]:
        """Get all recorded experiments sorted by timestamp.

        Returns:
            List of all experiments, sorted oldest to newest
        """
        experiments = []
        for path in self.experiments_dir.glob("*.yaml"):
            try:
                experiment = Experiment.load_yaml(path)
                experiments.append(experiment)
            except Exception as e:
                # Log warning and skip malformed files
                logger.warning(f"Failed to load experiment from {path.name}: {e}")
                continue

        # Sort by timestamp
        experiments.sort(key=lambda e: e.timestamp)
        return experiments

    def get_best(self) -> Experiment | None:
        """Get the experiment with the highest accuracy.

        Returns:
            The best experiment, or None if no experiments exist
        """
        experiments = self.get_history()
        if not experiments:
            return None

        return max(experiments, key=lambda e: e.accuracy)

    def get_by_id(self, experiment_id: str) -> Experiment | None:
        """Get an experiment by its ID.

        Args:
            experiment_id: The experiment ID to find

        Returns:
            The experiment if found, None otherwise
        """
        path = self.experiments_dir / f"{experiment_id}.yaml"
        if not path.exists():
            return None

        try:
            return Experiment.load_yaml(path)
        except Exception as e:
            logger.warning(f"Failed to load experiment {experiment_id}: {e}")
            return None

    def compare(
        self, exp_id_1: str, exp_id_2: str
    ) -> dict | None:
        """Compare two experiments.

        Args:
            exp_id_1: First experiment ID
            exp_id_2: Second experiment ID

        Returns:
            Comparison dict with accuracy diff and improvement status,
            or None if either experiment not found
        """
        exp1 = self.get_by_id(exp_id_1)
        exp2 = self.get_by_id(exp_id_2)

        if exp1 is None or exp2 is None:
            return None

        acc1 = exp1.accuracy
        acc2 = exp2.accuracy
        diff = acc2 - acc1

        return {
            "exp_001_accuracy": acc1,
            "exp_002_accuracy": acc2,
            f"{exp_id_1}_accuracy": acc1,
            f"{exp_id_2}_accuracy": acc2,
            "accuracy_diff": diff,
            "improved": diff > 0,
        }

    def get_lineage(self, experiment_id: str) -> list[Experiment]:
        """Get the lineage (ancestry) of an experiment.

        Args:
            experiment_id: The experiment to trace back

        Returns:
            List of experiments from root to the given experiment
        """
        lineage = []
        current = self.get_by_id(experiment_id)

        while current is not None:
            lineage.insert(0, current)
            if current.parent_experiment_id:
                current = self.get_by_id(current.parent_experiment_id)
            else:
                current = None

        return lineage

    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment.

        Args:
            experiment_id: The experiment ID to delete

        Returns:
            True if deleted, False if not found
        """
        path = self.experiments_dir / f"{experiment_id}.yaml"
        if path.exists():
            path.unlink()
            return True
        return False
