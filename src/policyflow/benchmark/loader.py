"""Golden dataset loader.

Loads and parses golden_dataset.yaml files into typed structures.
"""

from __future__ import annotations

from pathlib import Path

from policyflow.benchmark.models import GoldenDataset, GoldenTestCase


def load_golden_dataset(path: str | Path) -> GoldenDataset:
    """Load a golden dataset from a YAML file.

    Args:
        path: Path to the golden dataset YAML file

    Returns:
        Parsed GoldenDataset

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file contains invalid YAML
        pydantic.ValidationError: If the data doesn't match the schema
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset file not found: {path}")

    return GoldenDataset.load_yaml(path)


def load_test_cases(
    path: str | Path,
    *,
    category: str | None = None,
    ids: list[str] | None = None,
) -> list[GoldenTestCase]:
    """Load test cases from a golden dataset file with optional filtering.

    Args:
        path: Path to the golden dataset YAML file
        category: Optional category to filter by
        ids: Optional list of test case IDs to filter by

    Returns:
        List of test cases (filtered if filters provided)
    """
    dataset = load_golden_dataset(path)
    test_cases = dataset.test_cases

    if category is not None:
        test_cases = [tc for tc in test_cases if tc.category == category]

    if ids is not None:
        id_set = set(ids)
        test_cases = [tc for tc in test_cases if tc.id in id_set]

    return test_cases
