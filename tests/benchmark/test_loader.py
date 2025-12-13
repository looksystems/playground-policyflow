"""Tests for golden dataset loader."""

import tempfile
from pathlib import Path

import pytest
import yaml


class TestLoadGoldenDataset:
    """Tests for loading golden datasets from YAML files."""

    def test_load_golden_dataset_from_file(self, tmp_path):
        from policyflow.benchmark.loader import load_golden_dataset

        # Create a test dataset file
        dataset_content = {
            "policy_file": "policy.md",
            "description": "Test dataset",
            "test_cases": [
                {
                    "id": "test_001",
                    "name": "Test case 1",
                    "input_text": "Sample input text",
                    "expected": {
                        "policy_satisfied": True,
                        "criterion_results": {
                            "criterion_1": {"met": True},
                        },
                    },
                    "category": "clear_pass",
                    "notes": "Test notes",
                }
            ],
        }

        dataset_file = tmp_path / "golden_dataset.yaml"
        with open(dataset_file, "w") as f:
            yaml.dump(dataset_content, f)

        # Load the dataset
        dataset = load_golden_dataset(dataset_file)

        assert dataset.policy_file == "policy.md"
        assert dataset.description == "Test dataset"
        assert len(dataset.test_cases) == 1
        assert dataset.test_cases[0].id == "test_001"

    def test_load_golden_dataset_with_sub_results(self, tmp_path):
        from policyflow.benchmark.loader import load_golden_dataset

        dataset_content = {
            "policy_file": "policy.md",
            "description": "Test with sub-results",
            "test_cases": [
                {
                    "id": "test_001",
                    "name": "Test with nested results",
                    "input_text": "Input",
                    "expected": {
                        "policy_satisfied": True,
                        "criterion_results": {
                            "criterion_1": {
                                "met": True,
                                "sub_results": {
                                    "criterion_1a": {"met": True},
                                    "criterion_1b": {"met": False},
                                },
                            },
                        },
                    },
                    "category": "clear_pass",
                    "notes": "",
                }
            ],
        }

        dataset_file = tmp_path / "dataset.yaml"
        with open(dataset_file, "w") as f:
            yaml.dump(dataset_content, f)

        dataset = load_golden_dataset(dataset_file)
        criterion_1 = dataset.test_cases[0].expected.criterion_results["criterion_1"]
        assert criterion_1.sub_results is not None
        assert criterion_1.sub_results["criterion_1a"].met is True
        assert criterion_1.sub_results["criterion_1b"].met is False

    def test_load_golden_dataset_string_path(self, tmp_path):
        from policyflow.benchmark.loader import load_golden_dataset

        dataset_content = {
            "policy_file": "policy.md",
            "description": "Test",
            "test_cases": [],
        }

        dataset_file = tmp_path / "dataset.yaml"
        with open(dataset_file, "w") as f:
            yaml.dump(dataset_content, f)

        # Pass as string instead of Path
        dataset = load_golden_dataset(str(dataset_file))
        assert dataset.policy_file == "policy.md"

    def test_load_golden_dataset_file_not_found(self):
        from policyflow.benchmark.loader import load_golden_dataset

        with pytest.raises(FileNotFoundError):
            load_golden_dataset("/nonexistent/path/dataset.yaml")

    def test_load_golden_dataset_invalid_yaml(self, tmp_path):
        from policyflow.benchmark.loader import load_golden_dataset

        invalid_file = tmp_path / "invalid.yaml"
        with open(invalid_file, "w") as f:
            f.write("this is not: valid: yaml: {{}}")

        with pytest.raises(Exception):  # yaml.YAMLError or similar
            load_golden_dataset(invalid_file)


class TestLoadGoldenDatasetRealFile:
    """Test loading the actual golden_dataset.yaml from the project."""

    def test_load_real_golden_dataset(self):
        from policyflow.benchmark.loader import load_golden_dataset

        # Load the actual golden dataset from the project root
        dataset_path = Path(__file__).parent.parent.parent / "golden_dataset.yaml"
        if not dataset_path.exists():
            pytest.skip("golden_dataset.yaml not found")

        dataset = load_golden_dataset(dataset_path)

        # Verify structure
        assert dataset.policy_file == "policy.md"
        assert len(dataset.test_cases) > 0

        # Check a known test case
        test_001 = next((tc for tc in dataset.test_cases if tc.id == "test_001"), None)
        assert test_001 is not None
        assert test_001.category == "clear_pass"
        assert test_001.expected.policy_satisfied is True


class TestFilterTestCases:
    """Tests for filtering test cases."""

    def test_filter_by_category(self, tmp_path):
        from policyflow.benchmark.loader import load_golden_dataset

        dataset_content = {
            "policy_file": "policy.md",
            "description": "Test",
            "test_cases": [
                {
                    "id": "test_001",
                    "name": "Pass case",
                    "input_text": "Input 1",
                    "expected": {"policy_satisfied": True, "criterion_results": {}},
                    "category": "clear_pass",
                    "notes": "",
                },
                {
                    "id": "test_002",
                    "name": "Fail case",
                    "input_text": "Input 2",
                    "expected": {"policy_satisfied": False, "criterion_results": {}},
                    "category": "clear_fail",
                    "notes": "",
                },
                {
                    "id": "test_003",
                    "name": "Another pass",
                    "input_text": "Input 3",
                    "expected": {"policy_satisfied": True, "criterion_results": {}},
                    "category": "clear_pass",
                    "notes": "",
                },
            ],
        }

        dataset_file = tmp_path / "dataset.yaml"
        with open(dataset_file, "w") as f:
            yaml.dump(dataset_content, f)

        dataset = load_golden_dataset(dataset_file)

        # Filter by category
        clear_pass = dataset.filter_by_category("clear_pass")
        assert len(clear_pass) == 2
        assert all(tc.category == "clear_pass" for tc in clear_pass)

        clear_fail = dataset.filter_by_category("clear_fail")
        assert len(clear_fail) == 1

    def test_filter_by_ids(self, tmp_path):
        from policyflow.benchmark.loader import load_golden_dataset

        dataset_content = {
            "policy_file": "policy.md",
            "description": "Test",
            "test_cases": [
                {
                    "id": "test_001",
                    "name": "Case 1",
                    "input_text": "Input 1",
                    "expected": {"policy_satisfied": True, "criterion_results": {}},
                    "category": "clear_pass",
                    "notes": "",
                },
                {
                    "id": "test_002",
                    "name": "Case 2",
                    "input_text": "Input 2",
                    "expected": {"policy_satisfied": False, "criterion_results": {}},
                    "category": "clear_fail",
                    "notes": "",
                },
                {
                    "id": "test_003",
                    "name": "Case 3",
                    "input_text": "Input 3",
                    "expected": {"policy_satisfied": True, "criterion_results": {}},
                    "category": "clear_pass",
                    "notes": "",
                },
            ],
        }

        dataset_file = tmp_path / "dataset.yaml"
        with open(dataset_file, "w") as f:
            yaml.dump(dataset_content, f)

        dataset = load_golden_dataset(dataset_file)

        # Filter by IDs
        filtered = dataset.filter_by_ids(["test_001", "test_003"])
        assert len(filtered) == 2
        assert {tc.id for tc in filtered} == {"test_001", "test_003"}


class TestLoadTestCases:
    """Tests for load_test_cases convenience function."""

    def test_load_test_cases(self, tmp_path):
        from policyflow.benchmark.loader import load_test_cases

        dataset_content = {
            "policy_file": "policy.md",
            "description": "Test",
            "test_cases": [
                {
                    "id": "test_001",
                    "name": "Case 1",
                    "input_text": "Input",
                    "expected": {"policy_satisfied": True, "criterion_results": {}},
                    "category": "clear_pass",
                    "notes": "",
                }
            ],
        }

        dataset_file = tmp_path / "dataset.yaml"
        with open(dataset_file, "w") as f:
            yaml.dump(dataset_content, f)

        # Load just the test cases
        test_cases = load_test_cases(dataset_file)
        assert len(test_cases) == 1
        assert test_cases[0].id == "test_001"

    def test_load_test_cases_with_category_filter(self, tmp_path):
        from policyflow.benchmark.loader import load_test_cases

        dataset_content = {
            "policy_file": "policy.md",
            "description": "Test",
            "test_cases": [
                {
                    "id": "test_001",
                    "name": "Pass",
                    "input_text": "Input",
                    "expected": {"policy_satisfied": True, "criterion_results": {}},
                    "category": "clear_pass",
                    "notes": "",
                },
                {
                    "id": "test_002",
                    "name": "Fail",
                    "input_text": "Input",
                    "expected": {"policy_satisfied": False, "criterion_results": {}},
                    "category": "clear_fail",
                    "notes": "",
                },
            ],
        }

        dataset_file = tmp_path / "dataset.yaml"
        with open(dataset_file, "w") as f:
            yaml.dump(dataset_content, f)

        # Load with category filter
        test_cases = load_test_cases(dataset_file, category="clear_fail")
        assert len(test_cases) == 1
        assert test_cases[0].id == "test_002"

    def test_load_test_cases_with_id_filter(self, tmp_path):
        from policyflow.benchmark.loader import load_test_cases

        dataset_content = {
            "policy_file": "policy.md",
            "description": "Test",
            "test_cases": [
                {
                    "id": "test_001",
                    "name": "Case 1",
                    "input_text": "Input",
                    "expected": {"policy_satisfied": True, "criterion_results": {}},
                    "category": "clear_pass",
                    "notes": "",
                },
                {
                    "id": "test_002",
                    "name": "Case 2",
                    "input_text": "Input",
                    "expected": {"policy_satisfied": False, "criterion_results": {}},
                    "category": "clear_fail",
                    "notes": "",
                },
            ],
        }

        dataset_file = tmp_path / "dataset.yaml"
        with open(dataset_file, "w") as f:
            yaml.dump(dataset_content, f)

        # Load with ID filter
        test_cases = load_test_cases(dataset_file, ids=["test_002"])
        assert len(test_cases) == 1
        assert test_cases[0].id == "test_002"
