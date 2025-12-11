"""Unit tests for LengthGateNode."""

import pytest

from policyflow.nodes.length_gate import LengthGateNode


class TestLengthGateNodeBuckets:
    """Tests for length bucket routing."""

    def test_short_bucket(self):
        """Input within short threshold should return 'short'."""
        node = LengthGateNode(
            thresholds={"short": 100, "medium": 500, "long": 1000}
        )
        shared = {"input_text": "Hello world"}  # 11 chars

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "short"
        assert exec_res["bucket"] == "short"
        assert exec_res["char_count"] == 11

    def test_medium_bucket(self):
        """Input in medium range should return 'medium'."""
        node = LengthGateNode(
            thresholds={"short": 100, "medium": 500, "long": 1000}
        )
        shared = {"input_text": "x" * 200}  # 200 chars

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "medium"
        assert exec_res["bucket"] == "medium"
        assert exec_res["char_count"] == 200

    def test_long_bucket(self):
        """Input exceeding medium but within long should return 'long'."""
        node = LengthGateNode(
            thresholds={"short": 100, "medium": 500, "long": 1000}
        )
        shared = {"input_text": "x" * 800}  # 800 chars

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "long"
        assert exec_res["bucket"] == "long"

    def test_exceeds_all_thresholds(self):
        """Input exceeding all thresholds should return last bucket."""
        node = LengthGateNode(
            thresholds={"short": 100, "medium": 500, "long": 1000}
        )
        shared = {"input_text": "x" * 2000}  # 2000 chars

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "long"
        assert exec_res["bucket"] == "long"


class TestLengthGateNodeBoundaries:
    """Tests for threshold boundary conditions."""

    def test_exact_threshold_boundary(self):
        """Input exactly at threshold should fall into that bucket."""
        node = LengthGateNode(
            thresholds={"short": 100, "medium": 500}
        )
        shared = {"input_text": "x" * 100}  # Exactly 100 chars

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "short"
        assert exec_res["char_count"] == 100

    def test_one_over_threshold(self):
        """Input one character over threshold should fall into next bucket."""
        node = LengthGateNode(
            thresholds={"short": 100, "medium": 500}
        )
        shared = {"input_text": "x" * 101}  # 101 chars

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "medium"


class TestLengthGateNodeMetrics:
    """Tests for length metrics calculation."""

    def test_word_count_calculated(self):
        """word_count should be calculated correctly."""
        node = LengthGateNode(thresholds={"short": 1000})
        shared = {"input_text": "Hello world how are you"}  # 5 words

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["word_count"] == 5

    def test_word_count_multiline(self):
        """word_count should work across multiple lines."""
        node = LengthGateNode(thresholds={"short": 1000})
        shared = {"input_text": "Hello\nworld\nhow\nare\nyou"}  # 5 words

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["word_count"] == 5

    def test_char_count_includes_whitespace(self):
        """char_count should include whitespace characters."""
        node = LengthGateNode(thresholds={"short": 1000})
        shared = {"input_text": "Hello world"}  # 11 chars including space

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["char_count"] == 11


class TestLengthGateNodeEdgeCases:
    """Edge case tests for LengthGateNode."""

    def test_empty_input(self):
        """Empty input should return the first (smallest) bucket."""
        node = LengthGateNode(
            thresholds={"short": 100, "medium": 500}
        )
        shared = {"input_text": ""}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "short"
        assert exec_res["char_count"] == 0
        assert exec_res["word_count"] == 0

    def test_single_threshold(self):
        """Single threshold should work correctly."""
        node = LengthGateNode(thresholds={"tiny": 10})
        shared = {"input_text": "Hello"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "tiny"

    def test_result_stored_in_shared(self):
        """Result should be stored in shared['length_info']."""
        node = LengthGateNode(thresholds={"short": 100})
        shared = {"input_text": "Test message"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "length_info" in shared
        assert shared["length_info"]["char_count"] == 12
        assert shared["length_info"]["bucket"] == "short"

    def test_missing_input_text_key(self):
        """Missing input_text should default to empty string."""
        node = LengthGateNode(thresholds={"short": 100})
        shared = {}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["char_count"] == 0
        assert exec_res["word_count"] == 0

    def test_thresholds_sorted_automatically(self):
        """Thresholds should be sorted by value regardless of input order."""
        # Pass thresholds in reverse order
        node = LengthGateNode(
            thresholds={"long": 1000, "short": 100, "medium": 500}
        )
        shared = {"input_text": "x" * 50}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "short"

    def test_custom_bucket_names(self):
        """Custom bucket names should work."""
        node = LengthGateNode(
            thresholds={"tweet": 280, "paragraph": 1000, "essay": 5000}
        )
        shared = {"input_text": "x" * 150}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "tweet"

    def test_unicode_characters(self):
        """Unicode characters should be counted correctly."""
        node = LengthGateNode(thresholds={"short": 100})
        shared = {"input_text": "Hello 世界 🌍"}  # 10 chars (emoji is 1 char in Python)

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["char_count"] == 10

    def test_whitespace_only_input(self):
        """Whitespace-only input should have 0 words but count chars."""
        node = LengthGateNode(thresholds={"short": 100})
        shared = {"input_text": "   "}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["char_count"] == 3
        assert exec_res["word_count"] == 0
