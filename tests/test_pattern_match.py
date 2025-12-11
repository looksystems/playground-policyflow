"""Unit tests for PatternMatchNode."""

import pytest

from policyflow.nodes.pattern_match import PatternMatchNode


class TestPatternMatchNodeAnyMode:
    """Tests for PatternMatchNode with mode='any'."""

    def test_any_mode_single_match(self):
        """One pattern matches should return 'matched'."""
        node = PatternMatchNode(
            patterns=[r"\bpassword\b", r"\bsecret\b"],
            mode="any",
        )
        shared = {"input_text": "The password is hidden"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "matched"
        assert exec_res["is_matched"] is True
        assert exec_res["matched_count"] == 1
        assert len(exec_res["matched_patterns"]) == 1
        assert exec_res["matched_patterns"][0]["pattern"] == r"\bpassword\b"

    def test_any_mode_multiple_matches(self):
        """Multiple patterns match should return 'matched'."""
        node = PatternMatchNode(
            patterns=[r"\bpassword\b", r"\bsecret\b"],
            mode="any",
        )
        shared = {"input_text": "The password is a secret"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "matched"
        assert exec_res["matched_count"] == 2

    def test_any_mode_no_match(self):
        """No patterns match should return 'not_matched'."""
        node = PatternMatchNode(
            patterns=[r"\bpassword\b", r"\bsecret\b"],
            mode="any",
        )
        shared = {"input_text": "Hello world"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "not_matched"
        assert exec_res["is_matched"] is False
        assert exec_res["matched_count"] == 0


class TestPatternMatchNodeAllMode:
    """Tests for PatternMatchNode with mode='all'."""

    def test_all_mode_all_match(self):
        """All patterns match should return 'matched'."""
        node = PatternMatchNode(
            patterns=[r"\bHello\b", r"\bworld\b"],
            mode="all",
        )
        shared = {"input_text": "Hello world"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "matched"
        assert exec_res["is_matched"] is True
        assert exec_res["matched_count"] == 2

    def test_all_mode_partial_match(self):
        """Some patterns match should return 'not_matched'."""
        node = PatternMatchNode(
            patterns=[r"\bHello\b", r"\bgoodbye\b"],
            mode="all",
        )
        shared = {"input_text": "Hello world"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "not_matched"
        assert exec_res["is_matched"] is False
        assert exec_res["matched_count"] == 1


class TestPatternMatchNodeNoneMode:
    """Tests for PatternMatchNode with mode='none'."""

    def test_none_mode_no_match(self):
        """No patterns match should return 'matched' in none mode."""
        node = PatternMatchNode(
            patterns=[r"\bpassword\b", r"\bsecret\b"],
            mode="none",
        )
        shared = {"input_text": "Hello world"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "matched"
        assert exec_res["is_matched"] is True
        assert exec_res["matched_count"] == 0

    def test_none_mode_some_match(self):
        """Some patterns match should return 'not_matched' in none mode."""
        node = PatternMatchNode(
            patterns=[r"\bpassword\b", r"\bsecret\b"],
            mode="none",
        )
        shared = {"input_text": "The password is here"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "not_matched"
        assert exec_res["is_matched"] is False


class TestPatternMatchNodeEdgeCases:
    """Edge case tests for PatternMatchNode."""

    def test_invalid_regex_pattern(self, capsys):
        """Invalid regex should be handled gracefully."""
        node = PatternMatchNode(
            patterns=[r"[invalid", r"\bvalid\b"],
            mode="any",
        )
        shared = {"input_text": "This is valid text"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        # Should still match the valid pattern
        assert action == "matched"
        assert len(exec_res["failed_patterns"]) == 1
        assert exec_res["failed_patterns"][0]["pattern"] == r"[invalid"
        assert exec_res["matched_count"] == 1

    def test_match_details_captured(self):
        """Match positions and groups should be stored."""
        node = PatternMatchNode(
            patterns=[r"\d{3}-\d{2}-\d{4}"],  # SSN pattern
            mode="any",
        )
        shared = {"input_text": "My SSN is 123-45-6789"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["matched_patterns"][0]["matches"][0]["text"] == "123-45-6789"
        assert exec_res["matched_patterns"][0]["matches"][0]["start"] == 10
        assert exec_res["matched_patterns"][0]["matches"][0]["end"] == 21

    def test_multiple_matches_same_pattern(self):
        """Multiple matches of the same pattern should all be captured."""
        node = PatternMatchNode(
            patterns=[r"\d+"],
            mode="any",
        )
        shared = {"input_text": "Numbers: 123, 456, 789"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["matched_patterns"][0]["match_count"] == 3

    def test_empty_input(self):
        """Empty input should return 'not_matched' for any mode."""
        node = PatternMatchNode(
            patterns=[r"\bword\b"],
            mode="any",
        )
        shared = {"input_text": ""}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "not_matched"
        assert exec_res["is_matched"] is False

    def test_empty_patterns_list(self):
        """Empty patterns list should return 'not_matched' for any/all."""
        node = PatternMatchNode(patterns=[], mode="any")
        shared = {"input_text": "Hello world"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "not_matched"

    def test_empty_patterns_none_mode(self):
        """Empty patterns list in none mode should return 'matched'."""
        node = PatternMatchNode(patterns=[], mode="none")
        shared = {"input_text": "Hello world"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "matched"

    def test_result_stored_in_shared(self):
        """Result should be stored in shared['pattern_match_result']."""
        node = PatternMatchNode(patterns=[r"\btest\b"], mode="any")
        shared = {"input_text": "This is a test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "pattern_match_result" in shared
        assert shared["pattern_match_result"]["is_matched"] is True

    def test_case_sensitive_by_default(self):
        """Patterns should be case-sensitive by default."""
        node = PatternMatchNode(patterns=[r"\bHello\b"], mode="any")
        shared = {"input_text": "hello world"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "not_matched"

    def test_case_insensitive_with_flag(self):
        """Patterns with (?i) flag should be case-insensitive."""
        node = PatternMatchNode(patterns=[r"(?i)\bHello\b"], mode="any")
        shared = {"input_text": "hello world"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "matched"

    def test_capture_groups(self):
        """Capture groups should be available in match details."""
        node = PatternMatchNode(
            patterns=[r"(\w+)@(\w+)\.(\w+)"],
            mode="any",
        )
        shared = {"input_text": "Contact: test@example.com"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        groups = exec_res["matched_patterns"][0]["matches"][0]["groups"]
        assert groups == ("test", "example", "com")
