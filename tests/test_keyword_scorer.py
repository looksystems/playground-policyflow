"""Unit tests for KeywordScorerNode."""

import pytest

from policyflow.nodes.keyword_scorer import KeywordScorerNode


class TestKeywordScorerNodeThresholds:
    """Tests for keyword scoring threshold behavior."""

    def test_high_score_threshold(self):
        """Score above high threshold should return 'high'."""
        node = KeywordScorerNode(
            keywords={"urgent": 0.5, "critical": 0.8},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "This is urgent and critical"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "high"
        assert exec_res["score"] == 1.3  # 0.5 + 0.8
        assert exec_res["score_level"] == "high"

    def test_medium_score_threshold(self):
        """Score between medium and high should return 'medium'."""
        node = KeywordScorerNode(
            keywords={"urgent": 0.5, "critical": 0.8},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "This is urgent"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "medium"
        assert exec_res["score"] == 0.5
        assert exec_res["score_level"] == "medium"

    def test_low_score(self):
        """Score below medium threshold should return 'low'."""
        node = KeywordScorerNode(
            keywords={"urgent": 0.5, "critical": 0.8},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "Nothing special here"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "low"
        assert exec_res["score"] == 0.0
        assert exec_res["score_level"] == "low"


class TestKeywordScorerNodeWeights:
    """Tests for keyword weight behavior."""

    def test_negative_weights(self):
        """Negative weights should reduce the score."""
        node = KeywordScorerNode(
            keywords={"urgent": 1.0, "spam": -2.0},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "urgent spam message"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert exec_res["score"] == -1.0  # 1.0 + (-2.0)
        assert action == "low"

    def test_multiple_occurrences(self):
        """Same keyword appearing multiple times should multiply weight."""
        node = KeywordScorerNode(
            keywords={"urgent": 0.3},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "urgent urgent urgent"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert exec_res["score"] == pytest.approx(0.9)  # 0.3 * 3
        assert action == "high"
        assert exec_res["matched_keywords"]["urgent"]["count"] == 3
        assert exec_res["matched_keywords"]["urgent"]["contribution"] == pytest.approx(0.9)


class TestKeywordScorerNodeMatching:
    """Tests for keyword matching behavior."""

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        node = KeywordScorerNode(
            keywords={"urgent": 1.0},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "URGENT request"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["score"] == 1.0
        assert "urgent" in exec_res["matched_keywords"]

    def test_word_boundaries(self):
        """'urgent' should not match 'urgently'."""
        node = KeywordScorerNode(
            keywords={"urgent": 1.0},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "I urgently need help"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["score"] == 0.0
        assert "urgent" not in exec_res["matched_keywords"]

    def test_word_boundaries_exact_match(self):
        """Exact word match should still work."""
        node = KeywordScorerNode(
            keywords={"urgent": 1.0},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "This is urgent!"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["score"] == 1.0


class TestKeywordScorerNodeEdgeCases:
    """Edge case tests for KeywordScorerNode."""

    def test_empty_input(self):
        """Empty input should return low score."""
        node = KeywordScorerNode(
            keywords={"urgent": 1.0},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": ""}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "low"
        assert exec_res["score"] == 0.0
        assert exec_res["total_matches"] == 0

    def test_no_matches(self):
        """No keywords found should return low score."""
        node = KeywordScorerNode(
            keywords={"urgent": 1.0, "critical": 0.8},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "A regular message without special words"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "low"
        assert exec_res["score"] == 0.0
        assert len(exec_res["matched_keywords"]) == 0

    def test_result_stored_in_shared(self):
        """Result should be stored in shared['keyword_score']."""
        node = KeywordScorerNode(
            keywords={"test": 1.0},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "test message"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        node.post(shared, prep_res, exec_res)

        assert "keyword_score" in shared
        assert shared["keyword_score"]["score"] == 1.0
        assert shared["keyword_score"]["score_level"] == "high"

    def test_matched_keywords_details(self):
        """Matched keywords should include count, weight, and contribution."""
        node = KeywordScorerNode(
            keywords={"urgent": 0.5},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "urgent urgent"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        matched = exec_res["matched_keywords"]["urgent"]
        assert matched["count"] == 2
        assert matched["weight"] == 0.5
        assert matched["contribution"] == 1.0

    def test_total_matches_count(self):
        """total_matches should be sum of all keyword match counts."""
        node = KeywordScorerNode(
            keywords={"urgent": 0.5, "critical": 0.3},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "urgent critical urgent"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["total_matches"] == 3  # 2 urgent + 1 critical

    def test_missing_input_text_key(self):
        """Missing input_text should default to empty string."""
        node = KeywordScorerNode(
            keywords={"urgent": 1.0},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)

        assert exec_res["score"] == 0.0

    def test_threshold_boundary_high(self):
        """Score exactly at high threshold should be 'high'."""
        node = KeywordScorerNode(
            keywords={"test": 0.7},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "high"

    def test_threshold_boundary_medium(self):
        """Score exactly at medium threshold should be 'medium'."""
        node = KeywordScorerNode(
            keywords={"test": 0.3},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "test"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "medium"

    def test_phrase_keyword(self):
        """Multi-word phrases should be matched."""
        node = KeywordScorerNode(
            keywords={"high priority": 1.0},
            thresholds={"high": 0.7, "medium": 0.3},
        )
        shared = {"input_text": "This is a high priority request"}

        prep_res = node.prep(shared)
        exec_res = node.exec(prep_res)
        action = node.post(shared, prep_res, exec_res)

        assert action == "high"
        assert exec_res["score"] == 1.0
