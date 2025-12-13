"""Tests for hypothesis generator."""

from unittest.mock import MagicMock

import pytest

from policyflow.benchmark.models import (
    AnalysisReport,
    FailurePattern,
    Hypothesis,
    ProblematicCriterion,
)
from policyflow.models import ParsedWorkflowPolicy


class TestTemplateBasedHypothesisGenerator:
    """Tests for TemplateBasedHypothesisGenerator."""

    def test_generator_initialization(self):
        from policyflow.benchmark.hypothesis import TemplateBasedHypothesisGenerator

        generator = TemplateBasedHypothesisGenerator()
        assert generator is not None

    def test_generate_for_criterion_systematic_failure(self):
        from policyflow.benchmark.hypothesis import TemplateBasedHypothesisGenerator

        analysis = AnalysisReport(
            patterns=[
                FailurePattern(
                    pattern_type="criterion_systematic",
                    description="Criterion 'criterion_3' fails systematically",
                    affected_tests=["t1", "t2"],
                    severity="high",
                )
            ],
            problematic_criteria=[
                ProblematicCriterion(
                    criterion_id="criterion_3",
                    failure_rate=0.6,
                    false_positive_rate=0.2,
                    false_negative_rate=0.4,
                    common_failure_patterns=[],
                )
            ],
            recommendations=[],
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)

        generator = TemplateBasedHypothesisGenerator()
        hypotheses = generator.generate(analysis, mock_workflow)

        assert len(hypotheses) >= 1
        # Should suggest prompt tuning for systematic failures
        prompt_tuning = [h for h in hypotheses if h.change_type == "prompt_tuning"]
        assert len(prompt_tuning) >= 1

    def test_generate_for_false_positive_heavy(self):
        from policyflow.benchmark.hypothesis import TemplateBasedHypothesisGenerator

        analysis = AnalysisReport(
            patterns=[
                FailurePattern(
                    pattern_type="false_positive_heavy",
                    description="Criterion 'c1' has high false positive rate",
                    affected_tests=[],
                    severity="medium",
                )
            ],
            problematic_criteria=[],
            recommendations=[],
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)

        generator = TemplateBasedHypothesisGenerator()
        hypotheses = generator.generate(analysis, mock_workflow)

        # Should suggest adding confidence gate or threshold adjustment
        assert len(hypotheses) >= 1

    def test_generate_for_confidence_miscalibration(self):
        from policyflow.benchmark.hypothesis import TemplateBasedHypothesisGenerator

        analysis = AnalysisReport(
            patterns=[
                FailurePattern(
                    pattern_type="confidence_miscalibration",
                    description="High confidence predictions have low accuracy",
                    affected_tests=[],
                    severity="high",
                )
            ],
            problematic_criteria=[],
            recommendations=[],
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)

        generator = TemplateBasedHypothesisGenerator()
        hypotheses = generator.generate(analysis, mock_workflow)

        # Should suggest threshold adjustment
        threshold_hyps = [h for h in hypotheses if h.change_type == "threshold"]
        assert len(threshold_hyps) >= 1

    def test_generate_unique_hypothesis_ids(self):
        from policyflow.benchmark.hypothesis import TemplateBasedHypothesisGenerator

        analysis = AnalysisReport(
            patterns=[
                FailurePattern(
                    pattern_type="criterion_systematic",
                    description="Pattern 1",
                    affected_tests=[],
                    severity="high",
                ),
                FailurePattern(
                    pattern_type="false_positive_heavy",
                    description="Pattern 2",
                    affected_tests=[],
                    severity="medium",
                ),
            ],
            problematic_criteria=[],
            recommendations=[],
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)

        generator = TemplateBasedHypothesisGenerator()
        hypotheses = generator.generate(analysis, mock_workflow)

        # All IDs should be unique
        ids = [h.id for h in hypotheses]
        assert len(ids) == len(set(ids))

    def test_generate_no_patterns_returns_empty(self):
        from policyflow.benchmark.hypothesis import TemplateBasedHypothesisGenerator

        analysis = AnalysisReport(
            patterns=[],
            problematic_criteria=[],
            recommendations=[],
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)

        generator = TemplateBasedHypothesisGenerator()
        hypotheses = generator.generate(analysis, mock_workflow)

        assert hypotheses == []


class TestHypothesisGeneratorFactory:
    """Tests for hypothesis generator factory."""

    def test_create_template_generator(self):
        from policyflow.benchmark.hypothesis import create_hypothesis_generator

        generator = create_hypothesis_generator(mode="template")
        assert generator is not None

    def test_create_hybrid_generator(self):
        from policyflow.benchmark.hypothesis import create_hypothesis_generator

        generator = create_hypothesis_generator(mode="hybrid")
        assert generator is not None

    def test_invalid_mode_raises(self):
        from policyflow.benchmark.hypothesis import create_hypothesis_generator

        with pytest.raises(ValueError):
            create_hypothesis_generator(mode="invalid")


class TestVariableExtraction:
    """Tests for variable extraction from patterns - should use metadata, not regex."""

    def test_extract_criterion_from_metadata(self):
        """Test that criterion is extracted from metadata field, not description."""
        from policyflow.benchmark.hypothesis import TemplateBasedHypothesisGenerator

        analysis = AnalysisReport(
            patterns=[
                FailurePattern(
                    pattern_type="criterion_systematic",
                    description="Some description without quotes",  # No parseable format
                    affected_tests=["t1"],
                    severity="high",
                    metadata={"criterion": "my_criterion_id"},  # Use metadata instead
                )
            ],
            problematic_criteria=[],
            recommendations=[],
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)
        generator = TemplateBasedHypothesisGenerator()
        hypotheses = generator.generate(analysis, mock_workflow)

        # Should use criterion from metadata
        assert len(hypotheses) >= 1
        # Check that target was set correctly from metadata
        assert any(h.target == "my_criterion_id" for h in hypotheses)

    def test_extract_category_from_metadata(self):
        """Test that category is extracted from metadata field."""
        from policyflow.benchmark.hypothesis import TemplateBasedHypothesisGenerator

        analysis = AnalysisReport(
            patterns=[
                FailurePattern(
                    pattern_type="category_cluster",
                    description="High failure rate in some category",
                    affected_tests=["t1", "t2"],
                    severity="high",
                    metadata={"category": "edge_cases"},
                )
            ],
            problematic_criteria=[],
            recommendations=[],
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)
        generator = TemplateBasedHypothesisGenerator()
        hypotheses = generator.generate(analysis, mock_workflow)

        assert len(hypotheses) >= 1
        # Check that target was set correctly from metadata
        assert any(h.target == "edge_cases" for h in hypotheses)

    def test_fallback_to_regex_when_no_metadata(self):
        """Test backward compatibility - fall back to regex when no metadata."""
        from policyflow.benchmark.hypothesis import TemplateBasedHypothesisGenerator

        analysis = AnalysisReport(
            patterns=[
                FailurePattern(
                    pattern_type="criterion_systematic",
                    description="Criterion 'criterion_3' fails systematically",
                    affected_tests=["t1"],
                    severity="high",
                    # No metadata field
                )
            ],
            problematic_criteria=[],
            recommendations=[],
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)
        generator = TemplateBasedHypothesisGenerator()
        hypotheses = generator.generate(analysis, mock_workflow)

        # Should still work with regex fallback
        assert len(hypotheses) >= 1
        assert any(h.target == "criterion_3" for h in hypotheses)


class TestLLMHypothesisGenerator:
    """Tests for LLM-powered hypothesis generator."""

    def test_llm_generator_without_model_uses_templates(self):
        """Test that LLM generator falls back to templates when no model is set."""
        from policyflow.benchmark.hypothesis import LLMHypothesisGenerator

        analysis = AnalysisReport(
            patterns=[
                FailurePattern(
                    pattern_type="criterion_systematic",
                    description="Criterion fails",
                    affected_tests=["t1"],
                    severity="high",
                    metadata={"criterion": "test_crit"},
                )
            ],
            problematic_criteria=[],
            recommendations=[],
        )

        mock_workflow = MagicMock(spec=ParsedWorkflowPolicy)

        # No model configured - should use templates
        generator = LLMHypothesisGenerator(model=None)
        hypotheses = generator.generate(analysis, mock_workflow)

        # Should still generate hypotheses from templates
        assert len(hypotheses) >= 1

    def test_llm_generator_parse_hypotheses(self):
        """Test parsing of LLM response into Hypothesis objects."""
        from policyflow.benchmark.hypothesis import LLMHypothesisGenerator

        generator = LLMHypothesisGenerator()

        mock_response = {
            "hypotheses": [
                {
                    "id": "hyp_001",
                    "description": "Test hypothesis",
                    "change_type": "prompt_tuning",
                    "target": "node_1",
                    "suggested_change": {"prompt": "new prompt"},
                    "rationale": "Because it works",
                    "expected_impact": "Better accuracy",
                },
                {
                    "id": "hyp_002",
                    "description": "Second hypothesis",
                    "change_type": "threshold",
                    "target": "node_2",
                    "suggested_change": {"threshold": 0.8},
                    "rationale": "Higher threshold",
                    "expected_impact": "Better precision",
                },
            ]
        }

        hypotheses = generator._parse_hypotheses(mock_response)

        assert len(hypotheses) == 2
        assert hypotheses[0].id == "hyp_001"
        assert hypotheses[0].change_type == "prompt_tuning"
        assert hypotheses[1].id == "hyp_002"
        assert hypotheses[1].target == "node_2"

    def test_llm_generator_parse_handles_malformed(self):
        """Test that parsing handles malformed responses gracefully."""
        from policyflow.benchmark.hypothesis import LLMHypothesisGenerator

        generator = LLMHypothesisGenerator()

        # Empty response
        assert generator._parse_hypotheses({}) == []

        # Missing hypotheses key
        assert generator._parse_hypotheses({"other": "data"}) == []

        # None input
        assert generator._parse_hypotheses(None) == []

    def test_create_generator_with_model(self):
        """Test factory creates LLM generator with model."""
        from policyflow.benchmark.hypothesis import (
            LLMHypothesisGenerator,
            create_hypothesis_generator,
        )

        generator = create_hypothesis_generator(mode="llm", model="gpt-4")
        assert isinstance(generator, LLMHypothesisGenerator)
        assert generator.model == "gpt-4"
