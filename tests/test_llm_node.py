"""Unit tests for LLMNode model selection hierarchy."""

import os
from unittest.mock import patch

import pytest

from policyflow.nodes.llm_node import LLMNode
from policyflow.config import WorkflowConfig


class TestLLMNodeModelSelection:
    """Tests for LLMNode model selection hierarchy."""

    def test_explicit_model_param_takes_priority(self):
        """Explicit model parameter should take highest priority."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
                "CLASSIFIER_MODEL": "anthropic/claude-haiku-3-5",
            },
        ):
            config = WorkflowConfig()
            node = LLMNode(config=config, model="anthropic/claude-opus-4")
            assert node.model == "anthropic/claude-opus-4"

    def test_node_type_config_over_global_default(self):
        """Node type config should override global default when no explicit model."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
                "CLASSIFIER_MODEL": "anthropic/claude-haiku-3-5",
            },
        ):
            from policyflow.nodes.classifier import ClassifierNode

            config = WorkflowConfig()
            # ClassifierNode should get CLASSIFIER_MODEL
            node = ClassifierNode(
                categories=["spam", "ham"], config=config, cache_ttl=0
            )
            assert node.model == "anthropic/claude-haiku-3-5"

    def test_global_default_when_no_node_type_config(self):
        """Should use global default when node type not configured."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
            },
            clear=True,
        ):
            from policyflow.nodes.classifier import ClassifierNode

            config = WorkflowConfig()
            # No CLASSIFIER_MODEL set, should use POLICY_EVAL_MODEL
            node = ClassifierNode(
                categories=["spam", "ham"], config=config, cache_ttl=0
            )
            assert node.model == "anthropic/claude-sonnet-4"

    def test_hardcoded_fallback_when_no_env_vars(self):
        """Should use hardcoded fallback when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            config = WorkflowConfig()
            node = LLMNode(config=config)
            # Should use the hardcoded default from ModelConfig
            assert node.model == "anthropic/claude-sonnet-4-20250514"

    def test_different_node_types_use_different_models(self):
        """Different node types should use their respective configured models."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
                "CLASSIFIER_MODEL": "anthropic/claude-haiku-3-5",
                "SENTIMENT_MODEL": "anthropic/claude-opus-4",
            },
        ):
            from policyflow.nodes.classifier import ClassifierNode
            from policyflow.nodes.sentiment import SentimentNode

            config = WorkflowConfig()

            classifier = ClassifierNode(
                categories=["spam", "ham"], config=config, cache_ttl=0
            )
            sentiment = SentimentNode(config=config, cache_ttl=0)

            assert classifier.model == "anthropic/claude-haiku-3-5"
            assert sentiment.model == "anthropic/claude-opus-4"

    def test_explicit_model_overrides_node_type_config(self):
        """Explicit model param should override node type config."""
        with patch.dict(
            os.environ,
            {
                "CLASSIFIER_MODEL": "anthropic/claude-haiku-3-5",
            },
        ):
            from policyflow.nodes.classifier import ClassifierNode

            config = WorkflowConfig()
            node = ClassifierNode(
                categories=["spam", "ham"],
                config=config,
                model="anthropic/claude-opus-4",
                cache_ttl=0,
            )
            assert node.model == "anthropic/claude-opus-4"

    def test_data_extractor_node_uses_config(self):
        """DataExtractorNode should use DATA_EXTRACTOR_MODEL env var."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
                "DATA_EXTRACTOR_MODEL": "anthropic/claude-opus-4",
            },
        ):
            from policyflow.nodes.data_extractor import DataExtractorNode

            config = WorkflowConfig()
            node = DataExtractorNode(
                schema={"facts": ["topic"]}, config=config, cache_ttl=0
            )
            assert node.model == "anthropic/claude-opus-4"

    def test_sampler_node_uses_config(self):
        """SamplerNode should use SAMPLER_MODEL env var."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
                "SAMPLER_MODEL": "anthropic/claude-haiku-3-5",
            },
        ):
            from policyflow.nodes.sampler import SamplerNode

            config = WorkflowConfig()
            node = SamplerNode(
                n_samples=2,
                aggregation="majority",
                inner_prompt="test",
                config=config,
                cache_ttl=0,
            )
            assert node.model == "anthropic/claude-haiku-3-5"


class TestLLMNodeBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_class_default_model_still_exists(self):
        """Class-level default_model attribute should still exist."""
        assert hasattr(LLMNode, "default_model")
        assert LLMNode.default_model == "anthropic/claude-sonnet-4-20250514"

    def test_no_env_vars_uses_hardcoded_default(self):
        """Without env vars, should use the hardcoded default."""
        with patch.dict(os.environ, {}, clear=True):
            config = WorkflowConfig()
            node = LLMNode(config=config)
            assert node.model == "anthropic/claude-sonnet-4-20250514"

    def test_existing_code_with_explicit_model_still_works(self):
        """Existing code that passes explicit model should still work."""
        config = WorkflowConfig()
        node = LLMNode(config=config, model="custom-model")
        assert node.model == "custom-model"


class TestLLMNodePriorityHierarchy:
    """Tests to verify complete priority hierarchy."""

    def test_priority_1_explicit_param(self):
        """Priority 1: Explicit parameter beats everything."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "global",
                "CLASSIFIER_MODEL": "classifier-specific",
            },
        ):
            from policyflow.nodes.classifier import ClassifierNode

            config = WorkflowConfig()
            node = ClassifierNode(
                categories=["a", "b"],
                config=config,
                model="explicit-param",
                cache_ttl=0,
            )
            assert node.model == "explicit-param"

    def test_priority_2_type_specific_env(self):
        """Priority 2: Type-specific env var beats global default."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "global",
                "CLASSIFIER_MODEL": "classifier-specific",
            },
        ):
            from policyflow.nodes.classifier import ClassifierNode

            config = WorkflowConfig()
            node = ClassifierNode(
                categories=["a", "b"], config=config, cache_ttl=0
            )
            assert node.model == "classifier-specific"

    def test_priority_3_global_default(self):
        """Priority 3: Global default when no type-specific config."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "global",
            },
            clear=True,
        ):
            from policyflow.nodes.classifier import ClassifierNode

            config = WorkflowConfig()
            node = ClassifierNode(
                categories=["a", "b"], config=config, cache_ttl=0
            )
            assert node.model == "global"

    def test_priority_4_hardcoded_fallback(self):
        """Priority 4: Hardcoded fallback when nothing configured."""
        with patch.dict(os.environ, {}, clear=True):
            from policyflow.nodes.classifier import ClassifierNode

            config = WorkflowConfig()
            node = ClassifierNode(
                categories=["a", "b"], config=config, cache_ttl=0
            )
            assert node.model == "anthropic/claude-sonnet-4-20250514"
