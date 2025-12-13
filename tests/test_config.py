"""Unit tests for configuration management."""

import os
from unittest.mock import patch

import pytest

from policyflow.config import WorkflowConfig, ModelConfig


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_default_model_from_env(self):
        """Default model should come from POLICY_EVAL_MODEL env var."""
        with patch.dict(os.environ, {"POLICY_EVAL_MODEL": "anthropic/claude-opus-4"}):
            config = ModelConfig()
            assert config.default_model == "anthropic/claude-opus-4"

    def test_default_model_hardcoded_fallback(self):
        """Default model should fallback to hardcoded value if env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = ModelConfig()
            assert config.default_model == "anthropic/claude-sonnet-4-20250514"

    def test_node_type_models_from_env(self):
        """Node type models should be loaded from env vars."""
        with patch.dict(
            os.environ,
            {
                "CLASSIFIER_MODEL": "anthropic/claude-haiku-3-5",
                "DATA_EXTRACTOR_MODEL": "anthropic/claude-opus-4",
                "SENTIMENT_MODEL": "anthropic/claude-haiku-3-5",
                "SAMPLER_MODEL": "anthropic/claude-sonnet-4",
            },
        ):
            config = ModelConfig()
            assert config.classifier_model == "anthropic/claude-haiku-3-5"
            assert config.data_extractor_model == "anthropic/claude-opus-4"
            assert config.sentiment_model == "anthropic/claude-haiku-3-5"
            assert config.sampler_model == "anthropic/claude-sonnet-4"

    def test_node_type_models_optional(self):
        """Node type models should be None if env vars not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = ModelConfig()
            assert config.classifier_model is None
            assert config.data_extractor_model is None
            assert config.sentiment_model is None
            assert config.sampler_model is None

    def test_cli_task_models_from_env(self):
        """CLI task models should be loaded from env vars."""
        with patch.dict(
            os.environ,
            {
                "GENERATE_MODEL": "anthropic/claude-opus-4",
                "ANALYZE_MODEL": "anthropic/claude-sonnet-4",
                "HYPOTHESIZE_MODEL": "anthropic/claude-opus-4",
                "OPTIMIZE_MODEL": "anthropic/claude-sonnet-4",
            },
        ):
            config = ModelConfig()
            assert config.generate_model == "anthropic/claude-opus-4"
            assert config.analyze_model == "anthropic/claude-sonnet-4"
            assert config.hypothesize_model == "anthropic/claude-opus-4"
            assert config.optimize_model == "anthropic/claude-sonnet-4"

    def test_cli_task_models_optional(self):
        """CLI task models should be None if env vars not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = ModelConfig()
            assert config.generate_model is None
            assert config.analyze_model is None
            assert config.hypothesize_model is None
            assert config.optimize_model is None

    def test_get_model_for_node_type_with_specific_config(self):
        """get_model_for_node_type should return type-specific model if configured."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
                "CLASSIFIER_MODEL": "anthropic/claude-haiku-3-5",
            },
        ):
            config = ModelConfig()
            assert (
                config.get_model_for_node_type("ClassifierNode")
                == "anthropic/claude-haiku-3-5"
            )

    def test_get_model_for_node_type_fallback_to_default(self):
        """get_model_for_node_type should fallback to default if type not configured."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
            },
            clear=True,
        ):
            config = ModelConfig()
            # No CLASSIFIER_MODEL set, should use default
            assert (
                config.get_model_for_node_type("ClassifierNode")
                == "anthropic/claude-sonnet-4"
            )

    def test_get_model_for_node_type_unknown_type(self):
        """get_model_for_node_type should return default for unknown node types."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
            },
            clear=True,
        ):
            config = ModelConfig()
            assert (
                config.get_model_for_node_type("UnknownNode")
                == "anthropic/claude-sonnet-4"
            )

    def test_get_model_for_node_type_all_types(self):
        """get_model_for_node_type should support all documented node types."""
        with patch.dict(
            os.environ,
            {
                "CLASSIFIER_MODEL": "model-1",
                "DATA_EXTRACTOR_MODEL": "model-2",
                "SENTIMENT_MODEL": "model-3",
                "SAMPLER_MODEL": "model-4",
            },
        ):
            config = ModelConfig()
            assert config.get_model_for_node_type("ClassifierNode") == "model-1"
            assert config.get_model_for_node_type("DataExtractorNode") == "model-2"
            assert config.get_model_for_node_type("SentimentNode") == "model-3"
            assert config.get_model_for_node_type("SamplerNode") == "model-4"

    def test_get_model_for_task_with_specific_config(self):
        """get_model_for_task should return task-specific model if configured."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
                "GENERATE_MODEL": "anthropic/claude-opus-4",
            },
        ):
            config = ModelConfig()
            assert (
                config.get_model_for_task("generate") == "anthropic/claude-opus-4"
            )

    def test_get_model_for_task_fallback_to_default(self):
        """get_model_for_task should fallback to default if task not configured."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
            },
            clear=True,
        ):
            config = ModelConfig()
            # No GENERATE_MODEL set, should use default
            assert config.get_model_for_task("generate") == "anthropic/claude-sonnet-4"

    def test_get_model_for_task_unknown_task(self):
        """get_model_for_task should return default for unknown tasks."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
            },
            clear=True,
        ):
            config = ModelConfig()
            assert (
                config.get_model_for_task("unknown_task")
                == "anthropic/claude-sonnet-4"
            )

    def test_get_model_for_task_all_tasks(self):
        """get_model_for_task should support all documented CLI tasks."""
        with patch.dict(
            os.environ,
            {
                "GENERATE_MODEL": "model-1",
                "ANALYZE_MODEL": "model-2",
                "HYPOTHESIZE_MODEL": "model-3",
                "OPTIMIZE_MODEL": "model-4",
            },
        ):
            config = ModelConfig()
            assert config.get_model_for_task("generate") == "model-1"
            assert config.get_model_for_task("analyze") == "model-2"
            assert config.get_model_for_task("hypothesize") == "model-3"
            assert config.get_model_for_task("optimize") == "model-4"


class TestWorkflowConfigWithModels:
    """Tests for WorkflowConfig integration with ModelConfig."""

    def test_workflow_config_includes_model_config(self):
        """WorkflowConfig should include a models field."""
        config = WorkflowConfig()
        assert hasattr(config, "models")
        assert isinstance(config.models, ModelConfig)

    def test_workflow_config_models_uses_env_vars(self):
        """WorkflowConfig.models should respect environment variables."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-opus-4",
                "CLASSIFIER_MODEL": "anthropic/claude-haiku-3-5",
            },
        ):
            config = WorkflowConfig()
            assert config.models.default_model == "anthropic/claude-opus-4"
            assert config.models.classifier_model == "anthropic/claude-haiku-3-5"


class TestModelConfigPriority:
    """Tests for model selection priority hierarchy."""

    def test_priority_node_specific_over_default(self):
        """Node-specific env var should override global default."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
                "CLASSIFIER_MODEL": "anthropic/claude-haiku-3-5",
            },
        ):
            config = ModelConfig()
            # Classifier should use specific config
            assert (
                config.get_model_for_node_type("ClassifierNode")
                == "anthropic/claude-haiku-3-5"
            )
            # Other node types should use default
            assert (
                config.get_model_for_node_type("SentimentNode")
                == "anthropic/claude-sonnet-4"
            )

    def test_priority_task_specific_over_default(self):
        """Task-specific env var should override global default."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
                "GENERATE_MODEL": "anthropic/claude-opus-4",
            },
        ):
            config = ModelConfig()
            # Generate should use specific config
            assert config.get_model_for_task("generate") == "anthropic/claude-opus-4"
            # Other tasks should use default
            assert config.get_model_for_task("analyze") == "anthropic/claude-sonnet-4"

    def test_mixed_configuration(self):
        """Should handle mixed configuration of node types and tasks."""
        with patch.dict(
            os.environ,
            {
                "POLICY_EVAL_MODEL": "anthropic/claude-sonnet-4",
                "CLASSIFIER_MODEL": "anthropic/claude-haiku-3-5",
                "GENERATE_MODEL": "anthropic/claude-opus-4",
                "DATA_EXTRACTOR_MODEL": "anthropic/claude-opus-4",
            },
        ):
            config = ModelConfig()

            # Check node types
            assert (
                config.get_model_for_node_type("ClassifierNode")
                == "anthropic/claude-haiku-3-5"
            )
            assert (
                config.get_model_for_node_type("DataExtractorNode")
                == "anthropic/claude-opus-4"
            )
            assert (
                config.get_model_for_node_type("SentimentNode")
                == "anthropic/claude-sonnet-4"
            )

            # Check tasks
            assert config.get_model_for_task("generate") == "anthropic/claude-opus-4"
            assert config.get_model_for_task("analyze") == "anthropic/claude-sonnet-4"
