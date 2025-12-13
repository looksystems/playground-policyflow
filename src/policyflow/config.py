"""Configuration management using python-dotenv."""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field


def _find_dotenv() -> Path | None:
    """Find .env file by walking up from current directory."""
    current = Path.cwd()
    while current != current.parent:
        env_file = current / ".env"
        if env_file.exists():
            return env_file
        current = current.parent
    return None


# Load environment variables from .env file
_env_file = _find_dotenv()
if _env_file:
    load_dotenv(_env_file)


class CacheConfig(BaseModel):
    """Configuration for LLM response caching."""

    enabled: bool = Field(
        default_factory=lambda: os.getenv("POLICY_EVAL_CACHE_ENABLED", "true").lower()
        == "true",
        description="Whether caching is enabled",
    )
    ttl: int = Field(
        default_factory=lambda: int(os.getenv("POLICY_EVAL_CACHE_TTL", "3600")),
        ge=0,
        description="Cache TTL in seconds (0 = no expiration)",
    )
    directory: str = Field(
        default_factory=lambda: os.getenv("POLICY_EVAL_CACHE_DIR", ".cache"),
        description="Directory for cache files",
    )


class ThrottleConfig(BaseModel):
    """Configuration for LLM rate limiting."""

    enabled: bool = Field(
        default_factory=lambda: os.getenv("POLICY_EVAL_THROTTLE_ENABLED", "false").lower()
        == "true",
        description="Whether rate limiting is enabled",
    )
    requests_per_minute: int = Field(
        default_factory=lambda: int(
            os.getenv("POLICY_EVAL_THROTTLE_RPM", "60")
        ),
        ge=1,
        description="Maximum requests per minute",
    )


class ConfidenceGateConfig(BaseModel):
    """Configuration for confidence-based routing."""

    high_threshold: float = Field(
        default_factory=lambda: float(
            os.getenv("POLICY_EVAL_CONFIDENCE_HIGH", "0.8")
        ),
        ge=0.0,
        le=1.0,
        description="Confidence above this is high confidence",
    )
    low_threshold: float = Field(
        default_factory=lambda: float(
            os.getenv("POLICY_EVAL_CONFIDENCE_LOW", "0.5")
        ),
        ge=0.0,
        le=1.0,
        description="Confidence below this needs review",
    )


class PhoenixConfig(BaseModel):
    """Configuration for Arize Phoenix observability/tracing."""

    enabled: bool = Field(
        default_factory=lambda: os.getenv("PHOENIX_ENABLED", "false").lower() == "true",
        description="Enable Phoenix tracing (requires Phoenix server)",
    )
    endpoint: str = Field(
        default_factory=lambda: os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6007"
        ),
        description="Phoenix collector base URL (OTLP endpoint)",
    )
    project_name: str = Field(
        default_factory=lambda: os.getenv("PHOENIX_PROJECT_NAME", "policyflowuator"),
        description="Project name in Phoenix UI",
    )


class ModelConfig(BaseModel):
    """Configuration for model selection at different levels."""

    # Global default
    default_model: str = Field(
        default_factory=lambda: os.getenv(
            "POLICY_EVAL_MODEL",
            "anthropic/claude-sonnet-4-20250514"
        ),
        description="Global default model for all operations"
    )

    # Node type defaults
    classifier_model: str | None = Field(
        default_factory=lambda: os.getenv("CLASSIFIER_MODEL"),
        description="Default model for ClassifierNode"
    )
    data_extractor_model: str | None = Field(
        default_factory=lambda: os.getenv("DATA_EXTRACTOR_MODEL"),
        description="Default model for DataExtractorNode"
    )
    sentiment_model: str | None = Field(
        default_factory=lambda: os.getenv("SENTIMENT_MODEL"),
        description="Default model for SentimentNode"
    )
    sampler_model: str | None = Field(
        default_factory=lambda: os.getenv("SAMPLER_MODEL"),
        description="Default model for SamplerNode"
    )

    # CLI task defaults
    generate_model: str | None = Field(
        default_factory=lambda: os.getenv("GENERATE_MODEL"),
        description="Default model for generate-dataset command"
    )
    analyze_model: str | None = Field(
        default_factory=lambda: os.getenv("ANALYZE_MODEL"),
        description="Default model for analyze command"
    )
    hypothesize_model: str | None = Field(
        default_factory=lambda: os.getenv("HYPOTHESIZE_MODEL"),
        description="Default model for hypothesize command"
    )
    optimize_model: str | None = Field(
        default_factory=lambda: os.getenv("OPTIMIZE_MODEL"),
        description="Default model for optimize command"
    )

    def get_model_for_node_type(self, node_type: str) -> str:
        """Get model for a specific node type with fallback to default."""
        mapping = {
            "ClassifierNode": self.classifier_model,
            "DataExtractorNode": self.data_extractor_model,
            "SentimentNode": self.sentiment_model,
            "SamplerNode": self.sampler_model,
        }
        return mapping.get(node_type) or self.default_model

    def get_model_for_task(self, task: str) -> str:
        """Get model for a specific CLI task with fallback to default."""
        mapping = {
            "generate": self.generate_model,
            "analyze": self.analyze_model,
            "hypothesize": self.hypothesize_model,
            "optimize": self.optimize_model,
        }
        return mapping.get(task) or self.default_model


class WorkflowConfig(BaseModel):
    """Configuration for the evaluation workflow."""

    temperature: float = Field(
        default_factory=lambda: float(os.getenv("POLICY_EVAL_TEMPERATURE", "0.0")),
        description="LLM temperature",
    )
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("POLICY_EVAL_MAX_RETRIES", "3")),
        description="Max retries per node",
    )
    retry_wait: int = Field(
        default_factory=lambda: int(os.getenv("POLICY_EVAL_RETRY_WAIT", "2")),
        description="Seconds between retries",
    )
    confidence_gate: ConfidenceGateConfig = Field(
        default_factory=ConfidenceGateConfig,
        description="Confidence gate configuration",
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="LLM response cache configuration",
    )
    throttle: ThrottleConfig = Field(
        default_factory=ThrottleConfig,
        description="LLM rate limiting configuration",
    )
    phoenix: PhoenixConfig = Field(
        default_factory=PhoenixConfig,
        description="Phoenix observability configuration",
    )
    models: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model selection configuration",
    )


def get_config() -> WorkflowConfig:
    """Get the current workflow configuration."""
    return WorkflowConfig()
