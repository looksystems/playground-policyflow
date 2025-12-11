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
        default_factory=lambda: os.getenv("PHOENIX_PROJECT_NAME", "policy-evaluator"),
        description="Project name in Phoenix UI",
    )


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


def get_config() -> WorkflowConfig:
    """Get the current workflow configuration."""
    return WorkflowConfig()
