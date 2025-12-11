"""LiteLLM wrapper utilities."""

import logging
import os
import re

import litellm


# Filter out misleading "Proxy Server is not installed" warning from LiteLLM
# Phoenix tracing works fine without the proxy server
class _ProxyWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Proxy Server is not installed" not in record.getMessage()


logging.getLogger("LiteLLM").addFilter(_ProxyWarningFilter())
import yaml
from litellm import completion
from .config import WorkflowConfig


_tracing_initialized = False


def _init_tracing(config: WorkflowConfig) -> None:
    """Initialize Phoenix tracing if enabled.

    This sets up the LiteLLM callback for Arize Phoenix.
    Only initializes once, even if called multiple times.
    """
    global _tracing_initialized
    if _tracing_initialized or not config.phoenix.enabled:
        return

    # Set environment variables for LiteLLM's OpenTelemetry callback
    # LiteLLM uses OTEL_EXPORTER_OTLP_ENDPOINT for the OTLP endpoint
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.phoenix.endpoint
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"

    # Enable the callback
    if "arize_phoenix" not in litellm.callbacks:
        litellm.callbacks.append("arize_phoenix")

    _tracing_initialized = True


def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    config: WorkflowConfig | None = None,
    yaml_response: bool = True,
    span_name: str | None = None,
) -> dict | str:
    """
    Call the LLM with the given prompt.

    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        model: LLM model identifier (required - no global fallback)
        config: Workflow configuration
        yaml_response: Whether to parse response as YAML
        span_name: Optional name for the trace span (for observability)

    Returns:
        Parsed YAML dict or raw string response

    Raises:
        ValueError: If model is not provided
    """
    if model is None:
        raise ValueError("model parameter is required - no global fallback")

    config = config or WorkflowConfig()

    # Initialize Phoenix tracing if enabled
    _init_tracing(config)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": config.temperature,
    }

    # Add metadata for tracing if span_name provided
    if span_name:
        kwargs["metadata"] = {
            "generation_name": span_name,
        }

    response = completion(**kwargs)
    content = response.choices[0].message.content

    if yaml_response:
        return extract_yaml(content)
    return content


def extract_yaml(text: str) -> dict:
    """Extract YAML from text that may contain markdown code blocks."""
    # Try to find YAML in code block
    code_block_match = re.search(r"```(?:ya?ml)?\s*([\s\S]*?)\s*```", text)
    if code_block_match:
        text = code_block_match.group(1)

    # Try to parse as YAML directly
    text = text.strip()
    return yaml.safe_load(text)
