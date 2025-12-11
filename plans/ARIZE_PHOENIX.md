# Arize Phoenix Integration

Optional observability/tracing support for LLM calls using [Arize Phoenix](https://github.com/Arize-ai/phoenix).

## Features

- **LLM Tracing**: All LiteLLM calls are traced with inputs, outputs, latency, and token usage
- **Local Development**: Phoenix runs via docker-compose
- **Zero Config Default**: Disabled by default, enable with a single env var
- **OpenTelemetry**: Uses standard OTLP protocol for trace export

## Quick Start

```bash
# 1. Install tracing dependencies
uv pip install -e ".[tracing]"

# 2. Start Phoenix
docker-compose up -d phoenix

# 3. Enable tracing
export PHOENIX_ENABLED=true

# 4. Run evaluation (traces sent to Phoenix)
uv run policy-eval eval -p policy.md -i "text to evaluate"

# 5. View traces at http://localhost:6007
```

## Configuration

Environment variables (set in `.env` or export):

| Variable | Default | Description |
|----------|---------|-------------|
| `PHOENIX_ENABLED` | `false` | Set to `true` to enable tracing |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6007` | Phoenix OTLP HTTP endpoint |
| `PHOENIX_PROJECT_NAME` | `policy-evaluator` | Project name in Phoenix UI |

## Architecture

```
┌─────────────────┐     OTLP/HTTP      ┌─────────────────┐
│  policy-eval    │ ─────────────────► │     Phoenix     │
│  (LiteLLM)      │                    │   (port 6007)   │
└─────────────────┘                    └─────────────────┘
        │                                      │
        │ arize_phoenix callback               │
        │ via OpenTelemetry                    ▼
        │                              ┌─────────────────┐
        └─────────────────────────────►│   Phoenix UI    │
                                       │  Traces, Spans  │
                                       └─────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Phoenix container definition |
| `src/policyflow/config.py` | `PhoenixConfig` class |
| `src/policyflow/llm.py` | `_init_tracing()` function |
| `pyproject.toml` | Optional `[tracing]` dependencies |

## Dependencies

Tracing requires the `[tracing]` optional dependencies:

```toml
[project.optional-dependencies]
tracing = [
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp>=1.20.0",
]
```

Install with: `uv pip install -e ".[tracing]"`

## Notes

- Phoenix UI runs on port 6007 (mapped from internal 6006) to avoid conflicts
- Traces are sent via HTTP using the `arize_phoenix` LiteLLM callback
- If Phoenix is not running, tracing errors are non-blocking (evaluation continues)
