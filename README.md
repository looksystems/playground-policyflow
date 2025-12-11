# Policyflow

> **NOTE**: This is an experimental/learning project only and is under active development.

An LLM-powered compliance evaluation framework that automatically parses structured policy documents (in markdown) and evaluates any text against the extracted criteria. The system uses AI to intelligently extract requirements, sub-criteria, and logical relationships from policies, then builds dynamic evaluation workflows that provide granular pass/fail results with confidence scores and reasoning for each criterion.

Ideal for financial regulation compliance, content moderation, contract analysis, or any domain requiring automated policy enforcement with explainable, auditable results.

## Features

- **Generic**: Works with any policy document in markdown format
- **Dynamic Workflows**: LLM parses policies and generates evaluation workflows
- **Granular Sub-criteria**: Evaluates sub-criteria individually with early termination
- **Confidence Gating**: Routes results based on confidence thresholds for human review
- **Model-agnostic**: Uses LiteLLM to support 100+ LLM providers
- **Configurable**: Environment-based configuration with `.env` support

## Installation

```bash
uv sync
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# Required: API key for your LLM provider
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Model selection (default: anthropic/claude-sonnet-4-20250514)
POLICY_EVAL_MODEL=anthropic/claude-sonnet-4-20250514

# Optional: Confidence thresholds
POLICY_EVAL_CONFIDENCE_HIGH=0.8   # Above this = high confidence
POLICY_EVAL_CONFIDENCE_LOW=0.5    # Below this = needs review
```

## Usage

### CLI

```bash
# Evaluate text against a policy
uv run policy-eval eval --policy policy.md --input "text to evaluate"

# Evaluate from file
uv run policy-eval eval --policy policy.md --input-file input.txt

# Parse and display policy structure
uv run policy-eval parse --policy policy.md

# Batch processing
uv run policy-eval batch --policy policy.md --inputs texts.jsonl --output results.jsonl

# Use a different model
uv run policy-eval eval --policy policy.md --input "..." --model openai/gpt-4o
```

### Python API

Run with `uv run python your_script.py` or in a `uv run python` REPL:

```python
from policyflow import evaluate

result = evaluate(
    input_text="Based on your risk profile, I recommend buying XYZ",
    policy_path="policy.md"
)

# Overall result
print(f"Policy satisfied: {result.policy_satisfied}")
print(f"Confidence: {result.overall_confidence:.0%}")
print(f"Needs review: {result.needs_review}")

# Per-criterion breakdown
for cr in result.criterion_results:
    status = "MET" if cr.met else "NOT MET"
    print(f"{cr.criterion_name}: {status} ({cr.confidence:.0%})")

    # Sub-criterion details (if any)
    for sub in cr.sub_results:
        sub_status = "MET" if sub.met else "NOT MET"
        print(f"  - {sub.sub_criterion_name}: {sub_status}")
```

### Custom Configuration

```python
from policyflow import evaluate, WorkflowConfig, ConfidenceGateConfig

config = WorkflowConfig(
    model="openai/gpt-4o",
    temperature=0.0,
    confidence_gate=ConfidenceGateConfig(
        high_threshold=0.9,  # Stricter high confidence
        low_threshold=0.6    # More lenient low threshold
    )
)

result = evaluate(
    input_text="...",
    policy_path="policy.md",
    config=config
)
```

## Architecture

The evaluator uses a dynamic workflow built with PocketFlow:

```
Policy.md → Parse → Extract Criteria
                         │
                         ▼
              ┌──────────────────────┐
              │  For each criterion  │
              │                      │
              │  Has sub-criteria?   │
              │    YES → SubCriterionNode chain with early termination
              │    NO  → CriterionEvaluationNode
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  ConfidenceGateNode  │
              │                      │
              │  Routes based on     │
              │  confidence levels   │
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  ResultAggregator    │
              │                      │
              │  Combines results    │
              │  with policy logic   │
              └──────────────────────┘
                         │
                         ▼
                EvaluationResult
```

## Output Structure

```python
EvaluationResult:
  policy_satisfied: bool      # Overall pass/fail
  overall_confidence: float   # 0.0-1.0
  confidence_level: str       # "high", "medium", "low"
  needs_review: bool          # Human review recommended?
  low_confidence_criteria: [] # IDs needing attention
  criterion_results: [
    CriterionResult:
      criterion_id: str
      criterion_name: str
      met: bool
      reasoning: str
      confidence: float
      sub_results: [          # For criteria with sub-criteria
        SubCriterionResult:
          sub_criterion_id: str
          sub_criterion_name: str
          met: bool
          reasoning: str
          confidence: float
      ]
  ]
```

## Observability (Optional)

Enable LLM tracing with [Arize Phoenix](https://github.com/Arize-ai/phoenix):

```bash
# Install tracing dependencies
uv pip install -e ".[tracing]"

# Start Phoenix
docker-compose up -d phoenix

# Enable tracing and run
PHOENIX_ENABLED=true uv run policy-eval eval -p policy.md -i "text"

# View traces at http://localhost:6007
```

See [ARIZE_PHOENIX.md](ARIZE_PHOENIX.md) for full documentation.

## Testing

Install dev dependencies:

```bash
uv sync --extra dev
```

Run tests:

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_criterion.py

# Run tests matching a pattern
uv run pytest -k "confidence"
```

The test suite covers:
- **Node types**: Criterion evaluation, sub-criterion logic, confidence gating, aggregation
- **Utilities**: Pattern matching, text transforms, classification, sampling

Tests use mocked LLM responses to run quickly without API calls.

## Tech Stack

- [PocketFlow](https://github.com/The-Pocket/PocketFlow) - LLM workflow framework
- [LiteLLM](https://github.com/BerriAI/litellm) - Model-agnostic LLM calls
- [Jinja2](https://jinja.palletsprojects.com/) - Prompt template management
- [Pydantic](https://pydantic.dev/) - Data validation
- [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) - CLI
- [python-dotenv](https://github.com/theskumar/python-dotenv) - Environment configuration
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) - LLM observability (optional)
