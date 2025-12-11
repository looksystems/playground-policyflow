# Policyflow

> [!WARNING]  
> This is an experimental/learning project only and is under active development.

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
uv run policyflow [COMMAND] [OPTIONS]
```

#### Commands

##### `eval` - Evaluate text against a policy

```bash
uv run policyflow eval [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--policy PATH` | `-p` | Path to policy markdown file |
| `--workflow PATH` | `-w` | Path to pre-parsed workflow YAML (alternative to --policy) |
| `--input TEXT` | `-i` | Text to evaluate |
| `--input-file PATH` | `-f` | File containing text to evaluate |
| `--model TEXT` | `-m` | LiteLLM model identifier (e.g., `openai/gpt-4o`) |
| `--format TEXT` | | Output format: `pretty`, `yaml`, or `minimal` (default: `pretty`) |
| `--save-workflow PATH` | | Save parsed workflow to YAML file for reuse |

Examples:
```bash
# Evaluate inline text
uv run policyflow eval -p policy.md -i "text to evaluate"

# Evaluate from file
uv run policyflow eval -p policy.md -f input.txt

# Use a pre-parsed workflow (faster for repeated evaluations)
uv run policyflow eval -w workflow.yaml -i "text to evaluate"

# Use a different model and save the workflow
uv run policyflow eval -p policy.md -i "text" -m openai/gpt-4o --save-workflow workflow.yaml

# Get minimal output (just pass/fail and confidence)
uv run policyflow eval -p policy.md -i "text" --format minimal
```

##### `parse` - Parse and display policy structure

```bash
uv run policyflow parse [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--policy PATH` | `-p` | Path to policy markdown file (required) |
| `--model TEXT` | `-m` | LiteLLM model identifier |
| `--save-workflow PATH` | | Save parsed workflow to YAML file |
| `--format TEXT` | | Output format: `pretty` or `yaml` (default: `pretty`) |

Examples:
```bash
# Display policy structure
uv run policyflow parse -p policy.md

# Save workflow for later use
uv run policyflow parse -p policy.md --save-workflow workflow.yaml

# Output as YAML
uv run policyflow parse -p policy.md --format yaml
```

##### `batch` - Batch evaluate multiple inputs

```bash
uv run policyflow batch [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--policy PATH` | `-p` | Path to policy markdown file |
| `--workflow PATH` | `-w` | Path to pre-parsed workflow YAML |
| `--inputs PATH` | | YAML file with inputs list (required) |
| `--output PATH` | `-o` | Output YAML file (required) |
| `--model TEXT` | `-m` | LiteLLM model identifier |

Input file format (YAML):
```yaml
# List of strings
- "First text to evaluate"
- "Second text to evaluate"

# Or list of objects
- text: "First text to evaluate"
- input: "Second text to evaluate"
```

Examples:
```bash
# Batch evaluate from YAML
uv run policyflow batch -p policy.md --inputs texts.yaml -o results.yaml

# Use pre-parsed workflow for speed
uv run policyflow batch -w workflow.yaml --inputs texts.yaml -o results.yaml
```

##### `normalize` - Normalize policy into structured YAML (Step 1 of two-step parsing)

```bash
uv run policyflow normalize [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--policy PATH` | `-p` | Path to policy markdown file (required) |
| `--output PATH` | `-o` | Output YAML file path (required) |
| `--model TEXT` | `-m` | LiteLLM model identifier |
| `--format TEXT` | | Output format: `pretty` or `yaml` (default: `yaml`) |

Creates a normalized representation of the policy with hierarchical numbering (1, 1.1, 1.1.a style) that can be reviewed before workflow generation.

```bash
uv run policyflow normalize -p policy.md -o normalized.yaml
```

##### `generate-workflow` - Generate workflow from normalized policy (Step 2 of two-step parsing)

```bash
uv run policyflow generate-workflow [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--normalized PATH` | `-n` | Path to normalized policy YAML (required) |
| `--output PATH` | `-o` | Output workflow YAML file (required) |
| `--model TEXT` | `-m` | LiteLLM model identifier |
| `--format TEXT` | | Output format: `pretty` or `yaml` (default: `yaml`) |

```bash
uv run policyflow generate-workflow -n normalized.yaml -o workflow.yaml
```

##### `parse-two-step` - Complete two-step parsing in one command

```bash
uv run policyflow parse-two-step [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--policy PATH` | `-p` | Path to policy markdown file (required) |
| `--output-dir PATH` | `-d` | Output directory for artifacts (required) |
| `--model TEXT` | `-m` | LiteLLM model identifier |
| `--prefix TEXT` | | Filename prefix for outputs (default: `policy`) |

Creates both `{prefix}_normalized.yaml` and `{prefix}_workflow.yaml` in the output directory.

```bash
# Creates ./output/policy_normalized.yaml and ./output/policy_workflow.yaml
uv run policyflow parse-two-step -p policy.md -d ./output

# With custom prefix
uv run policyflow parse-two-step -p policy.md -d ./output --prefix my_policy
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
PHOENIX_ENABLED=true uv run policyflow eval -p policy.md -i "text"

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
