# Policyflow

> [!WARNING]
> This is an experimental/learning project only and is under active development.

An LLM-powered compliance evaluation framework that automatically parses structured policy documents (in markdown) and evaluates any text against the extracted criteria. The system uses AI to intelligently extract requirements, sub-criteria, and logical relationships from policies, then builds dynamic evaluation workflows that provide granular pass/fail results with confidence scores and reasoning for each criterion.

Ideal for financial regulation compliance, content moderation, contract analysis, or any domain requiring automated policy enforcement with explainable, auditable results.

## Features

- **Generic**: Works with any policy document in markdown format
- **Two-Step Parsing**: Normalizes policy then generates workflow for auditability
- **Explainable**: Node IDs match clause numbers for full traceability
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

### All Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POLICY_EVAL_MODEL` | `anthropic/claude-sonnet-4-20250514` | LiteLLM model identifier |
| `POLICY_EVAL_TEMPERATURE` | `0.0` | LLM temperature for evaluation |
| `POLICY_EVAL_CONFIDENCE_HIGH` | `0.8` | High confidence threshold |
| `POLICY_EVAL_CONFIDENCE_LOW` | `0.5` | Low confidence threshold (below = needs review) |
| `POLICY_EVAL_MAX_RETRIES` | `3` | Max retry attempts per LLM call |
| `POLICY_EVAL_RETRY_WAIT` | `2` | Seconds between retries |
| `POLICY_EVAL_CACHE_ENABLED` | `true` | Enable LLM response caching |
| `POLICY_EVAL_CACHE_TTL` | `3600` | Cache TTL in seconds (0 = no expiration) |
| `POLICY_EVAL_CACHE_DIR` | `.cache` | Directory for cache files |
| `POLICY_EVAL_THROTTLE_ENABLED` | `false` | Enable rate limiting |
| `POLICY_EVAL_THROTTLE_RPM` | `60` | Max requests per minute |
| `PHOENIX_ENABLED` | `false` | Enable Arize Phoenix tracing |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6007` | Phoenix collector URL |
| `PHOENIX_PROJECT_NAME` | `policyflowuator` | Project name in Phoenix UI |

## Usage

### CLI

```bash
uv run policyflow [COMMAND] [OPTIONS]
```

#### Commands

##### `parse` - Parse policy into executable workflow

```bash
uv run policyflow parse [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--policy PATH` | `-p` | Path to policy markdown file (required) |
| `--model TEXT` | `-m` | LiteLLM model identifier |
| `--save-workflow PATH` | | Save parsed workflow to YAML file |
| `--save-normalized PATH` | | Save intermediate normalized policy to YAML |
| `--format TEXT` | | Output format: `pretty` or `yaml` (default: `pretty`) |

Examples:
```bash
# Display policy structure
uv run policyflow parse -p policy.md

# Save workflow for later use
uv run policyflow parse -p policy.md --save-workflow workflow.yaml

# Save both normalized and workflow files
uv run policyflow parse -p policy.md --save-normalized norm.yaml --save-workflow workflow.yaml

# Output as YAML
uv run policyflow parse -p policy.md --format yaml
```

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

# Per-clause breakdown
for cr in result.clause_results:
    status = "MET" if cr.met else "NOT MET"
    print(f"{cr.clause_name}: {status} ({cr.confidence:.0%})")
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

### API Reference

The main package exports the following:

#### Functions

| Function | Description |
|----------|-------------|
| `evaluate()` | Main entry point - evaluate text against a policy |
| `parse_policy()` | Parse policy markdown into a `ParsedWorkflowPolicy` object |
| `normalize_policy()` | Parse policy into normalized structure (step 1) |
| `generate_workflow_from_normalized()` | Generate workflow from normalized policy (step 2) |
| `get_config()` | Get current `WorkflowConfig` from environment |

#### Classes

| Class | Description |
|-------|-------------|
| `DynamicWorkflowBuilder` | Workflow runner for evaluating text against a parsed workflow |
| `WorkflowConfig` | Configuration for evaluation (model, retries, cache, etc.) |
| `ConfidenceGateConfig` | Confidence threshold configuration |

#### Data Models

| Model | Description |
|-------|-------------|
| `NormalizedPolicy` | Normalized policy with sections and clauses |
| `ParsedWorkflowPolicy` | Parsed workflow with hierarchy |
| `EvaluationResult` | Complete evaluation result |
| `ClauseResult` | Result for a single clause |
| `Clause` | A single clause from a policy |
| `Section` | A section containing clauses |

#### Enums

| Enum | Values | Description |
|------|--------|-------------|
| `LogicOperator` | `ALL`, `ANY` | How criteria combine (AND/OR) |
| `ConfidenceLevel` | `HIGH`, `MEDIUM`, `LOW` | Confidence classification |
| `ClauseType` | `REQUIREMENT`, `DEFINITION`, `CONDITION`, `EXCEPTION`, `REFERENCE` | Clause type |

#### Utilities

| Utility | Description |
|---------|-------------|
| `YAMLMixin` | Mixin providing `to_yaml()`, `from_yaml()`, `save_yaml()`, `load_yaml()` |

## Architecture

The evaluator uses a two-step parsing process:

```
Policy.md → Normalize → NormalizedPolicy (YAML)
                              │
                              ▼
              Generate Workflow from Normalized
                              │
                              ▼
              ParsedWorkflowPolicy (YAML)
               - nodes with clause_X_X IDs
               - hierarchy mapping
                              │
                              ▼
              ┌──────────────────────┐
              │  DynamicWorkflow     │
              │                      │
              │  Executes nodes      │
              │  based on routes     │
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
  low_confidence_clauses: []  # IDs needing attention
  clause_results: [
    ClauseResult:
      clause_id: str
      clause_name: str
      met: bool
      reasoning: str
      confidence: float
      sub_results: [...]      # Nested clause results
  ]
```

## Advanced Usage

### Direct Workflow Control

For more control over the evaluation process:

```python
from policyflow import parse_policy, DynamicWorkflowBuilder, WorkflowConfig

# Parse policy (uses two-step process internally)
policy_text = open("policy.md").read()
parsed_workflow = parse_policy(policy_text)

# Create workflow builder and run evaluations
config = WorkflowConfig()
builder = DynamicWorkflowBuilder(parsed_workflow, config)

texts = ["First text to evaluate", "Second text to evaluate"]
results = [builder.run(text) for text in texts]
```

### Working with Normalized Policies

```python
from policyflow.parser import normalize_policy, generate_workflow_from_normalized
from policyflow.models import NormalizedPolicy

# Step 1: Normalize
normalized = normalize_policy(open("policy.md").read())
normalized.save_yaml("normalized.yaml")

# Review/edit normalized.yaml if needed...

# Step 2: Generate workflow
normalized = NormalizedPolicy.load_yaml("normalized.yaml")
workflow = generate_workflow_from_normalized(normalized)
workflow.save_yaml("workflow.yaml")
```

### YAML Serialization

All data models support YAML serialization via `YAMLMixin`:

```python
from policyflow import parse_policy, evaluate

# Save parsed workflow for reuse
workflow = parse_policy(open("policy.md").read())
workflow.save_yaml("workflow.yaml")

# Save evaluation results
result = evaluate(input_text="...", policy_path="policy.md")
result.save_yaml("evaluation_result.yaml")

# Load from YAML
from policyflow.models import ParsedWorkflowPolicy, EvaluationResult
workflow = ParsedWorkflowPolicy.load_yaml("workflow.yaml")
result = EvaluationResult.load_yaml("evaluation_result.yaml")
```

### Available Node Types

The workflow system includes node types for building evaluation pipelines:

| Node | Description |
|------|-------------|
| `LLMNode` | Base node for LLM-powered evaluation |
| `ConfidenceGateNode` | Routes based on confidence thresholds |
| `TransformNode` | Transforms input text (lowercase, truncate, etc.) |
| `LengthGateNode` | Routes based on text length |
| `KeywordScorerNode` | Scores text based on keyword presence |
| `PatternMatchNode` | Matches text against regex patterns |
| `DataExtractorNode` | Extracts structured data from text |
| `SamplerNode` | Runs multiple evaluations for consensus |
| `ClassifierNode` | Classifies text into categories |
| `SentimentNode` | Analyzes text sentiment |

Access nodes via:
```python
from policyflow.nodes import (
    PatternMatchNode,
    ClassifierNode,
    # ... etc
)
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

See [plans/ARIZE_PHOENIX.md](plans/ARIZE_PHOENIX.md) for full documentation.

## Benchmarking & Self-Improvement

Policyflow includes a comprehensive benchmarking system for measuring and improving workflow accuracy.

### Quick Start

```bash
# Generate test dataset from normalized policy
uv run policyflow generate-dataset --policy normalized.yaml --output golden_dataset.yaml

# Run benchmark against the dataset
uv run policyflow benchmark --workflow workflow.yaml --dataset golden_dataset.yaml --output report.yaml

# Analyze failures and get improvement recommendations
uv run policyflow analyze --report report.yaml --workflow workflow.yaml --output analysis.yaml

# Generate hypotheses for improvement
uv run policyflow hypothesize --analysis analysis.yaml --workflow workflow.yaml --output hypotheses.yaml

# Or run the full improvement loop at once
uv run policyflow improve --workflow workflow.yaml --dataset golden_dataset.yaml
```

### Automated Optimization

```bash
# Run optimization with budget constraints
uv run policyflow optimize --workflow workflow.yaml --dataset golden_dataset.yaml \
    --max-iterations 10 \
    --target-accuracy 0.95 \
    --output optimized_workflow.yaml
```

### Python API

```python
from policyflow.benchmark import (
    load_golden_dataset,
    SimpleBenchmarkRunner,
    BenchmarkConfig,
    create_analyzer,
    create_hypothesis_generator,
    HillClimbingOptimizer,
    OptimizationBudget,
)

# Load dataset and workflow
dataset = load_golden_dataset("golden_dataset.yaml")
workflow = load_workflow("workflow.yaml")

# Run benchmark
runner = SimpleBenchmarkRunner(BenchmarkConfig())
report = runner.run(workflow, dataset.test_cases)
print(f"Accuracy: {report.metrics.overall_accuracy:.2%}")

# Analyze failures (with optional LLM enhancement)
analyzer = create_analyzer(mode="hybrid", model="anthropic/claude-sonnet-4-20250514")
analysis = analyzer.analyze(report, workflow)

# Generate improvement hypotheses
generator = create_hypothesis_generator(mode="hybrid", model="anthropic/claude-sonnet-4-20250514")
hypotheses = generator.generate(analysis, workflow)

for h in hypotheses:
    print(f"- [{h.change_type}] {h.description}")
```

### Features

- **Golden Dataset Generation**: Template-based and LLM-enhanced test case generation
- **Comprehensive Metrics**: Per-criterion accuracy, precision, recall, F1, and confidence calibration
- **Failure Analysis**: Rule-based and LLM-enhanced pattern detection
- **Hypothesis Generation**: Actionable improvement suggestions with template and LLM modes
- **Automated Optimization**: Hill-climbing optimizer with configurable budget constraints
- **Experiment Tracking**: YAML-based tracking with history and comparison

See [plans/BENCHMARK_SYSTEM.md](plans/BENCHMARK_SYSTEM.md) for full documentation.

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
uv run pytest tests/test_workflow_builder.py

# Run tests matching a pattern
uv run pytest -k "confidence"
```

The test suite covers:
- **Node types**: Confidence gating, pattern matching, classification, etc.
- **Workflow builder**: Validation, max iterations, routing

Tests use mocked LLM responses to run quickly without API calls.

## Tech Stack

- [PocketFlow](https://github.com/The-Pocket/PocketFlow) - LLM workflow framework
- [LiteLLM](https://github.com/BerriAI/litellm) - Model-agnostic LLM calls
- [Jinja2](https://jinja.palletsprojects.com/) - Prompt template management
- [Pydantic](https://pydantic.dev/) - Data validation
- [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) - CLI
- [python-dotenv](https://github.com/theskumar/python-dotenv) - Environment configuration
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) - LLM observability (optional)
