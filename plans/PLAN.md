# Policyflow Implementation Plan

## Status: Implemented

## Overview

Policyflow is a generic policy evaluation framework that parses any markdown policy document, dynamically generates an evaluation workflow using PocketFlow, and evaluates input text using LiteLLM.

## Tech Stack
- **uv** - Python package management
- **PocketFlow** - LLM workflow framework (Node/Flow graph abstraction)
- **LiteLLM** - Model-agnostic LLM calls
- **Jinja2** - Prompt template management
- **python-dotenv** - Environment configuration
- **Pydantic** - Data validation
- **Typer + Rich** - CLI interface
- **PyYAML** - YAML serialization throughout

## Project Structure

```
policyflow/
├── pyproject.toml
├── .env.example
├── README.md
├── PLAN.md
├── src/
│   └── policyflow/
│       ├── __init__.py              # Python API exports
│       ├── cli.py                   # CLI (Typer)
│       ├── config.py                # dotenv config + ConfidenceGateConfig
│       ├── models.py                # Pydantic models
│       ├── parser.py                # Policy parsing
│       ├── workflow.py              # Dynamic workflow generation
│       ├── llm.py                   # LiteLLM wrapper
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── criterion.py         # CriterionEvaluationNode
│       │   ├── subcriterion.py      # SubCriterionNode
│       │   ├── confidence_gate.py   # ConfidenceGateNode
│       │   └── aggregate.py         # ResultAggregatorNode
│       ├── prompts/
│       │   └── __init__.py          # Prompt builder functions
│       └── templates/
│           ├── __init__.py          # Jinja2 template loader
│           ├── policy_parser.j2     # Policy parsing prompt
│           ├── criterion_eval.j2    # Criterion evaluation prompt
│           └── subcriterion_eval.j2 # Sub-criterion evaluation prompt
└── tests/
```

## Configuration (.env)

```env
# LLM Configuration
POLICY_EVAL_MODEL=anthropic/claude-sonnet-4-20250514
POLICY_EVAL_TEMPERATURE=0.0

# API Keys (LiteLLM uses standard env vars)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional settings
POLICY_EVAL_MAX_RETRIES=3
POLICY_EVAL_RETRY_WAIT=2

# Confidence thresholds
POLICY_EVAL_CONFIDENCE_HIGH=0.8
POLICY_EVAL_CONFIDENCE_LOW=0.5
```

## Node Implementations

### CriterionEvaluationNode
Evaluates a single criterion (without sub-criteria) against input text.
- **prep()**: Read input_text and criterion from shared store
- **exec()**: Call LLM to evaluate if criterion is met
- **post()**: Store CriterionResult in shared["criterion_results"]

### SubCriterionNode
Evaluates individual sub-criteria separately for granular visibility.
- **prep()**: Read input_text, parent criterion, and specific sub-criterion
- **exec()**: Call LLM to evaluate the single sub-criterion
- **post()**: Store SubCriterionResult, return action based on sub_logic:
  - `"satisfied"`: Sub-criterion met AND logic is ANY (skip remaining)
  - `"failed"`: Sub-criterion not met AND logic is ALL (skip remaining)
  - `"default"`: Continue to next sub-criterion

### SubCriterionAggregatorNode
Aggregates sub-criterion results into a criterion result.
- **prep()**: Gather sub-criterion results for this criterion
- **exec()**: Apply sub_logic (ANY/ALL) to determine if criterion is met
- **post()**: Store CriterionResult with sub_results populated

### ConfidenceGateNode
Routes workflow based on confidence thresholds.
- **prep()**: Read criterion_results from shared store
- **exec()**: Check confidence levels against thresholds
- **post()**: Return action based on confidence:
  - `"high_confidence"`: All criteria above high threshold (0.8)
  - `"needs_review"`: Some criteria between thresholds
  - `"low_confidence"`: Any criterion below low threshold (0.5)

### ResultAggregatorNode
Aggregates all criterion results according to policy logic.
- **prep()**: Gather all criterion results
- **exec()**: Apply policy logic (ALL/ANY) to determine overall result
- **post()**: Store final EvaluationResult

## Workflow Architecture

```
┌─────────────────┐
│  Parse Policy   │  (LLM extracts criteria from markdown)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION FLOW                               │
│                                                                 │
│  ┌────────────────────────────────────────────┐                 │
│  │ Criterion 1 (with sub-criteria)            │                 │
│  │  ┌─────────┐   ┌─────────┐                 │                 │
│  │  │SubCrit  │──▶│SubCrit  │──▶ Aggregator   │                 │
│  │  │  (a)    │   │  (b)    │                 │                 │
│  │  └────┬────┘   └────┬────┘                 │                 │
│  │       │ satisfied   │ failed              │                 │
│  │       └─────────────┴──────────▶          │                 │
│  └────────────────────────────────────────────┘                 │
│                         │                                       │
│                         ▼                                       │
│  ┌────────────────────────────────────────────┐                 │
│  │ Criterion 2 (simple - no sub-criteria)     │                 │
│  │  ┌──────────────────┐                      │                 │
│  │  │ CriterionEvalNode│                      │                 │
│  │  └──────────────────┘                      │                 │
│  └────────────────────────────────────────────┘                 │
│                         │                                       │
│                         ▼                                       │
│                 ┌───────────────┐                                │
│                 │ConfidenceGate │                                │
│                 └───────┬───────┘                                │
│                         │                                       │
│         ┌───────────────┼───────────────┐                       │
│         ▼               ▼               ▼                       │
│   [high_conf]    [needs_review]   [low_conf]                    │
│         │               │               │                       │
│         └───────────────┴───────────────┘                       │
│                         │                                       │
│                         ▼                                       │
│                 ┌─────────────┐                                  │
│                 │ Aggregator  │                                  │
│                 │   Node      │                                  │
│                 └─────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│EvaluationResult │
│ + confidence    │
│ + needs_review  │
└─────────────────┘
```

## Data Models

```python
class SubCriterionResult(BaseModel):
    sub_criterion_id: str
    sub_criterion_name: str
    met: bool
    reasoning: str
    confidence: float  # 0.0-1.0

class CriterionResult(BaseModel):
    criterion_id: str
    criterion_name: str
    met: bool
    reasoning: str
    confidence: float  # 0.0-1.0
    sub_results: list[SubCriterionResult]  # Populated for criteria with sub-criteria

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class EvaluationResult(BaseModel):
    policy_satisfied: bool
    input_text: str
    policy_title: str
    criterion_results: list[CriterionResult]
    overall_reasoning: str
    overall_confidence: float
    confidence_level: ConfidenceLevel  # Classified confidence
    needs_review: bool                 # Human review recommended?
    low_confidence_criteria: list[str] # IDs of criteria needing attention
```

## Python API

```python
from policyflow import evaluate, parse_policy, ConfidenceLevel

# Simple usage
result = evaluate(
    input_text="Based on your risk profile, I recommend buying XYZ",
    policy_path="policy.md"
)

print(result.policy_satisfied)      # True/False
print(result.confidence_level)      # ConfidenceLevel.HIGH/MEDIUM/LOW
print(result.needs_review)          # True/False

# Check sub-criterion details
for cr in result.criterion_results:
    print(f"{cr.criterion_name}: {'MET' if cr.met else 'NOT MET'}")
    for sub in cr.sub_results:
        print(f"  - {sub.sub_criterion_name}: {'MET' if sub.met else 'NOT MET'}")

# With custom config
from policyflow import WorkflowConfig, ConfidenceGateConfig

config = WorkflowConfig(
    model="openai/gpt-4o",
    confidence_gate=ConfidenceGateConfig(
        high_threshold=0.9,
        low_threshold=0.6
    )
)
result = evaluate(input_text="...", policy_path="policy.md", config=config)

# Workflow caching
from policyflow import PolicyEvaluationWorkflow

# Save workflow for later use
policy = parse_policy(open("policy.md").read(), config)
workflow = PolicyEvaluationWorkflow(policy, config)
workflow.save("workflow.yaml")

# Load and run cached workflow
workflow = PolicyEvaluationWorkflow.load("workflow.yaml", config)
result = workflow.run("text to evaluate")

# YAML serialization (YAMLMixin methods)
result.save_yaml("result.yaml")           # Save to file
yaml_str = result.to_yaml()               # Convert to string
policy = ParsedPolicy.load_yaml("p.yaml") # Load from file
policy = ParsedPolicy.from_yaml(yaml_str) # Parse from string
```

## CLI Interface

```bash
# Evaluate text
policyflow eval --policy policy.md --input "text to evaluate"
policyflow eval --policy policy.md --input-file input.txt

# Parse and show policy structure
policyflow parse --policy policy.md

# Batch processing (YAML input/output)
policyflow batch --policy policy.md --inputs texts.yaml --output results.yaml

# Override model
policyflow eval --policy policy.md --input "..." --model openai/gpt-4o

# Output as YAML
policyflow eval --policy policy.md --input "..." --format yaml

# Save workflow to YAML for caching
policyflow parse --policy policy.md --save-workflow workflow.yaml

# Load cached workflow (skips policy parsing)
policyflow eval --workflow workflow.yaml --input "text to evaluate"
policyflow batch --workflow workflow.yaml --inputs texts.yaml --output results.yaml
```

## Workflow Caching

Parsed policies can be saved as YAML files to skip the LLM parsing step on subsequent runs:

```bash
# Parse once, save workflow
policyflow parse -p policy.md --save-workflow workflow.yaml

# Reuse cached workflow
policyflow eval -w workflow.yaml -i "text"
```

Benefits:
- **Faster evaluations**: Skip LLM policy parsing
- **Consistent results**: Same criteria extraction every time
- **Cost savings**: Fewer API calls
- **Offline capability**: Run without re-parsing

## YAML Format

All data interchange uses YAML:
- Workflow cache files
- Batch input files
- Batch output files
- CLI output (with `--format yaml`)
- LLM responses (templates request YAML output)
