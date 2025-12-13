# Policyflow User Guide

A generic policy evaluation tool that uses LLM-powered workflows to check if text satisfies policy criteria.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Writing Policies](#writing-policies)
- [CLI Reference](#cli-reference)
  - [Benchmark Commands](#benchmark-commands)
- [Two-Step Parser](#two-step-parser)
- [Python API](#python-api)
- [Benchmarking API](#benchmarking-api)
- [Configuration](#configuration)
- [Workflow Caching](#workflow-caching)
- [Understanding Results](#understanding-results)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Installation

```bash
# Using uv (recommended)
uv add policyflow

# Using pip
pip install policyflow
```

### Requirements

- Python 3.11+
- An LLM API key (Anthropic, OpenAI, or other LiteLLM-supported provider)

## Quick Start

### 1. Set up your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

### 2. Create a policy file

Create `policy.md`:
```markdown
A compliant message must:

(1) Include a clear disclaimer
(2) Not make guarantees about:
    (a) future performance; or
    (b) specific returns
(3) Be appropriate for the target audience
```

### 3. Evaluate text

```bash
policyflow eval -p policy.md -i "This investment could lose value. Past performance is not indicative of future results."
```

## Writing Policies

Policies are written in markdown format. The evaluator uses an LLM to parse the policy into structured criteria.

### Basic Structure

```markdown
A [subject] must/should/is defined as:

(1) First criterion description
(2) Second criterion description
(3) Third criterion description
```

### Sub-criteria

Use lettered sub-items for OR/AND logic within a criterion:

```markdown
(1) The message must include:
    (a) a risk warning; or
    (b) a disclaimer; or
    (c) both
```

### Logic Keywords

The parser recognizes these keywords:
- **AND logic**: "and", "all of", "must all"
- **OR logic**: "or", "any of", "either"

### Example Policies

#### Simple Policy
```markdown
An acceptable response must:
(1) Be factually accurate
(2) Be professional in tone
(3) Address the user's question directly
```

#### Complex Policy with Sub-criteria
```markdown
A personal recommendation is one that:

(1) is made to a person as an investor or potential investor
(2) recommends any of the following actions:
    (a) buying or selling a security; or
    (b) holding an investment; or
    (c) exercising investment rights
(3) is either:
    (a) presented as suitable for the person; or
    (b) based on the person's circumstances
(4) is not issued exclusively to the public
```

## CLI Reference

### eval - Evaluate text against a policy

```bash
policyflow eval [OPTIONS]
```

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--policy` | `-p` | Path to policy markdown file |
| `--workflow` | `-w` | Path to pre-parsed workflow YAML |
| `--input` | `-i` | Text to evaluate |
| `--input-file` | `-f` | File containing text to evaluate |
| `--model` | `-m` | LiteLLM model identifier |
| `--format` | | Output format: `pretty`, `yaml`, `minimal` |
| `--save-workflow` | | Save parsed workflow to YAML file |

**Examples:**
```bash
# Basic evaluation
policyflow eval -p policy.md -i "Your text here"

# Read input from file
policyflow eval -p policy.md -f input.txt

# Use cached workflow
policyflow eval -w workflow.yaml -i "Your text here"

# Output as YAML
policyflow eval -p policy.md -i "Text" --format yaml

# Use a specific model
policyflow eval -p policy.md -i "Text" -m openai/gpt-4o
```

### parse - Parse and display policy structure

```bash
policyflow parse [OPTIONS]
```

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--policy` | `-p` | Path to policy markdown file (required) |
| `--model` | `-m` | LiteLLM model identifier |
| `--save-workflow` | | Save parsed workflow to YAML file |
| `--save-normalized` | | Save intermediate normalized policy to YAML |
| `--format` | | Output format: `pretty` or `yaml` |

**Examples:**
```bash
# View parsed structure
policyflow parse -p policy.md

# Save workflow for later use
policyflow parse -p policy.md --save-workflow workflow.yaml

# Save both normalized and workflow files
policyflow parse -p policy.md --save-normalized norm.yaml --save-workflow workflow.yaml

# Output as YAML
policyflow parse -p policy.md --format yaml
```

### batch - Batch evaluate multiple inputs

```bash
policyflow batch [OPTIONS]
```

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--policy` | `-p` | Path to policy markdown file |
| `--workflow` | `-w` | Path to pre-parsed workflow YAML |
| `--inputs` | | YAML file with inputs list |
| `--output` | `-o` | Output YAML file |
| `--model` | `-m` | LiteLLM model identifier |

**Input file format (YAML):**
```yaml
- "First text to evaluate"
- "Second text to evaluate"
- "Third text to evaluate"
```

**Example:**
```bash
policyflow batch -w workflow.yaml --inputs texts.yaml -o results.yaml
```

### Benchmark Commands

The benchmark system provides commands for testing and improving workflow accuracy.

#### generate-dataset - Generate test dataset from policy

```bash
policyflow generate-dataset [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--policy` | Path to normalized policy YAML (required) |
| `--output` | Output path for golden dataset (required) |
| `--cases-per-criterion` | Number of test cases per criterion (default: 3) |
| `--include-edge-cases` | Include edge case test cases |
| `--strategies` | Edge case strategies (comma-separated) |
| `--mode` | Generation mode: `template`, `llm`, `hybrid` |
| `--model` | LLM model for hybrid/llm mode |

**Example:**
```bash
policyflow generate-dataset --policy normalized.yaml \
    --cases-per-criterion 5 \
    --include-edge-cases \
    --output golden_dataset.yaml
```

#### benchmark - Run benchmark against test dataset

```bash
policyflow benchmark [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--workflow` | Path to workflow YAML (required) |
| `--dataset` | Path to golden dataset YAML (required) |
| `--output` | Output path for benchmark report |
| `--model` | LLM model for evaluation |

**Example:**
```bash
policyflow benchmark --workflow workflow.yaml --dataset golden_dataset.yaml --output report.yaml
```

#### analyze - Analyze benchmark failures

```bash
policyflow analyze [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--report` | Path to benchmark report (required) |
| `--workflow` | Path to workflow YAML (required) |
| `--output` | Output path for analysis report |
| `--mode` | Analysis mode: `rule_based`, `llm`, `hybrid` |
| `--model` | LLM model for hybrid/llm mode |

**Example:**
```bash
policyflow analyze --report report.yaml --workflow workflow.yaml --mode hybrid --output analysis.yaml
```

#### hypothesize - Generate improvement hypotheses

```bash
policyflow hypothesize [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--analysis` | Path to analysis report (required) |
| `--workflow` | Path to workflow YAML (required) |
| `--output` | Output path for hypotheses |
| `--mode` | Generation mode: `template`, `llm`, `hybrid` |
| `--model` | LLM model for hybrid/llm mode |

**Example:**
```bash
policyflow hypothesize --analysis analysis.yaml --workflow workflow.yaml --output hypotheses.yaml
```

#### optimize - Automated workflow optimization

```bash
policyflow optimize [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--workflow` | Path to workflow YAML (required) |
| `--dataset` | Path to golden dataset (required) |
| `--output` | Output path for optimized workflow (required) |
| `--max-iterations` | Maximum optimization iterations (default: 10) |
| `--max-llm-calls` | Maximum LLM calls budget |
| `--target-accuracy` | Stop when accuracy reaches this value |
| `--patience` | Stop after N iterations without improvement |
| `--model` | LLM model for analysis/hypothesis |

**Example:**
```bash
policyflow optimize --workflow workflow.yaml --dataset golden_dataset.yaml \
    --max-iterations 10 \
    --target-accuracy 0.95 \
    --patience 3 \
    --output optimized_workflow.yaml
```

#### improve - Full improvement loop (benchmark + analyze + hypothesize)

```bash
policyflow improve [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--workflow` | Path to workflow YAML (required) |
| `--dataset` | Path to golden dataset (required) |
| `--mode` | Analysis mode: `rule_based`, `llm`, `hybrid` |
| `--model` | LLM model for analysis/hypothesis |

**Example:**
```bash
policyflow improve --workflow workflow.yaml --dataset golden_dataset.yaml --mode hybrid
```

#### experiments - Manage experiment history

```bash
policyflow experiments [SUBCOMMAND]
```

**Subcommands:**
- `list` - List all experiments
- `compare EXP1 EXP2` - Compare two experiments
- `best` - Show best performing experiment

**Example:**
```bash
policyflow experiments list
policyflow experiments compare baseline exp_001
policyflow experiments best
```

## Two-Step Parser

The two-step parser provides enhanced control, auditability, and explainability for policy parsing.

### Overview

```
Raw Policy Markdown
        ↓ (Step 1: Normalize)
NormalizedPolicy (YAML)  ← Review/edit before workflow generation
        ↓ (Step 2: Generate Workflow)
ParsedWorkflowPolicy (YAML)  ← Node IDs match clause numbers
```

### Why Two Steps?

1. **Auditability**: Review the normalized policy before workflow generation
2. **Explainability**: Node IDs match clause numbers (e.g., `clause_1_1_a`)
3. **Editability**: Manually adjust the normalized policy if needed
4. **Traceability**: Map evaluation results back to specific clauses

### Numbering Convention

The normalizer uses hierarchical numbering:
- Sections: `1`, `2`, `3`
- Clauses: `1.1`, `1.2`, `2.1`
- Sub-clauses: `1.1.1`, `1.1.2`
- Deep sub-clauses: `1.1.1.a`, `1.1.1.b` (letters at depth 3+)

### Normalized Policy Structure

```yaml
title: Personal Recommendation Definition
version: "1.0"
description: Defines what constitutes a personal recommendation
sections:
  - number: "1"
    title: Definition Requirements
    logic: all
    clauses:
      - number: "1.1"
        title: Recipient Capacity
        text: "is made to a person in their capacity as:"
        clause_type: requirement
        logic: any
        sub_clauses:
          - number: "1.1.a"
            title: Investor Status
            text: "an investor or potential investor"
            clause_type: requirement
          - number: "1.1.b"
            title: Agent Status
            text: "agent for an investor"
            clause_type: requirement
```

### Clause Types

| Type | Description | Typical Node |
|------|-------------|--------------|
| `requirement` | Must be evaluated/checked | PatternMatchNode, ClassifierNode |
| `definition` | Defines terms | Context only |
| `condition` | If/then logic | ClassifierNode with routing |
| `exception` | Exceptions to rules | Short-circuit nodes |
| `reference` | External references | Context only |

### Workflow with Hierarchy

The generated workflow includes a `hierarchy` field mapping nodes to clauses:

```yaml
workflow:
  nodes:
    - id: clause_1_1_a      # Matches clause 1.1.a
      type: ClassifierNode
      # ...
  hierarchy:
    - clause_number: "1.1"
      clause_text: "is made to a person..."
      nodes: ["clause_1_1_a", "clause_1_1_b"]
      logic: any
      sub_groups:
        - clause_number: "1.1.a"
          nodes: ["clause_1_1_a"]
```

### Python API for Two-Step Parsing

```python
from policyflow import (
    parse_policy,
    normalize_policy,
    generate_workflow_from_normalized,
    NormalizedPolicy,
)

# Step 1: Normalize
with open("policy.md") as f:
    normalized = normalize_policy(f.read())
normalized.save_yaml("normalized.yaml")

# Review/edit normalized.yaml if needed...

# Step 2: Generate workflow
normalized = NormalizedPolicy.load_yaml("normalized.yaml")
workflow = generate_workflow_from_normalized(normalized)
workflow.save_yaml("workflow.yaml")

# Or do both steps at once with parse_policy()
workflow = parse_policy(
    policy_markdown,
    save_normalized="normalized.yaml"
)
```

### Result Traceability

Map evaluation results back to clause numbers:

```python
from policyflow import EvaluationResult

# After workflow execution, access the result
result: EvaluationResult = shared["result"]

# Per-clause breakdown
for cr in result.clause_results:
    status = "PASS" if cr.met else "FAIL"
    print(f"Clause {cr.clause_id}: {status} ({cr.confidence:.0%})")
    print(f"  Reasoning: {cr.reasoning}")

# Example output:
# Clause 1.1: PASS (92%)
#   Reasoning: Content addresses investor directly
# Clause 1.1.a: PASS (95%)
#   Reasoning: Clear investor status reference
# Clause 1.1.b: FAIL (88%)
#   Reasoning: No agent relationship mentioned
```

## Python API

### Basic Usage

```python
from policyflow import evaluate

result = evaluate(
    input_text="Your text to evaluate",
    policy_path="policy.md"
)

print(f"Satisfied: {result.policy_satisfied}")
print(f"Confidence: {result.overall_confidence:.0%}")
```

### Advanced Usage

```python
from policyflow import (
    parse_policy,
    DynamicWorkflowBuilder,
    WorkflowConfig,
)

# Configure the workflow
config = WorkflowConfig(
    model="anthropic/claude-sonnet-4-20250514",
    temperature=0.0,
)

# Parse the policy
with open("policy.md") as f:
    workflow = parse_policy(f.read(), config)

# Save for later use
workflow.save_yaml("workflow.yaml")

# Create builder and evaluate
builder = DynamicWorkflowBuilder(workflow, config)
shared = builder.run("Text to evaluate")
result = shared["result"]

# Access detailed results
for clause in result.clause_results:
    print(f"{clause.clause_name}: {'MET' if clause.met else 'NOT MET'}")
    print(f"  Reasoning: {clause.reasoning}")
    print(f"  Confidence: {clause.confidence:.0%}")
```

### Loading Cached Workflows

```python
from policyflow import DynamicWorkflowBuilder, WorkflowConfig
from policyflow.models import ParsedWorkflowPolicy

config = WorkflowConfig()
workflow = ParsedWorkflowPolicy.load_yaml("workflow.yaml")
builder = DynamicWorkflowBuilder(workflow, config)
shared = builder.run("Text to evaluate")
result = shared["result"]
```

### Controlling Max Iterations

Workflows have built-in protection against infinite loops. By default, execution stops after 100 node iterations:

```python
from policyflow import DynamicWorkflowBuilder

builder = DynamicWorkflowBuilder(policy, config)

# Use default limit (100 iterations)
result = builder.run("Text to evaluate")

# Custom limit for complex workflows
result = builder.run("Text to evaluate", max_iterations=200)

# Strict limit for simple workflows
result = builder.run("Text to evaluate", max_iterations=50)
```

If the limit is exceeded, a `RuntimeError` is raised with a message indicating a possible infinite loop.

### YAML Serialization

```python
from policyflow import EvaluationResult
from policyflow.models import ParsedWorkflowPolicy

# Save result to YAML
result.save_yaml("result.yaml")

# Load workflow from YAML
workflow = ParsedWorkflowPolicy.load_yaml("workflow.yaml")

# Convert to YAML string
yaml_str = result.to_yaml()
```

## Benchmarking API

The benchmark system provides a Python API for measuring and improving workflow accuracy.

### Running Benchmarks

```python
from policyflow.benchmark import (
    load_golden_dataset,
    SimpleBenchmarkRunner,
    BenchmarkConfig,
)
from policyflow.models import ParsedWorkflowPolicy

# Load dataset and workflow
dataset = load_golden_dataset("golden_dataset.yaml")
workflow = ParsedWorkflowPolicy.load_yaml("workflow.yaml")

# Configure and run benchmark
config = BenchmarkConfig(workflow_id=workflow.title)
runner = SimpleBenchmarkRunner(config)
report = runner.run(workflow, dataset.test_cases)

# View results
print(f"Overall Accuracy: {report.metrics.overall_accuracy:.2%}")
for crit_id, metrics in report.metrics.criterion_metrics.items():
    print(f"  {crit_id}: P={metrics.precision:.2f} R={metrics.recall:.2f} F1={metrics.f1:.2f}")
```

### Analyzing Failures

```python
from policyflow.benchmark import create_analyzer

# Create analyzer (rule-based, llm, or hybrid)
analyzer = create_analyzer(
    mode="hybrid",  # or "rule_based", "llm"
    model="anthropic/claude-sonnet-4-20250514"  # optional, for hybrid/llm mode
)

# Analyze the benchmark report
analysis = analyzer.analyze(report, workflow)

# View patterns
for pattern in analysis.patterns:
    print(f"[{pattern.severity}] {pattern.pattern_type}: {pattern.description}")

# View recommendations
for rec in analysis.recommendations:
    print(f"  - {rec}")
```

### Generating Hypotheses

```python
from policyflow.benchmark import create_hypothesis_generator

# Create hypothesis generator
generator = create_hypothesis_generator(
    mode="hybrid",  # or "template", "llm"
    model="anthropic/claude-sonnet-4-20250514"  # optional
)

# Generate improvement hypotheses
hypotheses = generator.generate(analysis, workflow)

for h in hypotheses:
    print(f"[{h.change_type}] {h.description}")
    print(f"  Target: {h.target}")
    print(f"  Change: {h.suggested_change}")
    print(f"  Rationale: {h.rationale}")
```

### Automated Optimization

```python
from policyflow.benchmark import (
    HillClimbingOptimizer,
    OptimizationBudget,
    create_analyzer,
    create_hypothesis_generator,
    BasicHypothesisApplier,
)

# Configure budget
budget = OptimizationBudget(
    max_iterations=10,
    max_llm_calls=100,
    target_metric=0.95,  # Stop when accuracy reaches 95%
    patience=3  # Stop after 3 iterations without improvement
)

# Create optimizer
optimizer = HillClimbingOptimizer(
    analyzer=create_analyzer(mode="hybrid"),
    hypothesis_generator=create_hypothesis_generator(mode="hybrid"),
    hypothesis_applier=BasicHypothesisApplier()
)

# Run optimization
result = optimizer.optimize(
    workflow=workflow,
    dataset=dataset,
    budget=budget,
    metric=lambda r: r.metrics.overall_accuracy
)

# View results
print(f"Converged: {result.converged} ({result.convergence_reason})")
print(f"Best Accuracy: {result.best_metric:.2%}")
print(f"Iterations: {len(result.history)}")

# Save optimized workflow
result.best_workflow.save_yaml("optimized_workflow.yaml")
```

### Experiment Tracking

```python
from policyflow.benchmark import FileBasedExperimentTracker, Experiment
from pathlib import Path
from datetime import datetime

# Create tracker
tracker = FileBasedExperimentTracker(Path("experiments/"))

# Record experiment
experiment = Experiment(
    id="exp_001",
    timestamp=datetime.now(),
    workflow_snapshot=workflow.to_yaml(),
    hypothesis_applied=None,  # or the hypothesis that was applied
    benchmark_report=report,
    parent_experiment_id=None  # or parent experiment ID for lineage
)
tracker.record(experiment)

# Get history and best
history = tracker.get_history()
best = tracker.get_best()
print(f"Best experiment: {best.id} with accuracy {best.accuracy:.2%}")

# Compare experiments
comparison = tracker.compare("exp_001", "exp_002")
print(f"Accuracy diff: {comparison['accuracy_diff']:.2%}")
```

### Generating Test Datasets

```python
from policyflow.benchmark import create_generator, GeneratorConfig
from policyflow.models import NormalizedPolicy

# Load normalized policy
policy = NormalizedPolicy.load_yaml("normalized.yaml")

# Configure generator
config = GeneratorConfig(
    cases_per_criterion=5,
    include_edge_cases=True,
    edge_case_strategies=["boundary", "negation", "implicit"],
    mode="hybrid"
)

# Create generator
generator = create_generator(
    mode="hybrid",
    model="anthropic/claude-sonnet-4-20250514"  # optional
)

# Generate dataset
dataset = generator.generate(policy, config)

# Save dataset
dataset.save_yaml("golden_dataset.yaml")
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POLICY_EVAL_MODEL` | `anthropic/claude-sonnet-4-20250514` | LiteLLM model identifier |
| `POLICY_EVAL_TEMPERATURE` | `0.0` | LLM temperature (0.0-1.0) |
| `POLICY_EVAL_MAX_RETRIES` | `3` | Max retries on LLM failure |
| `POLICY_EVAL_RETRY_WAIT` | `2` | Seconds between retries |
| `POLICY_EVAL_CONFIDENCE_HIGH` | `0.8` | High confidence threshold |
| `POLICY_EVAL_CONFIDENCE_LOW` | `0.5` | Low confidence threshold |

### .env File

Create a `.env` file in your project directory:

```env
ANTHROPIC_API_KEY=sk-ant-...
POLICY_EVAL_MODEL=anthropic/claude-sonnet-4-20250514
POLICY_EVAL_TEMPERATURE=0.0
POLICY_EVAL_CONFIDENCE_HIGH=0.85
POLICY_EVAL_CONFIDENCE_LOW=0.6
```

### Supported Models

Any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers) can be used:

```bash
# Anthropic
policyflow eval -p policy.md -i "text" -m anthropic/claude-sonnet-4-20250514

# OpenAI
policyflow eval -p policy.md -i "text" -m openai/gpt-4o

# Azure OpenAI
policyflow eval -p policy.md -i "text" -m azure/gpt-4
```

## Workflow Caching

Caching workflows improves performance and consistency by skipping the policy parsing step.

### Benefits

1. **Faster evaluations**: Skip the LLM call to parse the policy
2. **Consistent results**: Same criteria extraction every time
3. **Cost savings**: Fewer API calls for repeated evaluations
4. **Offline capability**: Run evaluations without re-parsing

### Workflow

```bash
# Step 1: Parse policy once and save
policyflow parse -p policy.md --save-workflow workflow.yaml

# Step 2: Use cached workflow for all subsequent evaluations
policyflow eval -w workflow.yaml -i "text 1"
policyflow eval -w workflow.yaml -i "text 2"
policyflow batch -w workflow.yaml --inputs texts.yaml -o results.yaml
```

### When to Regenerate

Regenerate the workflow when:
- The policy document changes
- You want different criterion extraction
- You upgrade to a better parsing model

## Understanding Results

### Result Structure

```yaml
policy_satisfied: true          # Overall policy result
policy_title: "Policy Name"     # Extracted policy title
overall_confidence: 0.85        # Average confidence (0.0-1.0)
overall_reasoning: "..."        # Summary explanation
confidence_level: high          # high, medium, or low
needs_review: false             # Flagged for human review?
low_confidence_clauses: []      # IDs of uncertain clauses

clause_results:                 # Per-clause breakdown
  - clause_id: clause_1_1
    clause_name: "First Clause"
    met: true
    reasoning: "..."
    confidence: 0.9
    sub_results: []             # Sub-clause results if any
```

### Confidence Levels

| Level | Threshold | Meaning |
|-------|-----------|---------|
| `high` | ≥ 0.8 | All criteria evaluated with high confidence |
| `medium` | 0.5 - 0.8 | Some criteria have moderate uncertainty |
| `low` | < 0.5 | At least one criterion has low confidence |

### Review Flags

Results are flagged for review when:
- `confidence_level` is `low` or `medium`
- `needs_review` is `true`
- `low_confidence_clauses` is not empty

## Best Practices

### Writing Effective Policies

1. **Be specific**: Vague criteria lead to inconsistent results
2. **Use clear structure**: Numbered criteria with lettered sub-items
3. **Include examples**: Help the LLM understand edge cases
4. **Test incrementally**: Evaluate sample texts as you write

### Optimizing Performance

1. **Cache workflows**: Always save and reuse parsed workflows
2. **Batch when possible**: Process multiple inputs in one call
3. **Choose appropriate models**: Smaller models for simple policies
4. **Set temperature to 0**: For consistent, deterministic results

### Handling Low Confidence

1. **Review flagged results**: Check `needs_review` and `low_confidence_clauses`
2. **Refine policy wording**: Ambiguous clauses cause low confidence
3. **Add context**: More detailed clause descriptions help
4. **Consider sub-clauses**: Break complex clauses into simpler parts

### Production Deployment

1. **Pre-parse policies**: Generate workflow YAML files at build time
2. **Version workflows**: Track changes to parsed policies
3. **Monitor confidence**: Alert on consistently low confidence scores
4. **Log results**: Keep evaluation history for auditing

## Troubleshooting

### Workflow Validation Warnings

When building workflows, you may see these warnings:

**"Workflow has no terminal nodes"**
- The workflow has no nodes with empty `routes: {}`
- This may cause the workflow to run indefinitely
- Fix: Ensure at least one node has `routes: {}` to end the workflow

**"Workflow contains cycles"**
- The workflow graph has circular references (A -> B -> A)
- This may cause the workflow to loop forever
- Fix: Review the node routing to ensure there's a path to a terminal node

### RuntimeError: Workflow exceeded N iterations

This error indicates a possible infinite loop. Solutions:
1. Check for terminal nodes in the workflow
2. Review node routing for unintended cycles
3. Increase `max_iterations` if the workflow legitimately needs more steps:
   ```python
   result = builder.run("text", max_iterations=200)
   ```
