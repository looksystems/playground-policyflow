# Policyflow User Guide

A generic policy evaluation tool that uses LLM-powered workflows to check if text satisfies policy criteria.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Writing Policies](#writing-policies)
- [CLI Reference](#cli-reference)
- [Two-Step Parser](#two-step-parser)
- [Python API](#python-api)
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
| `--format` | | Output format: `pretty` or `yaml` |

**Examples:**
```bash
# View parsed structure
policyflow parse -p policy.md

# Save workflow for later use
policyflow parse -p policy.md --save-workflow workflow.yaml

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

### normalize - Normalize policy document (Step 1)

```bash
policyflow normalize [OPTIONS]
```

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--policy` | `-p` | Path to policy markdown file (required) |
| `--output` | `-o` | Output YAML file path (required) |
| `--model` | `-m` | LiteLLM model identifier |
| `--format` | | Output format: `pretty` or `yaml` |

**Example:**
```bash
# Normalize a policy into structured format
policyflow normalize -p policy.md -o normalized.yaml
```

### generate-workflow - Generate workflow from normalized policy (Step 2)

```bash
policyflow generate-workflow [OPTIONS]
```

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--normalized` | `-n` | Path to normalized policy YAML (required) |
| `--output` | `-o` | Output workflow YAML file (required) |
| `--model` | `-m` | LiteLLM model identifier |
| `--format` | | Output format: `pretty` or `yaml` |

**Example:**
```bash
# Generate workflow from normalized policy
policyflow generate-workflow -n normalized.yaml -o workflow.yaml
```

### parse-two-step - Complete two-step parsing

```bash
policyflow parse-two-step [OPTIONS]
```

**Options:**
| Option | Short | Description |
|--------|-------|-------------|
| `--policy` | `-p` | Path to policy markdown file (required) |
| `--output-dir` | `-d` | Output directory for artifacts (required) |
| `--model` | `-m` | LiteLLM model identifier |
| `--prefix` | | Filename prefix for outputs (default: `policy`) |

**Example:**
```bash
# Parse policy in two steps, saving both artifacts
policyflow parse-two-step -p policy.md -d ./output/
# Creates: ./output/policy_normalized.yaml and ./output/policy_workflow.yaml
```

## Two-Step Parser

The two-step parser provides enhanced control, auditability, and explainability for policy parsing.

### Overview

```
Raw Policy Markdown
        ↓ (Step 1: Normalize)
NormalizedPolicy (YAML)  ← Review/edit before workflow generation
        ↓ (Step 2: Generate Workflow)
ParsedWorkflowPolicyV2 (YAML)  ← Node IDs match clause numbers
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
from policyflow.parser import (
    normalize_policy,
    generate_workflow_from_normalized,
    parse_policy_two_step,
)
from policyflow.models import NormalizedPolicy

# Step 1: Normalize
with open("policy.md") as f:
    normalized = normalize_policy(f.read())
normalized.save_yaml("normalized.yaml")

# Review/edit normalized.yaml if needed...

# Step 2: Generate workflow
normalized = NormalizedPolicy.load_yaml("normalized.yaml")
workflow = generate_workflow_from_normalized(normalized)
workflow.save_yaml("workflow.yaml")

# Or do both steps at once
workflow = parse_policy_two_step(
    policy_markdown,
    save_normalized="normalized.yaml"
)
```

### Result Traceability

Map evaluation results back to clause numbers:

```python
from policyflow.clause_mapping import (
    extract_clause_results,
    format_clause_results_report,
    summarize_results,
)

# After workflow execution
results = extract_clause_results(shared_store, normalized_policy)
report = format_clause_results_report(results)
print(report)
# [+] Clause 1.1: PASS (92% confidence)
#     Reasoning: Content addresses investor directly
#   [+] Clause 1.1.a: PASS (95% confidence)
#   [-] Clause 1.1.b: FAIL (88% confidence)

summary = summarize_results(results)
print(f"Pass rate: {summary['pass_rate']:.0%}")
print(f"Failed clauses: {summary['failed_clauses']}")
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
    PolicyEvaluationWorkflow,
    WorkflowConfig,
)

# Configure the workflow
config = WorkflowConfig(
    model="anthropic/claude-sonnet-4-20250514",
    temperature=0.0,
)

# Parse the policy
with open("policy.md") as f:
    policy = parse_policy(f.read(), config)

# Create workflow
workflow = PolicyEvaluationWorkflow(policy, config)

# Save for later use
workflow.save("workflow.yaml")

# Evaluate
result = workflow.run("Text to evaluate")

# Access detailed results
for criterion in result.criterion_results:
    print(f"{criterion.criterion_name}: {'MET' if criterion.met else 'NOT MET'}")
    print(f"  Reasoning: {criterion.reasoning}")
    print(f"  Confidence: {criterion.confidence:.0%}")
```

### Loading Cached Workflows

```python
from policyflow import PolicyEvaluationWorkflow, WorkflowConfig

config = WorkflowConfig()
workflow = PolicyEvaluationWorkflow.load("workflow.yaml", config)
result = workflow.run("Text to evaluate")
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
from policyflow import EvaluationResult, ParsedPolicy

# Save result to YAML
result.save_yaml("result.yaml")

# Load policy from YAML
policy = ParsedPolicy.load_yaml("workflow.yaml")

# Convert to YAML string
yaml_str = result.to_yaml()
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
input_text: "..."               # The evaluated text
policy_title: "Policy Name"     # Extracted policy title
overall_confidence: 0.85        # Average confidence (0.0-1.0)
overall_reasoning: "..."        # Summary explanation
confidence_level: high          # high, medium, or low
needs_review: false             # Flagged for human review?
low_confidence_criteria: []     # IDs of uncertain criteria

criterion_results:              # Per-criterion breakdown
  - criterion_id: criterion_1
    criterion_name: "First Criterion"
    met: true
    reasoning: "..."
    confidence: 0.9
    sub_results: []             # Sub-criterion results if any
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
- `low_confidence_criteria` is not empty

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

1. **Review flagged results**: Check `needs_review` and `low_confidence_criteria`
2. **Refine policy wording**: Ambiguous criteria cause low confidence
3. **Add context**: More detailed criterion descriptions help
4. **Consider sub-criteria**: Break complex criteria into simpler parts

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
