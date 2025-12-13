# Concepts & Workflow Primer

This document explains the core concepts and terminology used in PolicyFlow.

## What is PolicyFlow?

PolicyFlow automatically parses structured policy documents (in markdown) and evaluates any text against the extracted criteria. It's designed for:

- Financial regulation compliance
- Content moderation
- Contract analysis
- Any domain requiring automated policy enforcement with auditable results

## Core Terminology

### Policy Structure

```
Policy (markdown document)
  └── Section (logical grouping)
        └── Clause (evaluatable requirement)
              └── Sub-clause (nested requirement)
```

**Clause**: An individual evaluatable requirement with hierarchical numbering (e.g., `1`, `1.1`, `1.1.a`). Each clause has:
- `number`: Hierarchical identifier
- `text`: The requirement text
- `clause_type`: REQUIREMENT, DEFINITION, CONDITION, EXCEPTION, or REFERENCE
- `logic`: How sub-clauses combine (ALL or ANY)
- `sub_clauses`: Nested requirements

**Section**: A logical grouping of related clauses within a policy.

### Logic Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `ALL` | All sub-clauses must be satisfied (AND) | "must include (a) and (b)" |
| `ANY` | At least one sub-clause must be satisfied (OR) | "must include (a) or (b)" |

### Clause Types

| Type | Description | Typical Handling |
|------|-------------|------------------|
| `REQUIREMENT` | Must be evaluated/checked | PatternMatchNode, ClassifierNode |
| `DEFINITION` | Defines terms for context | Context only, not evaluated |
| `CONDITION` | If/then logic | ClassifierNode with routing |
| `EXCEPTION` | Exceptions to rules | Short-circuit evaluation |
| `REFERENCE` | References external docs | Context only |

### Confidence Levels

| Level | Threshold | Meaning |
|-------|-----------|---------|
| `HIGH` | ≥ 0.8 | High certainty in evaluation |
| `MEDIUM` | 0.5 - 0.8 | Moderate uncertainty |
| `LOW` | < 0.5 | Low confidence, needs review |

Results with `MEDIUM` or `LOW` confidence are flagged for human review via the `needs_review` field.

## Two-Step Parsing

PolicyFlow uses a two-step parsing process for maximum control and auditability:

```
┌─────────────────────┐
│  Raw Policy (.md)   │
└─────────┬───────────┘
          │
          ▼ Step 1: Normalize
┌─────────────────────┐
│ NormalizedPolicy    │  ← Human-reviewable YAML
│  - sections         │    Can be edited before
│  - clauses          │    workflow generation
│  - hierarchy        │
└─────────┬───────────┘
          │
          ▼ Step 2: Generate Workflow
┌─────────────────────┐
│ ParsedWorkflowPolicy│  ← Executable workflow
│  - nodes            │    Node IDs = clause numbers
│  - routes           │    (clause_1_1_a)
│  - hierarchy        │
└─────────┬───────────┘
          │
          ▼ Execution
┌─────────────────────┐
│  EvaluationResult   │  ← Per-clause results
│  - policy_satisfied │    with confidence and
│  - clause_results   │    reasoning
└─────────────────────┘
```

### Step 1: Normalization

Converts raw markdown to a structured `NormalizedPolicy`:

```python
from policyflow import normalize_policy

normalized = normalize_policy(policy_markdown)
normalized.save_yaml("normalized.yaml")  # Review/edit if needed
```

The normalized output preserves:
- Document hierarchy (sections → clauses → sub-clauses)
- Original clause text
- Inferred logic operators (ALL/ANY)
- Clause types

### Step 2: Workflow Generation

Converts `NormalizedPolicy` to executable `ParsedWorkflowPolicy`:

```python
from policyflow import generate_workflow_from_normalized

workflow = generate_workflow_from_normalized(normalized)
workflow.save_yaml("workflow.yaml")
```

Key feature: **Node IDs match clause numbers** for traceability:
- Clause `1.1` → Node ID `clause_1_1`
- Clause `1.1.a` → Node ID `clause_1_1_a`

### Combined Parsing

For convenience, both steps can run together:

```python
from policyflow import parse_policy

workflow = parse_policy(policy_markdown, save_normalized="normalized.yaml")
```

## Evaluation Workflow

```
┌─────────────────┐     ┌─────────────────┐
│   Input Text    │────▶│  Workflow       │
└─────────────────┘     │  (Node Graph)   │
                        └────────┬────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ clause_1_1    │    │ clause_1_2    │    │ clause_2_1    │
│ (Classifier)  │    │ (Pattern)     │    │ (Sentiment)   │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Aggregate       │
                    │ Results         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ EvaluationResult│
                    │ - satisfied     │
                    │ - confidence    │
                    │ - clause_results│
                    └─────────────────┘
```

## Node Types

Nodes are the building blocks of evaluation workflows. There are two categories:

### LLM-Based Nodes

Use LLM calls for semantic understanding:

| Node | Purpose |
|------|---------|
| `ClassifierNode` | Classify text into categories |
| `SentimentNode` | Analyze sentiment/tone |
| `DataExtractorNode` | Extract structured data |
| `SamplerNode` | Run multiple evaluations for consensus |

### Deterministic Nodes

No LLM required, fast and predictable:

| Node | Purpose |
|------|---------|
| `PatternMatchNode` | Regex/keyword pattern matching |
| `KeywordScorerNode` | Weighted keyword scoring |
| `TransformNode` | Text preprocessing |
| `LengthGateNode` | Route by text length |

### Internal Nodes

Used by the system for routing:

| Node | Purpose |
|------|---------|
| `ConfidenceGateNode` | Route based on confidence thresholds |

## Result Structure

```yaml
policy_satisfied: true           # Overall decision
overall_confidence: 0.87         # Average confidence
confidence_level: high           # HIGH, MEDIUM, or LOW
needs_review: false              # Flag for human review
low_confidence_clauses: []       # IDs of uncertain clauses

clause_results:
  - clause_id: clause_1_1
    clause_name: "Recipient Capacity"
    met: true
    confidence: 0.92
    reasoning: "Text addresses investor directly"
    sub_results:
      - clause_id: clause_1_1_a
        met: true
        confidence: 0.95
        reasoning: "Clear investor status reference"
```

## Benchmark System

PolicyFlow includes a comprehensive system for testing and optimizing workflows:

```
┌─────────────────┐
│ Golden Dataset  │  ← Test cases with expected results
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Benchmark       │  ← Run workflow against test cases
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analysis        │  ← Identify failure patterns
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hypothesize     │  ← Generate improvement ideas
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Optimize        │  ← Apply and test improvements
└─────────────────┘
```

See the [User Guide](USERGUIDE.md#benchmarking-api) for detailed usage.
