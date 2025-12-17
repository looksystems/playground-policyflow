# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolicyFlow is an LLM-powered compliance evaluation framework that parses structured policy documents (markdown) and evaluates text against extracted criteria. It's designed for financial regulation compliance, content moderation, contract analysis, and automated policy enforcement with explainable, auditable results.

**Status**: Experimental/learning project
**Language**: Python 3.11+
**Package Manager**: `uv`

## Common Commands

### Development Setup

```bash
uv sync                      # Install dependencies
uv sync --extra dev          # Install with dev dependencies
```

### CLI Usage

```bash
# Parse policy into workflow
uv run policyflow parse -p policy.md --save-workflow workflow.yaml

# Evaluate text against policy
uv run policyflow eval -p policy.md -i "text to evaluate"
uv run policyflow eval -w workflow.yaml -i "text"  # Using pre-parsed workflow

# Batch evaluation
uv run policyflow batch -p policy.md --inputs texts.yaml -o results.yaml
```

### Benchmarking & Optimization

```bash
# Generate test dataset
uv run policyflow generate-dataset --policy normalized.yaml -o golden_dataset.yaml

# Run benchmark
uv run policyflow benchmark --workflow workflow.yaml --dataset golden_dataset.yaml -o report.yaml

# Analyze failures
uv run policyflow analyze --report report.yaml --workflow workflow.yaml -o analysis.yaml

# Generate improvement hypotheses
uv run policyflow hypothesize --analysis analysis.yaml --workflow workflow.yaml -o hypotheses.yaml

# Automated optimization (hill-climbing)
uv run policyflow optimize --workflow workflow.yaml --dataset golden_dataset.yaml --max-iterations 10

# Quick test with limited data
uv run policyflow improve --workflow workflow.yaml --dataset golden_dataset.yaml --limit 1 --max-iterations 1
```

### Testing

```bash
uv run pytest                           # Run all tests
uv run pytest -v                        # Verbose output
uv run pytest tests/test_workflow_builder.py  # Specific file
uv run pytest -k "confidence"           # Tests matching pattern
```

### Observability (Optional)

```bash
docker-compose up -d phoenix            # Start Phoenix
PHOENIX_ENABLED=true uv run policyflow eval -p policy.md -i "text"
# View traces at http://localhost:6007
```

## Architecture

### Two-Step Parsing Pipeline

```
policy.md → normalize_policy() → NormalizedPolicy (YAML)
                ↓
        generate_workflow_from_normalized()
                ↓
        ParsedWorkflowPolicy (YAML with node graph)
                ↓
        DynamicWorkflowBuilder.run(input_text)
                ↓
        EvaluationResult (per-clause pass/fail, confidence, reasoning)
```

**Why two steps?**
- First step creates human-readable normalized YAML for review/editing
- Second step generates executable workflow with node IDs matching clause numbers (e.g., `clause_1_1_a`)
- Enables auditability and traceability of evaluation logic

### Node System

Nodes are the building blocks of evaluation workflows. They communicate via a **shared dictionary** containing input text, config, and evaluation state.

**Node Types**:
- **LLM Nodes**: `ClassifierNode`, `SentimentNode`, `DataExtractorNode`, `SamplerNode`
- **Deterministic Nodes**: `PatternMatchNode`, `KeywordScorerNode`, `TransformNode`, `LengthGateNode`, `ConfidenceGateNode`
- **Registry**: Dynamic node discovery via `src/policyflow/nodes/registry.py`

All nodes inherit from PocketFlow's `Node` class (prep → exec → post lifecycle).

**Node Creation Made Simple**:
- **@node_schema decorator**: Auto-generates `NodeSchema` from type hints and docstrings, eliminating ~85% of boilerplate
- **DeterministicNode base class**: Provides standard prep/post methods for simple deterministic nodes
- **CacheManager & RateLimiter**: Extracted from LLMNode for reusability and better separation of concerns

### Workflow Execution

The `DynamicWorkflowBuilder` (`src/policyflow/workflow_builder.py`) executes nodes based on routing logic. Nodes can route to different branches based on:
- Confidence thresholds (ConfidenceGateNode)
- Text patterns (PatternMatchNode)
- Text length (LengthGateNode)
- Classification results (ClassifierNode)

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

**Model Selection** (multi-level):
```env
POLICY_EVAL_MODEL=anthropic/claude-sonnet-4-20250514  # Global default

# Node type-specific (optional)
CLASSIFIER_MODEL=anthropic/claude-sonnet-4-20250514
SENTIMENT_MODEL=anthropic/claude-haiku-3-5-20250318
DATA_EXTRACTOR_MODEL=anthropic/claude-opus-4-5-20251101

# Task-specific for benchmarking (optional)
GENERATE_MODEL=anthropic/claude-opus-4-5-20251101
ANALYZE_MODEL=anthropic/claude-sonnet-4-20250514
OPTIMIZE_MODEL=anthropic/claude-sonnet-4-20250514
```

**Model Selection Priority**:
1. Explicit parameter in `workflow.yaml` or CLI `--model` flag
2. Type-specific env var (`CLASSIFIER_MODEL`, etc.)
3. Task-specific env var (`GENERATE_MODEL`, etc.)
4. Global default (`POLICY_EVAL_MODEL`)
5. Hardcoded fallback (`anthropic/claude-sonnet-4-20250514`)

**Other Key Settings**:
```env
POLICY_EVAL_CONFIDENCE_HIGH=0.8   # High confidence threshold
POLICY_EVAL_CONFIDENCE_LOW=0.5    # Low confidence threshold (below = needs review)
POLICY_EVAL_CACHE_ENABLED=true    # Enable LLM response caching
PHOENIX_ENABLED=false              # Enable Arize Phoenix tracing
```

See `.env.example` for complete list.

**Configuration System**:
- Uses **pydantic-settings** for type-safe environment variable loading
- Automatic type coercion (string → int, bool, etc.)
- Cross-field validation (e.g., high_threshold >= low_threshold)
- Can export JSON schema for documentation via `export_config_schema()`

## Key Directories

```
src/policyflow/
├── __init__.py              # Public API exports
├── models.py                # Core Pydantic models (Clause, Section, NormalizedPolicy, ParsedWorkflowPolicy, EvaluationResult)
├── parser.py                # Two-step parsing (normalize + generate_workflow)
├── workflow_builder.py      # Dynamic workflow execution engine
├── config.py                # Configuration management (pydantic-settings)
├── cli.py                   # CLI commands
├── llm.py                   # LiteLLM wrapper
├── cache.py                 # CacheManager (extracted from LLMNode)
├── rate_limiter.py          # RateLimiter (extracted from LLMNode)
├── nodes/                   # Node implementations (10 node types)
│   ├── registry.py          # Dynamic node discovery
│   ├── decorators.py        # @node_schema decorator
│   ├── base.py              # DeterministicNode base class
│   ├── llm_node.py          # Base class for LLM nodes
│   ├── classifier.py        # Text classification
│   ├── sentiment.py         # Sentiment analysis
│   ├── data_extractor.py    # Structured data extraction
│   ├── sampler.py           # Consensus from multiple evals
│   ├── confidence_gate.py   # Confidence-based routing
│   ├── pattern_match.py     # Regex matching
│   ├── keyword_scorer.py    # Weighted keyword scoring
│   ├── length_gate.py       # Text length validation
│   └── transform.py         # Text preprocessing
├── benchmark/               # Benchmarking & optimization
│   ├── models.py            # Golden dataset, metrics models
│   ├── generator.py         # Test case generation
│   ├── runner.py            # Benchmark execution
│   ├── analyzer.py          # Failure pattern analysis
│   ├── hypothesis.py        # Improvement hypotheses
│   ├── optimizer.py         # Hill-climbing optimization
│   ├── tracker.py           # Experiment tracking
│   └── cli.py               # Benchmark CLI
├── prompts/                 # Prompt template functions
└── templates/               # Jinja2 prompt templates

tests/
├── conftest.py              # Pytest fixtures (mock LLM responses, sample data)
├── test_*.py                # Node type tests (14 files)
├── benchmark/               # Benchmark system tests (13 files)
└── test_workflow_builder.py # Workflow execution tests

docs/
├── architecture.md          # System architecture diagrams
├── concepts.md              # Core concepts and design philosophy
├── USERGUIDE.md             # Complete user guide
├── cli-cheatsheet.md        # CLI reference
├── source-guide.md          # Extending the system
└── NODES.md                 # Node type reference

plans/                       # Design documents (10+ files)
```

## Design Principles

1. **Structured Intermediate Representations**: Two-step parsing creates human-readable YAML for auditability
2. **Explainable by Design**: Node IDs match clause numbers (`clause_1_1_a`) for full traceability
3. **Calibrated Uncertainty**: Explicit confidence scores with threshold-based routing
4. **Hybrid Evaluation**: Mix deterministic nodes (fast, predictable) with LLM nodes (semantic understanding)
5. **Continuous Improvement**: Integrated benchmark system for testing and optimization

## Testing Approach

- **Framework**: pytest + pytest-asyncio
- **Mocked LLM Responses**: Tests run fast without API calls
- **Comprehensive Fixtures**: Shared fixtures in `conftest.py` for mock responses, sample data, and test configs
- **Coverage**: 496 total tests (27 test files in `/tests/` + 13 in `/tests/benchmark/`)
- **Test-Driven Development**: Recent improvements added 100 new tests to ensure quality

## Common Patterns

### YAML Serialization

All models inherit from `YAMLMixin`:
```python
# Save to YAML
workflow.save_yaml("workflow.yaml")
result.save_yaml("result.yaml")

# Load from YAML
workflow = ParsedWorkflowPolicy.load_yaml("workflow.yaml")
result = EvaluationResult.load_yaml("result.yaml")
```

### Working with the Parser

```python
from policyflow.parser import normalize_policy, generate_workflow_from_normalized

# Step 1: Normalize (creates human-readable YAML)
normalized = normalize_policy(policy_text)
normalized.save_yaml("normalized.yaml")

# Step 2: Generate workflow (creates executable node graph)
workflow = generate_workflow_from_normalized(normalized)
workflow.save_yaml("workflow.yaml")
```

### Dynamic Node Discovery

The node registry (`src/policyflow/nodes/registry.py`) automatically discovers all node types by scanning the `nodes/` directory. New nodes are automatically available without modifying core code.

## Documentation

- **README.md**: Quick start, features, installation, configuration
- **docs/architecture.md**: System architecture, data flow, module organization
- **docs/concepts.md**: Core concepts, design philosophy, two-step parsing
- **docs/USERGUIDE.md**: Complete usage guide with examples
- **plans/BENCHMARK_SYSTEM.md**: Comprehensive benchmark system design
- **plans/ARIZE_PHOENIX.md**: Observability setup guide

## Tech Stack

- **PocketFlow**: LLM workflow framework (node execution lifecycle)
- **LiteLLM**: Model-agnostic LLM calls (100+ providers)
- **Pydantic**: Data validation and models
- **pydantic-settings**: Environment-based configuration
- **Typer + Rich**: CLI with styled output
- **Jinja2**: Prompt template management
- **pytest**: Testing framework
- **Arize Phoenix**: LLM observability (optional)
