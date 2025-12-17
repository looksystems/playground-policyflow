# Architecture Guide

This document describes PolicyFlow's system architecture, data flow, and component relationships.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PolicyFlow                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │    CLI       │    │  Python API  │    │   Benchmark System   │  │
│  │  (cli.py)    │    │ (__init__.py)│    │   (benchmark/)       │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                   │                       │               │
│         └───────────────────┼───────────────────────┘               │
│                             │                                        │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      Core Layer                               │   │
│  │  ┌────────────┐  ┌─────────────────┐  ┌──────────────────┐  │   │
│  │  │   Parser   │  │ WorkflowBuilder │  │     Config       │  │   │
│  │  │ (parser.py)│  │(workflow_builder)│ │   (config.py)    │  │   │
│  │  └────────────┘  └─────────────────┘  └──────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                             │                                        │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      Node System                              │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │   │
│  │  │   LLM Nodes     │  │ Deterministic   │  │   Registry   │  │   │
│  │  │ (classifier,    │  │ (pattern_match, │  │ (registry.py)│  │   │
│  │  │  sentiment...)  │  │  transform...)  │  │              │  │   │
│  │  └─────────────────┘  └─────────────────┘  └──────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                             │                                        │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    External Dependencies                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │   │
│  │  │PocketFlow│  │ LiteLLM  │  │ Pydantic │  │   Phoenix   │  │   │
│  │  │(workflow)│  │  (LLM)   │  │ (models) │  │(observation)│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Pipeline

```
                        ┌─────────────────────────────────┐
                        │       Policy Document           │
                        │          (markdown)             │
                        └───────────────┬─────────────────┘
                                        │
                                        ▼
                  ┌─────────────────────────────────────────────┐
                  │            Step 1: normalize_policy()       │
                  │                                             │
                  │  - Parse markdown structure                 │
                  │  - Extract sections/clauses                 │
                  │  - Assign hierarchical numbering            │
                  │  - Infer logic operators (ALL/ANY)          │
                  └───────────────────┬─────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────────┐
                        │       NormalizedPolicy          │
                        │           (YAML)                │
                        │                                 │
                        │  - sections[].clauses[]         │
                        │  - hierarchical structure       │
                        │  - can be reviewed/edited       │
                        └───────────────┬─────────────────┘
                                        │
                                        ▼
            ┌─────────────────────────────────────────────────────────┐
            │          Step 2: generate_workflow_from_normalized()    │
            │                                                         │
            │  - Create node configurations                           │
            │  - Map node IDs to clause numbers                       │
            │  - Build routing between nodes                          │
            │  - Generate workflow hierarchy                          │
            └─────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────────┐
                        │     ParsedWorkflowPolicy        │
                        │           (YAML)                │
                        │                                 │
                        │  - workflow.nodes[]             │
                        │  - workflow.hierarchy[]         │
                        │  - workflow.start_node          │
                        └───────────────┬─────────────────┘
                                        │
                                        ▼
            ┌─────────────────────────────────────────────────────────┐
            │             DynamicWorkflowBuilder.build()              │
            │                                                         │
            │  Phase 1: Instantiate nodes from NodeConfig             │
            │  Phase 2: Wire up routes between nodes                  │
            │  Validate: Check for terminals, detect cycles           │
            └─────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────────┐
                        │      PocketFlow Flow            │
                        │    (executable workflow)        │
                        └───────────────┬─────────────────┘
                                        │
                                        │ + input_text
                                        ▼
            ┌─────────────────────────────────────────────────────────┐
            │              DynamicWorkflowBuilder.run()               │
            │                                                         │
            │  - Initialize shared store with input_text              │
            │  - Execute node graph                                   │
            │  - Collect clause results                               │
            │  - Aggregate into EvaluationResult                      │
            └─────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────────┐
                        │       EvaluationResult          │
                        │                                 │
                        │  - policy_satisfied: bool       │
                        │  - overall_confidence: float    │
                        │  - clause_results[]             │
                        │  - needs_review: bool           │
                        └─────────────────────────────────┘
```

## Module Organization

```
src/policyflow/
├── __init__.py              # Public API exports
├── models.py                # Core Pydantic data models
├── parser.py                # Two-step policy parsing
├── workflow_builder.py      # Dynamic workflow construction
├── config.py                # Configuration management (pydantic-settings)
├── llm.py                   # LLM communication (LiteLLM wrapper)
├── cli.py                   # Command-line interface
├── clause_mapping.py        # Clause/policy mapping utilities
├── numbering.py             # Clause numbering utilities
├── cache.py                 # CacheManager (extracted from LLMNode)
├── rate_limiter.py          # RateLimiter (extracted from LLMNode)
│
├── nodes/                   # Processing pipeline nodes
│   ├── __init__.py          # Node registration and exports
│   ├── schema.py            # Node self-documentation schemas
│   ├── registry.py          # Dynamic node discovery
│   ├── decorators.py        # @node_schema decorator
│   ├── base.py              # DeterministicNode base class
│   ├── llm_node.py          # Base class for LLM nodes
│   ├── pattern_match.py     # Regex pattern matching
│   ├── classifier.py        # LLM text classification
│   ├── sentiment.py         # LLM sentiment analysis
│   ├── keyword_scorer.py    # Weighted keyword scoring
│   ├── confidence_gate.py   # Confidence-based routing
│   ├── length_gate.py       # Text length validation
│   ├── data_extractor.py    # LLM data extraction
│   ├── sampler.py           # Multiple evaluation sampling
│   └── transform.py         # Text preprocessing
│
├── benchmark/               # Benchmarking and optimization
│   ├── __init__.py          # Benchmark exports
│   ├── models.py            # Golden dataset, metrics models
│   ├── generator.py         # Test case generation
│   ├── runner.py            # Benchmark execution
│   ├── analyzer.py          # Results analysis
│   ├── optimizer.py         # Workflow optimization
│   ├── tracker.py           # Experiment tracking
│   └── cli.py               # Benchmark CLI commands
│
├── prompts/                 # LLM prompt management
│   └── __init__.py          # Prompt template functions
│
└── templates/               # Jinja2 templates
    ├── classifier.j2
    ├── sentiment.j2
    └── data_extractor.j2
```

## Node System Architecture

### Node Lifecycle

Every node follows the PocketFlow lifecycle pattern:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Node Execution                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐                                                    │
│  │ shared dict │  Contains: input_text, prior results, config       │
│  └──────┬──────┘                                                    │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────────────────────────────────────────┐                │
│  │              prep(shared) -> prep_res           │                │
│  │                                                 │                │
│  │  - Extract needed data from shared store        │                │
│  │  - Prepare input for execution                  │                │
│  │  - Example: {"text": shared["input_text"]}      │                │
│  └──────────────────────┬──────────────────────────┘                │
│                         │                                            │
│                         ▼                                            │
│  ┌─────────────────────────────────────────────────┐                │
│  │              exec(prep_res) -> exec_res         │                │
│  │                                                 │                │
│  │  - Core processing logic (stateless)            │                │
│  │  - Pattern matching, LLM call, etc.             │                │
│  │  - No side effects on shared store              │                │
│  └──────────────────────┬──────────────────────────┘                │
│                         │                                            │
│                         ▼                                            │
│  ┌─────────────────────────────────────────────────┐                │
│  │     post(shared, prep_res, exec_res) -> action  │                │
│  │                                                 │                │
│  │  - Store results in shared store                │                │
│  │  - Return action string for routing             │                │
│  │  - Example: "matched" or "not_matched"          │                │
│  └──────────────────────┬──────────────────────────┘                │
│                         │                                            │
│                         ▼                                            │
│  ┌─────────────────────────────────────────────────┐                │
│  │                  Route to Next Node             │                │
│  │                                                 │                │
│  │  - Look up action in routes dict                │                │
│  │  - If found: go to routes[action]               │                │
│  │  - If not found: end execution                  │                │
│  └─────────────────────────────────────────────────┘                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Node Class Hierarchy

```
pocketflow.Node (base)
│
├── LLMNode (policyflow)
│   │   - Provides caching, rate limiting
│   │   - call_llm() method
│   │
│   ├── ClassifierNode
│   │       Categories + descriptions -> category name
│   │
│   ├── SentimentNode
│   │       Text -> positive/negative/neutral/mixed
│   │
│   ├── DataExtractorNode
│   │       Schema -> extracted structured data
│   │
│   └── SamplerNode
│           N evaluations -> consensus/majority/split
│
└── Direct Node subclasses (deterministic)
    │
    ├── PatternMatchNode
    │       Patterns + mode -> matched/not_matched
    │
    ├── KeywordScorerNode
    │       Weighted keywords -> high/medium/low
    │
    ├── TransformNode
    │       Operations -> default (preprocessed text)
    │
    ├── LengthGateNode
    │       Min/max bounds -> valid/invalid
    │
    ├── ConfidenceGateNode
    │       Thresholds -> high_confidence/needs_review/low_confidence
    │
    └── SamplerNode
            Probabilities -> selected route
```

### Node Registry

Nodes register themselves with a schema for discovery:

```python
# Registration pattern
@register_node
class MyNode(Node):
    parser_schema = NodeSchema(
        name="MyNode",
        description="What it does",
        category="deterministic",  # or "llm", "internal"
        parameters=[...],
        actions=["action1", "action2"],
        parser_exposed=True,  # Include in parser prompts
    )
```

Registry functions:
- `register_node(cls)`: Register a node class
- `get_node_class(name)`: Retrieve by name
- `get_all_nodes()`: Get all registered nodes
- `get_parser_schemas()`: Get schemas for parser

## Class Relationships

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Data Models                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  NormalizedPolicy ─────────────────┐                                │
│    │                               │                                │
│    ├── sections: list[Section]     │                                │
│    │     │                         │  parse_policy()                │
│    │     └── clauses: list[Clause] │       │                        │
│    │           │                   │       │                        │
│    │           └── sub_clauses     │       │                        │
│    │                               │       ▼                        │
│    │                      ┌────────────────────────────┐            │
│    │                      │   ParsedWorkflowPolicy     │            │
│    │                      │                            │            │
│    │                      │   ├── workflow             │            │
│    │                      │   │   ├── nodes[]          │            │
│    │                      │   │   ├── start_node       │            │
│    │                      │   │   └── hierarchy[]      │            │
│    │                      │   │                        │            │
│    └──────────────────────┼───┤                        │            │
│           reference       │   └────────────────────────┘            │
│                           │               │                         │
│                           │               │ DynamicWorkflowBuilder  │
│                           │               ▼                         │
│                           │      ┌─────────────────┐                │
│                           │      │  Flow (runtime) │                │
│                           │      │   ├── nodes{}   │                │
│                           │      │   └── routes    │                │
│                           │      └────────┬────────┘                │
│                           │               │                         │
│                           │               │ run(input_text)         │
│                           │               ▼                         │
│                           │      ┌─────────────────────┐            │
│                           │      │  EvaluationResult   │            │
│                           │      │   ├── satisfied     │            │
│                           │      │   ├── confidence    │            │
│                           │      │   └── clause_results│            │
│                           │      │         │           │            │
│                           │      │         └── ClauseResult[]       │
│                           │      └─────────────────────┘            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          Configuration                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  WorkflowConfig                                                     │
│    ├── model: str                                                   │
│    ├── temperature: float                                           │
│    ├── max_retries: int                                             │
│    ├── confidence_gate: ConfidenceGateConfig                        │
│    │     ├── high_threshold: float (default: 0.8)                   │
│    │     └── low_threshold: float (default: 0.5)                    │
│    ├── cache: CacheConfig                                           │
│    │     ├── enabled: bool                                          │
│    │     ├── ttl: int                                               │
│    │     └── directory: str                                         │
│    ├── throttle: ThrottleConfig                                     │
│    │     ├── enabled: bool                                          │
│    │     └── requests_per_minute: int                               │
│    └── phoenix: PhoenixConfig                                       │
│          ├── enabled: bool                                          │
│          ├── endpoint: str                                          │
│          └── project_name: str                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Shared Store Data Flow

During workflow execution, nodes communicate via a shared dictionary:

```
shared = {
    # Input
    "input_text": "Text to evaluate",
    "workflow_config": WorkflowConfig(...),

    # Node results (stored by post())
    "clause_1_1_result": {
        "met": True,
        "confidence": 0.92,
        "reasoning": "...",
    },
    "clause_1_2_result": {...},

    # Specialized results
    "classification": {
        "category": "compliant",
        "confidence": 0.88,
    },
    "sentiment": {
        "label": "neutral",
        "confidence": 0.75,
    },
    "pattern_match_result": {
        "is_matched": True,
        "matched_patterns": ["disclaimer"],
    },

    # Final output
    "result": EvaluationResult(...)
}
```

## Benchmark System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Benchmark System                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────┐         ┌────────────────┐                      │
│  │ NormalizedPolicy│────────▶│   Generator    │                      │
│  └────────────────┘         │                │                      │
│                             │ - Template     │                      │
│                             │ - LLM          │                      │
│                             │ - Hybrid       │                      │
│                             └───────┬────────┘                      │
│                                     │                               │
│                                     ▼                               │
│                             ┌────────────────┐                      │
│                             │ GoldenDataset  │                      │
│                             │ - test_cases[] │                      │
│                             │ - expected     │                      │
│                             └───────┬────────┘                      │
│                                     │                               │
│  ┌────────────────┐                 │                               │
│  │ Workflow       │─────────────────┤                               │
│  └────────────────┘                 │                               │
│                                     ▼                               │
│                             ┌────────────────┐                      │
│                             │    Runner      │                      │
│                             │ (benchmark)    │                      │
│                             └───────┬────────┘                      │
│                                     │                               │
│                                     ▼                               │
│                             ┌────────────────┐                      │
│                             │BenchmarkReport │                      │
│                             │ - metrics      │                      │
│                             │ - failures     │                      │
│                             └───────┬────────┘                      │
│                                     │                               │
│                                     ▼                               │
│                             ┌────────────────┐                      │
│                             │   Analyzer     │                      │
│                             │ - Rule-based   │                      │
│                             │ - LLM          │                      │
│                             │ - Hybrid       │                      │
│                             └───────┬────────┘                      │
│                                     │                               │
│                                     ▼                               │
│                             ┌────────────────┐                      │
│                             │AnalysisReport  │                      │
│                             │ - patterns     │                      │
│                             │ - recommendations│                    │
│                             └───────┬────────┘                      │
│                                     │                               │
│                                     ▼                               │
│                             ┌────────────────┐                      │
│                             │HypothesisGen   │                      │
│                             │ - Template     │                      │
│                             │ - LLM          │                      │
│                             │ - Hybrid       │                      │
│                             └───────┬────────┘                      │
│                                     │                               │
│                                     ▼                               │
│                             ┌────────────────┐                      │
│                             │ Optimizer      │                      │
│                             │ (hill climbing)│                      │
│                             └───────┬────────┘                      │
│                                     │                               │
│                                     ▼                               │
│                             ┌────────────────┐                      │
│                             │Improved Workflow│                     │
│                             └────────────────┘                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## External Dependencies

| Dependency | Purpose |
|------------|---------|
| **PocketFlow** | Workflow execution engine (DAG-based node execution) |
| **LiteLLM** | Multi-provider LLM interface (100+ providers) |
| **Pydantic** | Data validation and serialization |
| **PyYAML** | YAML serialization for persistence |
| **Jinja2** | Prompt template rendering |
| **Typer** | CLI framework |
| **Rich** | Styled terminal output |
| **python-dotenv** | Environment configuration |
| **Arize Phoenix** | Optional LLM observability and tracing |
