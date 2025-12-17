# Source Code Guide

This document provides a guide to the PolicyFlow source code, key classes, and how to extend the system.

## Directory Structure

```
src/policyflow/
├── __init__.py              # Public API exports
├── models.py                # Core data models (Pydantic)
├── parser.py                # Two-step policy parsing
├── workflow_builder.py      # Workflow construction & execution
├── config.py                # Configuration (env vars)
├── llm.py                   # LLM wrapper (LiteLLM)
├── cli.py                   # Main CLI commands
├── clause_mapping.py        # Clause/policy mapping
├── numbering.py             # Clause numbering utilities
│
├── nodes/                   # Node implementations
│   ├── __init__.py          # Registration & exports
│   ├── schema.py            # NodeSchema, NodeParameter
│   ├── registry.py          # Node registry
│   ├── llm_node.py          # LLMNode base class
│   ├── pattern_match.py     # PatternMatchNode
│   ├── classifier.py        # ClassifierNode
│   ├── sentiment.py         # SentimentNode
│   ├── keyword_scorer.py    # KeywordScorerNode
│   ├── confidence_gate.py   # ConfidenceGateNode
│   ├── length_gate.py       # LengthGateNode
│   ├── data_extractor.py    # DataExtractorNode
│   ├── sampler.py           # SamplerNode
│   └── transform.py         # TransformNode
│
├── benchmark/               # Benchmarking system
│   ├── __init__.py
│   ├── models.py            # GoldenDataset, BenchmarkReport
│   ├── generator.py         # Test case generation
│   ├── runner.py            # Benchmark execution
│   ├── analyzer.py          # Failure analysis
│   ├── optimizer.py         # Workflow optimization
│   ├── tracker.py           # Experiment tracking
│   └── cli.py               # Benchmark CLI
│
├── prompts/
│   └── __init__.py          # Prompt template functions
│
└── templates/               # Jinja2 templates
    ├── classifier.j2
    ├── sentiment.j2
    └── data_extractor.j2
```

## Key Files

| File | Purpose |
|------|---------|
| `models.py` | All Pydantic data models |
| `parser.py` | `normalize_policy()`, `generate_workflow_from_normalized()`, `parse_policy()` |
| `workflow_builder.py` | `DynamicWorkflowBuilder` class |
| `config.py` | `WorkflowConfig` and environment loading |
| `cli.py` | CLI entry point and core commands |
| `nodes/llm_node.py` | `LLMNode` base class with caching |
| `nodes/registry.py` | Node registration system |

## Core Classes

### models.py - Data Models

**Policy Structure:**

```python
class Clause(BaseModel):
    number: str                    # "1.1.a"
    title: str | None
    text: str                      # Requirement text
    clause_type: ClauseType        # REQUIREMENT, DEFINITION, etc.
    logic: LogicOperator           # ALL or ANY
    sub_clauses: list[Clause]

class Section(BaseModel):
    number: str                    # "1", "2"
    title: str
    clauses: list[Clause]
    logic: LogicOperator

class NormalizedPolicy(YAMLMixin, BaseModel):
    title: str
    version: str | None
    description: str | None
    sections: list[Section]
    raw_text: str
    metadata: dict
```

**Workflow Structure:**

```python
class NodeConfig(BaseModel):
    id: str                        # "clause_1_1_a"
    type: str                      # "ClassifierNode"
    params: dict                   # Constructor args
    routes: dict[str, str]         # action -> next_node_id

class NodeGroup(BaseModel):
    clause_number: str
    nodes: list[str]               # Node IDs
    sub_groups: list[NodeGroup]
    logic: LogicOperator

class HierarchicalWorkflowDefinition(BaseModel):
    nodes: list[NodeConfig]
    start_node: str
    hierarchy: list[NodeGroup]

class ParsedWorkflowPolicy(YAMLMixin, BaseModel):
    title: str
    description: str | None
    workflow: HierarchicalWorkflowDefinition
    normalized_policy_ref: str | None
    raw_text: str | None
```

**Results:**

```python
class ClauseResult(BaseModel):
    clause_id: str
    clause_name: str | None
    met: bool
    reasoning: str
    confidence: float
    sub_results: list[ClauseResult]

class EvaluationResult(YAMLMixin, BaseModel):
    policy_satisfied: bool
    input_text: str
    clause_results: list[ClauseResult]
    overall_reasoning: str | None
    overall_confidence: float
    confidence_level: ConfidenceLevel
    needs_review: bool
    low_confidence_clauses: list[str]
```

### parser.py - Policy Parsing

```python
def normalize_policy(
    policy_markdown: str,
    config: WorkflowConfig | None = None
) -> NormalizedPolicy:
    """Step 1: Parse markdown to structured NormalizedPolicy."""

def generate_workflow_from_normalized(
    normalized: NormalizedPolicy,
    config: WorkflowConfig | None = None
) -> ParsedWorkflowPolicy:
    """Step 2: Generate executable workflow from normalized policy."""

def parse_policy(
    policy_markdown: str,
    config: WorkflowConfig | None = None,
    save_normalized: str | None = None
) -> ParsedWorkflowPolicy:
    """Combined: Run both steps, optionally save intermediate."""
```

### workflow_builder.py - Workflow Execution

```python
class DynamicWorkflowBuilder:
    def __init__(
        self,
        policy: ParsedWorkflowPolicy,
        config: WorkflowConfig | None = None
    ):
        self.policy = policy
        self.config = config or WorkflowConfig()
        self.nodes: dict[str, Node] = {}

    def build(self) -> Flow:
        """Build executable PocketFlow Flow from policy."""
        # Phase 1: Instantiate nodes
        # Phase 2: Wire routes
        # Validate workflow

    def run(
        self,
        input_text: str,
        max_iterations: int = 100
    ) -> dict:
        """Build and execute workflow, return shared store."""
```

### config.py - Configuration

```python
class WorkflowConfig(BaseModel):
    model: str = "anthropic/claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_retries: int = 3
    retry_wait: int = 2
    confidence_gate: ConfidenceGateConfig
    cache: CacheConfig
    throttle: ThrottleConfig
    phoenix: PhoenixConfig

def get_config() -> WorkflowConfig:
    """Load config from environment variables."""
```

## Node System

### Node Schema (nodes/schema.py)

```python
@dataclass
class NodeParameter:
    name: str
    type: str                      # "str", "int", "list[str]", etc.
    description: str
    required: bool = True
    default: Any = None

@dataclass
class NodeSchema:
    name: str
    description: str
    category: str                  # "deterministic", "llm", "internal"
    parameters: list[NodeParameter]
    actions: list[str]             # Possible return values
    yaml_example: str = ""
    parser_exposed: bool = True    # Include in parser prompts
```

### LLMNode Base Class (nodes/llm_node.py)

```python
class LLMNode(Node):
    default_model: str = "anthropic/claude-sonnet-4-20250514"

    def __init__(
        self,
        config: WorkflowConfig,
        model: str | None = None,
        cache_ttl: int = 3600
    ):
        self.config = config
        self.model = model or self.default_model
        self.cache_ttl = cache_ttl

    def call_llm(
        self,
        prompt: str,
        system_prompt: str = "",
        yaml_response: bool = True,
        span_name: str = "llm_call"
    ) -> dict:
        """Call LLM with caching and rate limiting."""
```

### Node Registry (nodes/registry.py)

```python
_NODE_REGISTRY: dict[str, type[Node]] = {}

def register_node(cls: type[Node]) -> type[Node]:
    """Register a node class (decorator)."""
    _NODE_REGISTRY[cls.parser_schema.name] = cls
    return cls

def get_node_class(name: str) -> type[Node]:
    """Get node class by name."""

def get_all_nodes() -> dict[str, type[Node]]:
    """Get all registered nodes."""

def get_parser_schemas() -> list[NodeSchema]:
    """Get schemas for parser-exposed nodes."""
```

## Node Catalog

| Node | File | Category | Actions |
|------|------|----------|---------|
| `PatternMatchNode` | `pattern_match.py` | deterministic | `matched`, `not_matched` |
| `KeywordScorerNode` | `keyword_scorer.py` | deterministic | `high`, `medium`, `low` |
| `TransformNode` | `transform.py` | deterministic | `default` |
| `LengthGateNode` | `length_gate.py` | deterministic | `within_range`, `too_short`, `too_long` |
| `ClassifierNode` | `classifier.py` | llm | category names |
| `SentimentNode` | `sentiment.py` | llm | `positive`, `negative`, `neutral`, `mixed` |
| `DataExtractorNode` | `data_extractor.py` | llm | `default` |
| `SamplerNode` | `sampler.py` | llm | `consensus`, `majority`, `split` |
| `ConfidenceGateNode` | `confidence_gate.py` | internal | `high_confidence`, `needs_review`, `low_confidence` |

## Adding a Custom Node

PolicyFlow provides two approaches for creating nodes, depending on complexity:

### Simple Deterministic Nodes: Use DeterministicNode

For nodes with standard input/output patterns, extend `DeterministicNode`:

```python
# src/policyflow/nodes/my_node.py
from policyflow.nodes.base import DeterministicNode
from policyflow.nodes.decorators import node_schema

@node_schema(
    description="What this node does",
    category="deterministic",
    actions=["pass", "fail"],
    parser_exposed=True
)
class MyNode(DeterministicNode):
    def __init__(self, threshold: float, mode: str = "strict"):
        super().__init__()
        self.threshold = threshold
        self.mode = mode
        self.output_key = "my_node_result"  # Optional: store result in shared store

    def exec(self, prep_res: dict) -> dict:
        # Your logic here
        text = prep_res.get("input_text", "")
        score = len(text) / 100
        return {"score": score, "passed": score >= self.threshold}

    def get_action(self, exec_res: dict) -> str:
        return "pass" if exec_res["passed"] else "fail"
```

The `@node_schema` decorator automatically generates `parser_schema` from your `__init__` type hints.

### Complex Nodes: Manual Implementation

For nodes requiring custom prep/post logic:

```python
# src/policyflow/nodes/my_node.py
from pocketflow import Node
from policyflow.nodes.decorators import node_schema

@node_schema(
    description="What this node does",
    category="deterministic",
    actions=["pass", "fail"],
    parser_exposed=True
)
class MyNode(Node):
    def __init__(self, threshold: float, mode: str = "strict"):
        super().__init__()
        self.threshold = threshold
        self.mode = mode

    def prep(self, shared: dict) -> dict:
        # Custom preparation logic
        return {"text": shared.get("input_text", "")}

    def exec(self, prep_res: dict) -> dict:
        # Your logic here
        score = len(prep_res["text"]) / 100
        return {"score": score, "passed": score >= self.threshold}

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        # Custom post-processing logic
        shared["my_node_result"] = exec_res
        return "pass" if exec_res["passed"] else "fail"
```

### 2. Register the Node

```python
# src/policyflow/nodes/__init__.py
from .my_node import MyNode
from .registry import register_node

register_node(MyNode)
```

### 3. Creating an LLM Node

LLMNode provides built-in caching and rate limiting via CacheManager and RateLimiter:

```python
from pydantic import BaseModel, Field
from policyflow.nodes.llm_node import LLMNode
from policyflow.nodes.decorators import node_schema

class AnalysisResult(BaseModel):
    summary: str = Field(description="Analysis summary")
    score: float = Field(ge=0.0, le=1.0)

@node_schema(
    description="LLM-powered analysis",
    category="llm",
    actions=["high", "low"],
    parser_exposed=True
)
class MyLLMNode(LLMNode):
    default_model: str = "anthropic/claude-sonnet-4-20250514"

    def __init__(self, config, prompt: str, model: str | None = None):
        super().__init__(config=config, model=model)
        self.prompt = prompt

    def prep(self, shared: dict) -> dict:
        return {"text": shared.get("input_text", "")}

    def exec(self, prep_res: dict) -> dict:
        # call_llm() automatically uses CacheManager and RateLimiter
        return self.call_llm(
            prompt=f"{self.prompt}\n\nText: {prep_res['text']}",
            system_prompt="Analyze the text and return YAML.",
            yaml_response=True,
            span_name="my_llm_node"
        )

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        shared["my_analysis"] = exec_res
        return "high" if exec_res.get("score", 0) > 0.7 else "low"
```

**Benefits of LLMNode**:
- **CacheManager**: Thread-safe file-based caching with TTL support
- **RateLimiter**: Token bucket algorithm prevents API rate limiting
- **Automatic model selection**: Falls back through type-specific → global → hardcoded defaults
- **Phoenix tracing**: Optional observability integration

## Numbering Utilities

```python
from policyflow.numbering import (
    clause_number_to_node_id,    # "1.1.a" -> "clause_1_1_a"
    node_id_to_clause_number,    # "clause_1_1_a" -> "1.1.a"
    generate_clause_number,      # Generate next in sequence
    parse_clause_depth,          # "1.1.a" -> 2
    get_parent_clause_number,    # "1.1.a" -> "1.1"
    clause_sort_key,             # For sorting clause numbers
    is_ancestor_of,              # Check if one clause is ancestor of another
)
```

## Benchmark System

### Key Classes

```python
# benchmark/models.py
class TestCase(BaseModel):
    id: str
    input_text: str
    expected_result: bool
    expected_criteria: dict[str, bool]
    category: str | None
    tags: list[str]

class GoldenDataset(YAMLMixin, BaseModel):
    test_cases: list[TestCase]
    metadata: dict

class BenchmarkMetrics(BaseModel):
    overall_accuracy: float
    precision: float
    recall: float
    f1: float
    criterion_metrics: dict[str, CriterionMetrics]

class BenchmarkReport(YAMLMixin, BaseModel):
    workflow_id: str
    metrics: BenchmarkMetrics
    results: list[TestResult]
    failures: list[FailureCase]
```

### Factory Functions

```python
from policyflow.benchmark import (
    create_generator,           # Test case generator
    create_analyzer,            # Failure analyzer
    create_hypothesis_generator,# Hypothesis generator
    load_golden_dataset,        # Load dataset from YAML
    SimpleBenchmarkRunner,      # Run benchmarks
    HillClimbingOptimizer,      # Optimize workflows
)
```

## Public API

The main exports from `policyflow`:

```python
from policyflow import (
    # Parsing
    parse_policy,
    normalize_policy,
    generate_workflow_from_normalized,

    # Execution
    evaluate,
    DynamicWorkflowBuilder,

    # Configuration
    WorkflowConfig,
    get_config,

    # Models
    NormalizedPolicy,
    ParsedWorkflowPolicy,
    EvaluationResult,
    ClauseResult,
    Clause,
    Section,

    # Enums
    LogicOperator,
    ClauseType,
    ConfidenceLevel,
)
```
