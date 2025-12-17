# Node Quick Reference

## Creating Nodes (The Easy Way)

PolicyFlow provides tools to dramatically reduce boilerplate when creating nodes:

### The @node_schema Decorator

Instead of manually writing `NodeSchema` definitions, use the `@node_schema` decorator which auto-generates schemas from your `__init__` type hints:

```python
from policyflow.nodes.decorators import node_schema

@node_schema(
    description="Classify text into categories",
    category="llm",
    actions=["<category_name>"],
    parser_exposed=True
)
class MyNode(Node):
    def __init__(self, categories: list[str], threshold: float = 0.7):
        # Parameters automatically detected:
        # - categories: required (no default)
        # - threshold: optional (has default)
        super().__init__()
        self.categories = categories
        self.threshold = threshold
```

The decorator eliminates ~20-30 lines per node by automatically extracting:
- Parameter names and types from type hints
- Required vs optional based on default values
- Parameter descriptions from docstrings (if present)

### The DeterministicNode Base Class

For simple deterministic nodes, extend `DeterministicNode` to get standard prep/post methods:

```python
from policyflow.nodes.base import DeterministicNode
from policyflow.nodes.decorators import node_schema

@node_schema(
    description="Score text based on keyword matches",
    category="deterministic",
    actions=["high", "low"],
    parser_exposed=True
)
class MyKeywordNode(DeterministicNode):
    def __init__(self, keywords: list[str], threshold: float = 0.5):
        super().__init__()
        self.keywords = keywords
        self.threshold = threshold
        self.output_key = "keyword_result"  # Optional: store in shared

    def exec(self, prep_res: dict) -> dict:
        # Only implement the core logic
        text = prep_res.get("input_text", "")
        score = sum(1 for kw in self.keywords if kw in text) / len(self.keywords)
        return {"score": score}

    def get_action(self, exec_res: dict) -> str:
        return "high" if exec_res["score"] >= self.threshold else "low"
```

`DeterministicNode` provides:
- Standard `prep()` extracting `input_text` from shared store
- Standard `post()` storing results and calling `get_action()`
- You only implement `exec()` (core logic) and `get_action()` (routing)

### Benefits

- **85% less boilerplate**: Typical node reduced from ~80 lines to ~15 lines
- **Consistency**: All nodes follow the same patterns
- **Type safety**: Decorator validates type hints match actual parameters
- **Easier maintenance**: Changes to base classes propagate automatically

## Node Lifecycle

```
shared dict → prep() → exec() → post() → action string → next node
```

## Available Nodes

### Deterministic Nodes (No LLM)

| Node | Description | Parameters | Actions |
|------|-------------|------------|---------|
| **PatternMatchNode** | Match regex/keyword patterns | `patterns: list[str]`, `mode: any\|all\|none` | `matched`, `not_matched` |
| **LengthGateNode** | Route by text length | `min_length: int`, `max_length: int` | `within_range`, `too_short`, `too_long` |
| **KeywordScorerNode** | Score weighted keywords | `keywords: dict[str, float]`, `threshold: float` | `above_threshold`, `below_threshold` |
| **TransformNode** | Preprocess text | `operations: list[str]` | `default` |

**TransformNode operations:** `lowercase`, `strip_html`, `truncate:N`, `normalize_whitespace`

### LLM-Based Nodes

All LLM nodes accept a `model` parameter to specify which LLM to use. Each node class defines a `default_model` that is used if not overridden.

| Node | Description | Parameters | Actions |
|------|-------------|------------|---------|
| **ClassifierNode** | Classify into categories | `categories: list[str]`, `model: str`, `descriptions: dict` | category names |
| **SentimentNode** | Analyze sentiment/tone | `model: str`, `granularity: basic\|detailed` | `positive`, `negative`, `neutral`, `mixed` |
| **DataExtractorNode** | Extract structured data | `schema: dict`, `model: str` | `default` |
| **SamplerNode** | Run N evaluations for consensus | `n_samples: int`, `aggregation: str`, `inner_prompt: str`, `model: str` | `consensus`, `majority`, `split` |

### Internal Nodes (Used by Parser)

| Node | Description | Actions |
|------|-------------|---------|
| **ConfidenceGateNode** | Route by confidence | `high_confidence`, `needs_review`, `low_confidence` |

## Shared Store Keys

### Input Keys
- `input_text` - Main text to evaluate
- `policy` - Parsed policy object
- `workflow_config` - Configuration settings

### Output Keys
- `pattern_match_result` - PatternMatchNode result
- `keyword_score` - KeywordScorerNode score
- `length_info` - LengthGateNode info
- `classification` - ClassifierNode result
- `sentiment` - SentimentNode result
- `extracted_data` - DataExtractorNode result
- `clause_X_X_result` - Result for clause with ID X.X
- `confidence_gate_result` - ConfidenceGateNode result
- `result` - Final EvaluationResult object

## Creating a Custom Node

Using the modern decorator approach:

```python
from policyflow.nodes.base import DeterministicNode
from policyflow.nodes.decorators import node_schema
from policyflow.nodes import register_node

@node_schema(
    description="What this node does",
    category="deterministic",  # or "llm", "internal"
    actions=["action1", "action2"],
    parser_exposed=True,
)
class MyNode(DeterministicNode):
    def __init__(self, param1: str, param2: int = 10):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.output_key = "my_result"  # Optional: store in shared

    def exec(self, prep_res: dict) -> dict:
        # Your logic here
        text = prep_res.get("input_text", "")
        return {"result": text.upper()}

    def get_action(self, exec_res: dict) -> str:
        return "action1"

# Register the node
register_node(MyNode)
```

The `@node_schema` decorator automatically generates `NodeSchema` from your type hints, eliminating manual parameter definitions.

## Creating an LLM Node

Using the modern decorator approach with LLMNode:

```python
from pydantic import BaseModel, Field
from policyflow.nodes import LLMNode
from policyflow.nodes.decorators import node_schema

# Define the result model in the same file as the node
class MyAnalysisResult(BaseModel):
    """Result from MyLLMNode."""
    summary: str = Field(description="Analysis summary")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")

@node_schema(
    description="LLM-powered analysis",
    category="llm",
    actions=["success", "failure"],
    parser_exposed=True,
)
class MyLLMNode(LLMNode):
    # Each node defines its own default model
    default_model: str = "anthropic/claude-sonnet-4-20250514"

    def __init__(self, config, model: str | None = None, prompt: str = "", cache_ttl: int = 3600):
        super().__init__(config=config, model=model, cache_ttl=cache_ttl)
        self.prompt = prompt

    def prep(self, shared: dict) -> dict:
        return {"text": shared.get("input_text", "")}

    def exec(self, prep_res: dict) -> dict:
        # call_llm() uses CacheManager and RateLimiter automatically
        return self.call_llm(
            prompt=f"{self.prompt}\n\nText: {prep_res['text']}",
            system_prompt="You are an analyst.",
            yaml_response=True,
            span_name="my_llm_node"
        )

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        shared["my_analysis"] = exec_res
        return "success"
```

**LLMNode Benefits**:
- Built-in CacheManager for thread-safe response caching
- Built-in RateLimiter to prevent API rate limiting
- Automatic model selection with fallback chain
- Phoenix tracing integration (optional)

## Node Routing

```python
# Linear flow
node1 >> node2 >> node3

# Conditional routing
node1 - "action1" >> node2
node1 - "action2" >> node3

# Example: LengthGateNode routing
length_gate - "within_range" >> classifier
length_gate - "too_short" >> reject_node
length_gate - "too_long" >> truncate_node
```

## Registry Functions

```python
from policyflow.nodes import (
    register_node,
    get_node_class,
    get_all_nodes,
    get_parser_schemas,
)

# Register a node
register_node(MyNode)

# Get node class by name
cls = get_node_class("PatternMatchNode")

# Get all registered nodes
all_nodes = get_all_nodes()

# Get schemas for parser-exposed nodes
schemas = get_parser_schemas()
```

## Clause-Numbered Nodes (Two-Step Parser)

When using the two-step parser, node IDs follow clause numbering:

### Naming Convention

| Clause Number | Node ID |
|---------------|---------|
| `1.1` | `clause_1_1` |
| `1.1.a` | `clause_1_1_a` |
| `2.1.1.b` | `clause_2_1_1_b` |

### Numbering Utilities

```python
from policyflow.numbering import (
    clause_number_to_node_id,    # "1.1.a" -> "clause_1_1_a"
    node_id_to_clause_number,    # "clause_1_1_a" -> "1.1.a"
    generate_clause_number,      # Generate next in sequence
    parse_clause_depth,          # "1.1.a" -> 2
    get_parent_clause_number,    # "1.1.a" -> "1.1"
    clause_sort_key,             # For sorting
    is_ancestor_of,              # Check hierarchy
)
```

### Workflow Hierarchy

The two-step parser generates workflows with a `hierarchy` field:

```yaml
workflow:
  nodes:
    - id: clause_1_1_a
      type: ClassifierNode
      # ...
  start_node: preprocess
  hierarchy:
    - clause_number: "1.1"
      clause_text: "Original clause text"
      nodes: ["clause_1_1_a", "clause_1_1_b"]
      logic: any
      sub_groups:
        - clause_number: "1.1.a"
          nodes: ["clause_1_1_a"]
```

### Result Traceability

```python
from policyflow import EvaluationResult, ClauseResult

# After workflow execution, results are available in shared["result"]
result: EvaluationResult = shared["result"]

# Access per-clause breakdown
for cr in result.clause_results:
    status = "PASS" if cr.met else "FAIL"
    print(f"Clause {cr.clause_id}: {status} ({cr.confidence:.0%})")
    print(f"  Reasoning: {cr.reasoning}")
```
