# Node Quick Reference

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
| **CriterionEvaluationNode** | Evaluate single criterion | `default` |
| **SubCriterionNode** | Evaluate sub-criteria | `default` |
| **ResultAggregatorNode** | Aggregate with AND/OR logic | `default` |
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
- `criterion_results` - List of criterion evaluations
- `evaluation_result` - Final aggregated result

## Creating a Custom Node

```python
from pocketflow import Node
from policyflow.nodes import NodeSchema, NodeParameter, register_node

class MyNode(Node):
    parser_schema = NodeSchema(
        name="MyNode",
        description="What this node does",
        category="deterministic",  # or "llm", "internal"
        parameters=[
            NodeParameter("param1", "str", "Description", required=True),
            NodeParameter("param2", "int", "Optional param", required=False, default=10),
        ],
        actions=["action1", "action2"],
        parser_exposed=True,
    )

    def __init__(self, param1: str, param2: int = 10):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def prep(self, shared: dict) -> dict:
        return {"text": shared.get("input_text", "")}

    def exec(self, prep_res: dict) -> dict:
        # Your logic here
        return {"result": prep_res["text"].upper()}

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        shared["my_result"] = exec_res["result"]
        return "action1"

# Register the node
register_node(MyNode)
```

## Creating an LLM Node

Each node defines its own `default_model` and result model directly in its file:

```python
from pydantic import BaseModel, Field
from policyflow.nodes import LLMNode, NodeSchema, NodeParameter

# Define the result model in the same file as the node
class MyAnalysisResult(BaseModel):
    """Result from MyLLMNode."""
    summary: str = Field(description="Analysis summary")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")

class MyLLMNode(LLMNode):
    # Each node defines its own default model
    default_model: str = "anthropic/claude-sonnet-4-20250514"

    parser_schema = NodeSchema(
        name="MyLLMNode",
        description="LLM-powered analysis",
        category="llm",
        parameters=[...],
        actions=["success", "failure"],
        parser_exposed=True,
    )

    def __init__(self, config, model: str | None = None, prompt: str = "", cache_ttl: int = 3600):
        super().__init__(config=config, model=model, cache_ttl=cache_ttl)
        self.prompt = prompt

    def prep(self, shared: dict) -> dict:
        return {"text": shared.get("input_text", "")}

    def exec(self, prep_res: dict) -> dict:
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
from policyflow.clause_mapping import (
    ClauseResult,
    extract_clause_results,
    format_clause_results_report,
    summarize_results,
)

# After workflow execution
results = extract_clause_results(shared, normalized_policy)
print(format_clause_results_report(results))
# [+] Clause 1.1: PASS (92%)
#   [+] Clause 1.1.a: PASS (95%)
#   [-] Clause 1.1.b: FAIL (88%)
```
