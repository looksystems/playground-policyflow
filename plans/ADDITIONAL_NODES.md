# Additional Node Types for PolicyFlow

## Status: COMPLETE

All 8 new node types have been implemented with self-documenting schemas for the dynamic parser.

---

## Summary

| Node | Type | Parser Exposed | File |
|------|------|----------------|------|
| LLMNode | Base Class | No | `nodes/llm_node.py` |
| PatternMatchNode | Deterministic | Yes | `nodes/pattern_match.py` |
| LengthGateNode | Deterministic | Yes | `nodes/length_gate.py` |
| KeywordScorerNode | Deterministic | Yes | `nodes/keyword_scorer.py` |
| TransformNode | Deterministic | Yes | `nodes/transform.py` |
| ClassifierNode | LLM-based | Yes | `nodes/classifier.py` |
| DataExtractorNode | LLM-based | Yes | `nodes/data_extractor.py` |
| SentimentNode | LLM-based | Yes | `nodes/sentiment.py` |
| SamplerNode | LLM-based | Yes | `nodes/sampler.py` |

---

## Architecture

- All nodes follow the 3-phase lifecycle: `prep()` → `exec()` → `post()`
- Nodes return action strings for routing ("default", "matched", "high", etc.)
- Shared store passes data between nodes
- Deterministic nodes inherit from `pocketflow.Node`
- LLM-based nodes inherit from `LLMNode` (which provides caching + throttling)
- **Each node declares a `parser_schema` class attribute for self-documentation**
- **Dynamic parser discovers nodes via registry and includes them in LLM prompts**

---

## Part 1: LLMNode Base Class

**File**: `src/policyflow/nodes/llm_node.py`

Base class for all LLM-based nodes with built-in caching and rate limiting.

```python
from policyflow.nodes import LLMNode

class MyCustomNode(LLMNode):
    def __init__(self, config: WorkflowConfig):
        super().__init__(config=config, cache_ttl=3600, rate_limit=60)

    def exec(self, prep_res: dict) -> dict:
        return self.call_llm(prompt="...", system_prompt="...")
```

**Features**:
- YAML file-based cache in `.cache/` directory
- Token bucket rate limiting (requests per minute)
- Thread-safe caching and rate limiting
- Automatic TTL expiration

---

## Part 2: Deterministic Nodes

### PatternMatchNode

**File**: `src/policyflow/nodes/pattern_match.py`

Check input against regex/keyword patterns.

```python
from policyflow.nodes import PatternMatchNode

node = PatternMatchNode(
    patterns=[r"\b(password|secret)\b", r"\d{3}-\d{2}-\d{4}"],
    mode="any"  # "any" | "all" | "none"
)
```

- **Actions**: `"matched"` | `"not_matched"`
- **Stores**: `shared["pattern_match_result"]` with matched patterns, match details

---

### KeywordScorerNode

**File**: `src/policyflow/nodes/keyword_scorer.py`

Score input based on weighted keyword presence.

```python
from policyflow.nodes import KeywordScorerNode

node = KeywordScorerNode(
    keywords={"urgent": 0.5, "critical": 0.8, "spam": -1.0},
    thresholds={"high": 0.7, "medium": 0.3}
)
```

- **Actions**: `"high"` | `"medium"` | `"low"`
- **Stores**: `shared["keyword_score"]` with score, level, matched keywords breakdown

---

### LengthGateNode

**File**: `src/policyflow/nodes/length_gate.py`

Route based on input length.

```python
from policyflow.nodes import LengthGateNode

node = LengthGateNode(
    thresholds={"short": 100, "medium": 1000, "long": 5000}
)
```

- **Actions**: Returns bucket name (`"short"` | `"medium"` | `"long"` | etc.)
- **Stores**: `shared["length_info"]` with char_count, word_count, bucket

---

### TransformNode

**File**: `src/policyflow/nodes/transform.py`

Transform/preprocess input text.

```python
from policyflow.nodes import TransformNode

node = TransformNode(
    transforms=["lowercase", "strip_html", "normalize_whitespace", "truncate:1000"],
    input_key="input_text",
    output_key="input_text"
)
```

**Supported transforms**:
- `lowercase` / `uppercase`
- `strip_html` - Remove HTML tags
- `normalize_whitespace` - Collapse whitespace
- `strip_urls` - Remove URLs
- `strip_emails` - Remove email addresses
- `truncate:N` - Truncate to N characters
- `trim` - Strip leading/trailing whitespace

- **Actions**: `"default"`
- **Stores**: Updates `shared[output_key]`

---

## Part 3: LLM-Based Nodes

### ClassifierNode

**File**: `src/policyflow/nodes/classifier.py`
**Template**: `src/policyflow/templates/classifier.j2`

Classify input into predefined categories.

```python
from policyflow.nodes import ClassifierNode
from policyflow.config import WorkflowConfig

node = ClassifierNode(
    categories=["spam", "legitimate", "unclear"],
    config=WorkflowConfig(),
    descriptions={
        "spam": "Unwanted or malicious content",
        "legitimate": "Valid and appropriate content",
        "unclear": "Requires human review"
    }
)
```

- **Actions**: Returns the classified category name
- **Stores**: `shared["classification"]` with category, confidence, reasoning

---

### DataExtractorNode

**File**: `src/policyflow/nodes/data_extractor.py`
**Template**: `src/policyflow/templates/extractor.j2`

Extract structured data from input.

```python
from policyflow.nodes import DataExtractorNode
from policyflow.config import WorkflowConfig

node = DataExtractorNode(
    schema={
        "entities": {"people": "list of person names", "orgs": "list of organizations"},
        "values": {"amounts": "monetary amounts", "dates": "date references"},
        "facts": ["main topic", "sentiment"]
    },
    config=WorkflowConfig()
)
```

- **Actions**: `"default"`
- **Stores**: `shared["extracted_data"]` with structured extraction results

---

### SentimentNode

**File**: `src/policyflow/nodes/sentiment.py`
**Template**: `src/policyflow/templates/sentiment.j2`

Classify sentiment/emotional tone.

```python
from policyflow.nodes import SentimentNode
from policyflow.config import WorkflowConfig

node = SentimentNode(
    config=WorkflowConfig(),
    granularity="detailed"  # "basic" | "detailed"
)
```

- **Actions**: `"positive"` | `"negative"` | `"neutral"` | `"mixed"`
- **Stores**: `shared["sentiment"]` with label, confidence, (optional: emotions, intensity)

---

### SamplerNode

**File**: `src/policyflow/nodes/sampler.py`

Run evaluation N times and aggregate for consensus.

```python
from policyflow.nodes import SamplerNode
from policyflow.config import WorkflowConfig

node = SamplerNode(
    n_samples=5,
    aggregation="majority",  # "majority" | "unanimous" | "any"
    inner_prompt="Is this content appropriate? Answer with result: true/false",
    config=WorkflowConfig()
)
```

- **Actions**: `"consensus"` | `"majority"` | `"split"`
- **Stores**: `shared["sample_results"]` with individual_results, aggregated_result, agreement_ratio

---

## Configuration

Cache and throttle settings are configured in `WorkflowConfig`:

```python
from policyflow.config import WorkflowConfig

config = WorkflowConfig()
# config.cache.enabled = True
# config.cache.ttl = 3600  # seconds
# config.cache.directory = ".cache"
# config.throttle.enabled = False
# config.throttle.requests_per_minute = 60
```

Environment variables:
- `POLICY_EVAL_CACHE_ENABLED` (default: "true")
- `POLICY_EVAL_CACHE_TTL` (default: "3600")
- `POLICY_EVAL_CACHE_DIR` (default: ".cache")
- `POLICY_EVAL_THROTTLE_ENABLED` (default: "false")
- `POLICY_EVAL_THROTTLE_RPM` (default: "60")

---

## Result Types

All result types are defined in `src/policyflow/models.py`:

- `PatternMatchResult` - For PatternMatchNode
- `KeywordScoreResult` - For KeywordScorerNode
- `LengthInfo` - For LengthGateNode
- `ClassificationResult` - For ClassifierNode
- `SentimentResult` - For SentimentNode
- `SampleResults` - For SamplerNode

---

## Files Created

| File | Description |
|------|-------------|
| `src/policyflow/nodes/llm_node.py` | Base class with caching + throttling |
| `src/policyflow/nodes/pattern_match.py` | Regex/keyword matching |
| `src/policyflow/nodes/length_gate.py` | Length-based routing |
| `src/policyflow/nodes/keyword_scorer.py` | Weighted keyword scoring |
| `src/policyflow/nodes/transform.py` | Text preprocessing |
| `src/policyflow/nodes/classifier.py` | LLM classification |
| `src/policyflow/nodes/data_extractor.py` | LLM data extraction |
| `src/policyflow/nodes/sentiment.py` | LLM sentiment analysis |
| `src/policyflow/nodes/sampler.py` | Multi-sample consensus |
| `src/policyflow/templates/classifier.j2` | Classifier prompt template |
| `src/policyflow/templates/extractor.j2` | Extractor prompt template |
| `src/policyflow/templates/sentiment.j2` | Sentiment prompt template |

## Files Modified

| File | Changes |
|------|---------|
| `src/policyflow/nodes/__init__.py` | Export all new nodes, registry integration |
| `src/policyflow/models.py` | Add result types, workflow models |
| `src/policyflow/config.py` | Add CacheConfig, ThrottleConfig |

---

## Dynamic Parser Integration

All nodes are now self-documenting through the `parser_schema` class attribute. The parser dynamically discovers exposed nodes and includes their documentation in the LLM prompt.

### New Files for Dynamic Parser

| File | Description |
|------|-------------|
| `src/policyflow/nodes/schema.py` | `NodeSchema` and `NodeParameter` dataclasses |
| `src/policyflow/nodes/registry.py` | Node registration and discovery |
| `src/policyflow/workflow_builder.py` | Build workflows from parsed configs |

### Node Schema Structure

Each node declares a `parser_schema` class attribute:

```python
from policyflow.nodes import NodeSchema, NodeParameter

class MyNode(Node):
    parser_schema = NodeSchema(
        name="MyNode",
        description="Brief description for LLM",
        category="deterministic",  # or "llm", "internal"
        parameters=[
            NodeParameter("param1", "str", "Description", required=True),
            NodeParameter("param2", "int", "Description", required=False, default=10),
        ],
        actions=["action1", "action2"],
        yaml_example="...",
        parser_exposed=True,  # False to hide from parser
    )
```

### Registry Functions

```python
from policyflow.nodes import (
    get_parser_schemas,  # Get schemas for parser-exposed nodes
    get_node_class,      # Lookup node class by name
    get_all_nodes,       # Get all registered nodes
    register_node,       # Register a custom node
)

# Get all parser-exposed node schemas
schemas = get_parser_schemas()  # Returns list of NodeSchema
```

### Using the Dynamic Parser

```python
from policyflow.parser import parse_policy_to_workflow
from policyflow.workflow_builder import DynamicWorkflowBuilder

# Parse policy into workflow definition
policy = """
# Content Moderation Policy
Check all user content for:
1. Profanity patterns
2. Spam indicators
3. Negative sentiment
"""
parsed = parse_policy_to_workflow(policy)

# Build and run the workflow
builder = DynamicWorkflowBuilder(parsed)
result = builder.run("Text to evaluate")

# Or build the Flow separately
flow = builder.build()
shared = {"input_text": "Text to evaluate"}
flow.run(shared)
```

### Workflow Output Schema

The parser generates YAML with this structure:

```yaml
title: Policy Title
description: What the policy does
workflow:
  nodes:
    - id: check_patterns
      type: PatternMatchNode
      params:
        patterns: ["\\bspam\\b", "\\bfree money\\b"]
        mode: any
      routes:
        matched: flag_spam
        not_matched: check_sentiment
    - id: check_sentiment
      type: SentimentNode
      params:
        granularity: basic
      routes:
        negative: human_review
        positive: approve
        neutral: approve
        mixed: human_review
  start_node: check_patterns
```

### Adding Custom Nodes

1. Create your node class with `parser_schema`:

```python
from pocketflow import Node
from policyflow.nodes import NodeSchema, NodeParameter, register_node

@register_node
class MyCustomNode(Node):
    parser_schema = NodeSchema(
        name="MyCustomNode",
        description="Does something custom",
        category="deterministic",
        parameters=[...],
        actions=["success", "failure"],
        yaml_example="...",
        parser_exposed=True,
    )

    def prep(self, shared): ...
    def exec(self, prep_res): ...
    def post(self, shared, prep_res, exec_res): ...
```

2. The node will automatically appear in parser prompts and be available for workflow building.
