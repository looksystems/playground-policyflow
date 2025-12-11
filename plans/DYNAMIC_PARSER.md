# Dynamic Node-Aware Policy Parser

## Summary

Adapt the policy parser to dynamically discover and document available nodes. Each node class will contain its own parser schema, and the parser will scan nodes to construct the LLM prompt automatically.

## Key Design Decisions

- **Schema declaration**: Class attribute `parser_schema` holding a `NodeSchema` dataclass
- **Exposure control**: `parser_exposed` boolean flag in schema (configurable per node)
- **Discovery**: Registry pattern with functions to collect parser-exposed schemas
- **Prompt generation**: Pass schemas to Jinja2 template for dynamic rendering

---

## Implementation Steps

### Phase 1: Schema Infrastructure (2 new files)

**1. Create `src/policyflow/nodes/schema.py`**
```python
@dataclass
class NodeParameter:
    name: str
    type: str  # e.g., "list[str]", "int"
    description: str
    required: bool = True
    default: Any = None

@dataclass
class NodeSchema:
    name: str              # Class name
    description: str       # Brief LLM description
    category: str          # "deterministic" | "llm"
    parameters: list[NodeParameter]
    actions: list[str]     # Possible return actions
    yaml_example: str      # Minimal YAML config
    parser_exposed: bool = True
```

**2. Create `src/policyflow/nodes/registry.py`**
- `_node_registry: dict[str, Type[Node]]` - global registry
- `register_node(cls)` - decorator/function to register
- `get_node_class(name)` - lookup by name
- `get_parser_schemas()` - return schemas where `parser_exposed=True`

---

### Phase 2: Add Schemas to Nodes (12 files)

Add `parser_schema` class attribute to each node:

**Parser-exposed nodes (`parser_exposed=True`):**
- `src/policyflow/nodes/pattern_match.py`
- `src/policyflow/nodes/length_gate.py`
- `src/policyflow/nodes/keyword_scorer.py`
- `src/policyflow/nodes/transform.py`
- `src/policyflow/nodes/classifier.py`
- `src/policyflow/nodes/data_extractor.py`
- `src/policyflow/nodes/sentiment.py`
- `src/policyflow/nodes/sampler.py`

**Internal nodes (`parser_exposed=False`):**
- `src/policyflow/nodes/criterion.py`
- `src/policyflow/nodes/subcriterion.py`
- `src/policyflow/nodes/aggregate.py`
- `src/policyflow/nodes/confidence_gate.py`

---

### Phase 3: Node Registration

**Modify `src/policyflow/nodes/__init__.py`**
- Import registry functions
- Register all node classes after import
- Export `get_node_class`, `get_parser_schemas`

---

### Phase 4: Update Models

**Modify `src/policyflow/models.py`** - add:
```python
class NodeConfig(BaseModel):
    id: str
    type: str
    params: dict[str, Any] = {}
    routes: dict[str, str] = {}  # action -> next_node_id

class WorkflowDefinition(BaseModel):
    nodes: list[NodeConfig]
    start_node: str

class ParsedWorkflowPolicy(BaseModel):
    title: str
    description: str
    workflow: WorkflowDefinition
    raw_text: str
```

---

### Phase 5: Dynamic Prompt Generation

**Modify `src/policyflow/templates/policy_parser.j2`**
- Add section listing available nodes from `available_nodes` variable
- For each node: name, description, parameters, actions, YAML example

**Modify `src/policyflow/prompts/__init__.py`**
- `get_policy_parser_prompt()` calls `get_parser_schemas()` and passes to template

---

### Phase 6: Workflow Builder (1 new file)

**Create `src/policyflow/workflow_builder.py`**
```python
class DynamicWorkflowBuilder:
    def __init__(self, policy: ParsedWorkflowPolicy, config: WorkflowConfig)
    def build(self) -> Flow:
        # Instantiate nodes from config
        # Wire routes using PocketFlow operators
        # Return Flow(start=start_node)
```

---

### Phase 7: Update Parser

**Modify `src/policyflow/parser.py`**
- Add `parse_policy_to_workflow()` function returning `ParsedWorkflowPolicy`
- Keep existing `parse_policy()` for backward compatibility

---

## Files Summary

| File | Action |
|------|--------|
| `src/policyflow/nodes/schema.py` | Create |
| `src/policyflow/nodes/registry.py` | Create |
| `src/policyflow/workflow_builder.py` | Create |
| `src/policyflow/nodes/__init__.py` | Modify |
| `src/policyflow/nodes/pattern_match.py` | Add schema |
| `src/policyflow/nodes/length_gate.py` | Add schema |
| `src/policyflow/nodes/keyword_scorer.py` | Add schema |
| `src/policyflow/nodes/transform.py` | Add schema |
| `src/policyflow/nodes/classifier.py` | Add schema |
| `src/policyflow/nodes/data_extractor.py` | Add schema |
| `src/policyflow/nodes/sentiment.py` | Add schema |
| `src/policyflow/nodes/sampler.py` | Add schema |
| `src/policyflow/nodes/criterion.py` | Add schema (exposed=False) |
| `src/policyflow/nodes/subcriterion.py` | Add schema (exposed=False) |
| `src/policyflow/nodes/aggregate.py` | Add schema (exposed=False) |
| `src/policyflow/nodes/confidence_gate.py` | Add schema (exposed=False) |
| `src/policyflow/models.py` | Add new models |
| `src/policyflow/templates/policy_parser.j2` | Add node docs |
| `src/policyflow/prompts/__init__.py` | Pass schemas to template |
| `src/policyflow/parser.py` | Add new parser function |
