# Two-Step Policy Parser

> **Status**: Implemented

## Overview

A two-step policy parser that:
1. **Step 1**: Normalizes raw policy markdown into structured sections/clauses with hierarchical numbering
2. **Step 2**: Generates workflow from normalized document with node IDs matching clause numbers

```
Raw Policy Markdown
        ↓ (Step 1: Normalize)
NormalizedPolicy (YAML)  ← Persisted for review/audit
        ↓ (Step 2: Generate Workflow)
ParsedWorkflowPolicyV2 (YAML)
```

## Features
- **Numbering**: 1, 1.1, 1.1.a style (numeric sections/clauses, letters at depth 3+)
- **Persistence**: Normalized document saved as YAML between steps for review/audit
- **Node mapping**: LLM decides appropriate node type for each clause
- **Nesting depth**: Supports 4+ levels (arbitrary depth)
- **Explainability**: Node IDs match clause numbers for traceability

---

## Usage

### CLI Commands

```bash
# Step 1: Normalize policy document
policy-eval normalize --policy policy.md --output normalized.yaml

# Step 2: Generate workflow from normalized document
policy-eval generate-workflow --normalized normalized.yaml --output workflow.yaml

# Both steps combined
policy-eval parse-two-step --policy policy.md --output-dir ./output/
```

### Python API

```python
from policyflow.parser import (
    normalize_policy,
    generate_workflow_from_normalized,
    parse_policy_two_step,
)
from policyflow.models import NormalizedPolicy

# Step 1: Normalize
normalized = normalize_policy(policy_markdown)
normalized.save_yaml("normalized.yaml")

# Step 2: Generate workflow
workflow = generate_workflow_from_normalized(normalized)
workflow.save_yaml("workflow.yaml")

# Or both steps at once
workflow = parse_policy_two_step(
    policy_markdown,
    save_normalized="normalized.yaml"
)
```

---

## Implementation

### Files Created

| File | Purpose |
|------|---------|
| `src/policyflow/numbering.py` | Clause numbering utilities |
| `src/policyflow/clause_mapping.py` | Result traceability utilities |
| `src/policyflow/templates/policy_normalizer.j2` | Step 1 LLM prompt |
| `src/policyflow/templates/workflow_from_normalized.j2` | Step 2 LLM prompt |

### Files Modified

| File | Changes |
|------|---------|
| `src/policyflow/models.py` | Added `ClauseType`, `Clause`, `Section`, `NormalizedPolicy`, `NodeGroup`, `HierarchicalWorkflowDefinition`, `ParsedWorkflowPolicyV2` |
| `src/policyflow/parser.py` | Added `normalize_policy()`, `generate_workflow_from_normalized()`, `parse_policy_two_step()` |
| `src/policyflow/prompts/__init__.py` | Added `get_normalize_policy_prompt()`, `get_workflow_from_normalized_prompt()` |
| `src/policyflow/cli.py` | Added `normalize`, `generate-workflow`, `parse-two-step` commands |

---

## Data Models

### NormalizedPolicy (Step 1 Output)

```python
class Clause(BaseModel):
    number: str          # "1", "1.1", "1.1.a"
    title: str
    text: str            # Original clause text
    clause_type: ClauseType  # requirement|definition|condition|exception|reference
    sub_clauses: list["Clause"]
    logic: LogicOperator | None  # all|any for sub-clause combination

class Section(BaseModel):
    number: str
    title: str
    description: str
    clauses: list[Clause]
    logic: LogicOperator

class NormalizedPolicy(BaseModel):
    title: str
    version: str
    description: str
    sections: list[Section]
    logic: LogicOperator
    raw_text: str
    metadata: dict
```

### ParsedWorkflowPolicyV2 (Step 2 Output)

```python
class NodeGroup(BaseModel):
    clause_number: str
    clause_text: str
    nodes: list[str]       # Node IDs for this clause
    sub_groups: list["NodeGroup"]
    logic: LogicOperator | None

class HierarchicalWorkflowDefinition(BaseModel):
    nodes: list[NodeConfig]
    start_node: str
    hierarchy: list[NodeGroup]  # Preserves document structure

class ParsedWorkflowPolicyV2(BaseModel):
    title: str
    description: str
    workflow: HierarchicalWorkflowDefinition
    normalized_policy_ref: str | None
    raw_text: str
```

---

## Numbering Utilities

```python
from policyflow.numbering import (
    generate_clause_number,      # Generate next clause number
    clause_number_to_node_id,    # "1.1.a" -> "clause_1_1_a"
    node_id_to_clause_number,    # "clause_1_1_a" -> "1.1.a"
    parse_clause_depth,          # "1.1.a" -> 2
    get_parent_clause_number,    # "1.1.a" -> "1.1"
    clause_sort_key,             # For sorting clause numbers
    is_ancestor_of,              # Check hierarchy relationship
)
```

---

## Clause Mapping (Result Traceability)

```python
from policyflow.clause_mapping import (
    ClauseResult,
    extract_clause_results,      # Extract results from workflow shared store
    build_hierarchical_results,  # Nest results by clause hierarchy
    format_clause_results_report,  # Human-readable report
    summarize_results,           # Statistics (pass rate, etc.)
)

# After workflow execution
results = extract_clause_results(shared, normalized_policy)
report = format_clause_results_report(results)
print(report)
# [+] Clause 1.1: PASS (92% confidence)
#     Reasoning: Content addresses investor directly
#   [+] Clause 1.1.a: PASS (95% confidence)
#   [-] Clause 1.1.b: FAIL (88% confidence)
```

---

## Key Design Decisions

1. **Node ID Convention**: `clause_<number>` with dots replaced by underscores
   - Clause 1.1 → `clause_1_1`
   - Clause 1.1.a → `clause_1_1_a`

2. **Hierarchy Preservation**: `hierarchy` field in workflow mirrors document structure

3. **File Naming Convention**: `<name>_normalized.yaml` and `<name>_workflow.yaml`

4. **Backward Compatibility**: Existing `parse_policy()` and `parse_policy_to_workflow()` unchanged

5. **Clause Types**: Help inform node type selection
   - `requirement` → PatternMatchNode, ClassifierNode, SamplerNode
   - `definition` → Context only (no dedicated node)
   - `condition` → ClassifierNode with routing
   - `exception` → Short-circuit nodes
