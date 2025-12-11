# Fix: Infinite Loop in policy-eval eval Command

**Status: IMPLEMENTED** (2025-12-11)

## Root Cause
The LLM-generated workflow creates a cyclic graph with no terminal state. PocketFlow's `_orch` method (`while curr:`) loops forever because every node routes to another node.

**Key issue**: The `policy_parser.j2` template doesn't explain how to create terminal nodes (nodes that end the workflow), so the LLM doesn't know to include end states.

## Implementation Steps

### Step 1: Update Template with Terminal Node Instructions
**File**: `src/policyflow/templates/policy_parser.j2`

Add to the `## Instructions` section:
```
8. Every workflow MUST have at least one terminal node (a node with `routes: {}` or omitted routes)
9. Terminal nodes end the workflow - use them for final aggregation or result storage
```

Add to the `## Tips` section:
```
- To end a workflow, use a node with empty routes: `routes: {}`
- The workflow terminates when a node returns an action with no matching route
```

Update the output format example to show a terminal node:
```yaml
    - id: final_result
      type: AggregateNode
      params: {}
      routes: {}  # Terminal node - ends workflow
```

### Step 2: Add Workflow Validation
**File**: `src/policyflow/workflow_builder.py`

Add a `_validate_workflow()` method called from `build()`:

```python
def _validate_workflow(self, nodes: dict[str, Node]) -> None:
    """Validate workflow has no cycles and has terminal nodes."""
    # Check for terminal nodes (nodes with empty routes)
    has_terminal = any(
        not node_config.routes
        for node_config in self.policy.workflow.nodes
    )
    if not has_terminal:
        import warnings
        warnings.warn(
            "Workflow has no terminal nodes (nodes with empty routes). "
            "This may cause infinite execution."
        )

    # Check for cycles using DFS
    visited = set()
    rec_stack = set()

    def has_cycle(node_id: str) -> bool:
        visited.add(node_id)
        rec_stack.add(node_id)
        node_config = next(
            (n for n in self.policy.workflow.nodes if n.id == node_id),
            None
        )
        if node_config:
            for target_id in node_config.routes.values():
                if target_id not in visited:
                    if has_cycle(target_id):
                        return True
                elif target_id in rec_stack:
                    return True
        rec_stack.remove(node_id)
        return False

    start_id = self.policy.workflow.start_node
    if has_cycle(start_id):
        import warnings
        warnings.warn(
            "Workflow contains cycles. This may cause infinite execution."
        )
```

Call `_validate_workflow(nodes)` in `build()` after Phase 2 (wiring routes).

### Step 3: Add Max Iterations Safety
**File**: `src/policyflow/workflow_builder.py`

Modify `run()` method to add iteration limiting:

```python
def run(self, input_text: str, max_iterations: int = 100) -> dict:
    """Build workflow and run it with the given input.

    Args:
        input_text: Text to process through the workflow
        max_iterations: Maximum node executions before raising error

    Returns:
        The shared store after workflow execution

    Raises:
        RuntimeError: If max_iterations is exceeded
    """
    flow = self.build()
    shared = {"input_text": input_text, "_iteration_count": 0}

    # Wrap node execution to track iterations
    original_run = flow.start._run.__func__
    def counted_run(node_self, shared):
        shared["_iteration_count"] = shared.get("_iteration_count", 0) + 1
        if shared["_iteration_count"] > max_iterations:
            raise RuntimeError(
                f"Workflow exceeded {max_iterations} iterations. "
                "Possible infinite loop detected."
            )
        return original_run(node_self, shared)

    # Patch all nodes in the workflow
    for node_config in self.policy.workflow.nodes:
        # ... apply patch to each node

    flow.run(shared)
    del shared["_iteration_count"]  # Clean up internal counter
    return shared
```

## Files Modified
1. `src/policyflow/templates/policy_parser.j2` - Added terminal node documentation
2. `src/policyflow/workflow_builder.py` - Added validation and iteration limit
3. `tests/test_workflow_builder.py` - Added tests for validation and iteration limiting

## Implementation Notes

All three steps were implemented as specified:

1. **Template updated** with instructions 8-9 about terminal nodes, tips about `routes: {}`, and an example terminal node in the output format.

2. **Workflow validation** added via `_validate_workflow()` method that:
   - Warns if no terminal nodes exist
   - Detects cycles using DFS and warns if found

3. **Max iterations safety** added via:
   - `max_iterations` parameter on `run()` (default: 100)
   - `_collect_all_nodes()` helper to traverse the flow graph
   - `RuntimeError` raised if iterations exceed the limit

Tests added: 6 new tests covering validation warnings and iteration limiting.
