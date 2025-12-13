# PolicyFlow Developer Documentation

PolicyFlow is an LLM-powered compliance evaluation framework that parses policy documents and evaluates text against extracted criteria, providing granular pass/fail results with confidence scores and detailed reasoning.

## Documentation

| Document | Description |
|----------|-------------|
| [Concepts & Workflow](concepts.md) | Core concepts, terminology, and evaluation workflow |
| [Architecture](architecture.md) | System architecture, data flow, and class relationships |
| [CLI Cheatsheet](cli-cheatsheet.md) | Quick reference for all CLI commands |
| [Source Code Guide](source-guide.md) | Guide to source code structure and extending the system |
| [User Guide](USERGUIDE.md) | Complete user guide with examples and best practices |
| [Node Reference](NODES.md) | Quick reference for all node types |

## Quick Example

```python
from policyflow import evaluate

result = evaluate(
    input_text="This investment may lose value. Past performance does not guarantee future results.",
    policy_path="policy.md"
)

print(f"Satisfied: {result.policy_satisfied}")
print(f"Confidence: {result.overall_confidence:.0%}")
for clause in result.clause_results:
    print(f"  {clause.clause_id}: {'PASS' if clause.met else 'FAIL'}")
```

## Key Features

- **Two-step parsing**: Normalize policies to YAML, then generate executable workflows
- **Explainable results**: Node IDs match clause numbers for full traceability
- **Model-agnostic**: Supports 100+ LLM providers via LiteLLM
- **Extensible nodes**: Mix LLM-based and deterministic evaluation nodes
- **Benchmark system**: Test and optimize workflow accuracy
