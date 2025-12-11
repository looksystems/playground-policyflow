# Refactor: Node-Defined LLM Models

## Goal
Each node defines its own LLM model. Class provides default, instance can override. No global fallback.

## Current State
- `WorkflowConfig.model` is global (from `POLICY_EVAL_MODEL` env var)
- `LLMNode.call_llm()` passes `config=self.config` to `llm.call_llm()`
- `llm.call_llm()` extracts `config.model` for the LiteLLM call

## Changes

### 1. `config.py`
- Add `DEFAULT_LLM_MODEL = "anthropic/claude-sonnet-4-20250514"` constant
- Remove `model` field from `WorkflowConfig`

### 2. `llm.py`
- Add required `model: str` parameter to `call_llm()`
- Use explicit `model` param instead of `config.model`

### 3. `nodes/llm_node.py`
- Add class attribute: `default_model: str = DEFAULT_LLM_MODEL`
- Add `model: str | None = None` parameter to `__init__`
- Set `self.model = model if model is not None else self.default_model`
- Update `call_llm()` to pass `model=self.model` to `_call_llm()`

### 4. LLMNode Subclasses
Each subclass adds `model` parameter and can define own `default_model`:

**`nodes/classifier.py`** - `ClassifierNode`
**`nodes/sentiment.py`** - `SentimentNode`
**`nodes/data_extractor.py`** - `DataExtractorNode`
**`nodes/sampler.py`** - `SamplerNode`

Pattern:
```python
class SomeNode(LLMNode):
    default_model: str = DEFAULT_LLM_MODEL  # or node-specific default

    def __init__(self, ..., model: str | None = None, ...):
        super().__init__(config=config, model=model, ...)
```

### 5. Non-LLMNode Classes That Use LLM
These extend `Node` directly but call `llm.call_llm()`:

**`nodes/criterion.py`** - `CriterionEvaluationNode`
**`nodes/subcriterion.py`** - `SubCriterionNode`

Pattern:
```python
class SomeNode(Node):
    default_model: str = DEFAULT_LLM_MODEL

    def __init__(self, ..., model: str | None = None, ...):
        self.model = model if model is not None else self.default_model

    def exec(self, prep_res):
        return call_llm(..., model=self.model, ...)
```

## Files to Modify (in order)

1. `src/policyflow/config.py` - Add constant, remove field
2. `src/policyflow/llm.py` - Add `model` parameter
3. `src/policyflow/nodes/llm_node.py` - Add class attr + instance param
4. `src/policyflow/nodes/classifier.py`
5. `src/policyflow/nodes/sentiment.py`
6. `src/policyflow/nodes/data_extractor.py`
7. `src/policyflow/nodes/sampler.py`
8. `src/policyflow/nodes/criterion.py`
9. `src/policyflow/nodes/subcriterion.py`
