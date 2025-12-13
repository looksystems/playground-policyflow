# Flexible Model Configuration Implementation Plan

## Overview
Add support for configuring different LLM models at multiple levels:
- **CLI task level**: Different models for generate, benchmark, analyze, optimize, hypothesize
- **Node type level**: Different models for ClassifierNode, SentimentNode, DataExtractorNode, SamplerNode
- **Instance level**: Individual nodes can still override via `model` parameter
- **LMStudio support**: OpenAI-compatible format with OPENAI_API_BASE

## Model Selection Hierarchy (Highest to Lowest Priority)
1. **Explicit parameter**: `model="..."` in workflow.yaml or CLI `--model` flag
2. **Type-specific default**: From env vars like `CLASSIFIER_MODEL`, `GENERATE_MODEL`
3. **Global default**: From `POLICY_EVAL_MODEL` (existing)
4. **Hardcoded fallback**: `"anthropic/claude-sonnet-4-20250514"`

## Environment Variables

### Node Type Models (Optional)
```bash
CLASSIFIER_MODEL=anthropic/claude-sonnet-4-20250514
DATA_EXTRACTOR_MODEL=anthropic/claude-sonnet-4-20250514
SENTIMENT_MODEL=anthropic/claude-sonnet-4-20250514
SAMPLER_MODEL=anthropic/claude-sonnet-4-20250514
```

### CLI Task Models (Optional)
```bash
GENERATE_MODEL=anthropic/claude-opus-4-5-20251101
ANALYZE_MODEL=anthropic/claude-sonnet-4-20250514
HYPOTHESIZE_MODEL=anthropic/claude-opus-4-5-20251101
OPTIMIZE_MODEL=anthropic/claude-sonnet-4-20250514
```

### Global Default (Existing)
```bash
POLICY_EVAL_MODEL=anthropic/claude-sonnet-4-20250514
```

### LMStudio Support
```bash
OPENAI_API_BASE=http://localhost:1234/v1
POLICY_EVAL_MODEL=openai/local-model-name
```

## Implementation Steps

### Step 1: Add ModelConfig to config.py
**File**: `src/policyflow/config.py`

Add new Pydantic model before WorkflowConfig:

```python
class ModelConfig(BaseModel):
    """Configuration for model selection at different levels."""

    # Global default
    default_model: str = Field(
        default_factory=lambda: os.getenv(
            "POLICY_EVAL_MODEL",
            "anthropic/claude-sonnet-4-20250514"
        )
    )

    # Node type defaults
    classifier_model: str | None = Field(
        default_factory=lambda: os.getenv("CLASSIFIER_MODEL")
    )
    data_extractor_model: str | None = Field(
        default_factory=lambda: os.getenv("DATA_EXTRACTOR_MODEL")
    )
    sentiment_model: str | None = Field(
        default_factory=lambda: os.getenv("SENTIMENT_MODEL")
    )
    sampler_model: str | None = Field(
        default_factory=lambda: os.getenv("SAMPLER_MODEL")
    )

    # CLI task defaults
    generate_model: str | None = Field(
        default_factory=lambda: os.getenv("GENERATE_MODEL")
    )
    analyze_model: str | None = Field(
        default_factory=lambda: os.getenv("ANALYZE_MODEL")
    )
    hypothesize_model: str | None = Field(
        default_factory=lambda: os.getenv("HYPOTHESIZE_MODEL")
    )
    optimize_model: str | None = Field(
        default_factory=lambda: os.getenv("OPTIMIZE_MODEL")
    )

    def get_model_for_node_type(self, node_type: str) -> str:
        """Get model for a specific node type with fallback to default."""
        mapping = {
            "ClassifierNode": self.classifier_model,
            "DataExtractorNode": self.data_extractor_model,
            "SentimentNode": self.sentiment_model,
            "SamplerNode": self.sampler_model,
        }
        return mapping.get(node_type) or self.default_model

    def get_model_for_task(self, task: str) -> str:
        """Get model for a specific CLI task with fallback to default."""
        mapping = {
            "generate": self.generate_model,
            "analyze": self.analyze_model,
            "hypothesize": self.hypothesize_model,
            "optimize": self.optimize_model,
        }
        return mapping.get(task) or self.default_model
```

Then add to WorkflowConfig:

```python
class WorkflowConfig(BaseModel):
    # ... existing fields ...

    models: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model selection configuration",
    )
```

### Step 2: Update LLMNode Model Selection
**File**: `src/policyflow/nodes/llm_node.py`

Update `__init__` method (around line 28-46):

```python
def __init__(
    self,
    config: WorkflowConfig,
    model: str | None = None,
    cache_ttl: int = 3600,
    rate_limit: int = None,
):
    """
    Initialize LLM node with caching and rate limiting.

    Args:
        config: Workflow configuration
        model: LLM model identifier (uses config-based default if not provided)
        cache_ttl: Cache time-to-live in seconds, 0 = disabled
        rate_limit: Requests per minute, None = unlimited
    """
    super().__init__(max_retries=config.max_retries)
    self.config = config

    # Model selection hierarchy: explicit param > config for node type > class default
    if model is not None:
        self.model = model
    else:
        node_type = self.__class__.__name__
        self.model = config.models.get_model_for_node_type(node_type)

    self.cache_ttl = cache_ttl
    self.rate_limit = rate_limit
    self._instance_id = id(self)

    # ... rest of init unchanged ...
```

**Note**: Keep `default_model` class attribute for backward compatibility, but it's now superseded by config.

### Step 3: Add --model Flags to CLI Commands
**File**: `src/policyflow/benchmark/cli.py`

#### 3a. Update generate-dataset command (around line 180)

Add parameter to function signature:
```python
model: Annotated[
    str | None,
    typer.Option("--model", "-m", help="Model to use for generation"),
] = None,
```

Update model resolution in function body:
```python
# After loading policy, before creating generator
config = WorkflowConfig()
effective_model = model or config.models.get_model_for_task("generate")

# When creating generator (only pass model if mode uses LLM)
generator = create_generator(
    mode=mode,
    model=effective_model if mode in ("llm", "hybrid") else None
)
```

#### 3b. Update analyze command (around line 260)

Add parameter:
```python
model: Annotated[
    str | None,
    typer.Option("--model", "-m", help="Model to use for analysis"),
] = None,
```

Update model resolution:
```python
config = WorkflowConfig()
effective_model = model or config.models.get_model_for_task("analyze")

analyzer = create_analyzer(
    mode=mode,
    model=effective_model if mode in ("llm", "hybrid") else None
)
```

#### 3c. Update hypothesize command (around line 340)

Add parameter:
```python
model: Annotated[
    str | None,
    typer.Option("--model", "-m", help="Model to use for hypothesis generation"),
] = None,
```

Update model resolution:
```python
config = WorkflowConfig()
effective_model = model or config.models.get_model_for_task("hypothesize")

hypothesis_generator = create_hypothesis_generator(
    mode=mode,
    model=effective_model if mode in ("llm", "hybrid") else None
)
```

#### 3d. Update optimize command (around line 400)

Add parameter:
```python
model: Annotated[
    str | None,
    typer.Option("--model", "-m", help="Model to use for optimization"),
] = None,
```

Update model resolution and pass to components:
```python
config = WorkflowConfig()
effective_model = model or config.models.get_model_for_task("optimize")

# Pass model to analyzer and hypothesis generator
analyzer = create_analyzer(mode=analyze_mode, model=effective_model)
hypothesis_generator = create_hypothesis_generator(mode=hypothesis_mode, model=effective_model)
```

### Step 4: Update .env.example
**File**: `.env.example`

Add comprehensive model configuration section:

```bash
# ============================================================================
# LLM Model Configuration
# ============================================================================

# Global default model (used when more specific models not configured)
POLICY_EVAL_MODEL=anthropic/claude-sonnet-4-20250514

# Node Type Models (optional - override defaults for specific node types)
# CLASSIFIER_MODEL=anthropic/claude-sonnet-4-20250514
# DATA_EXTRACTOR_MODEL=anthropic/claude-sonnet-4-20250514
# SENTIMENT_MODEL=anthropic/claude-sonnet-4-20250514
# SAMPLER_MODEL=anthropic/claude-sonnet-4-20250514

# Benchmark Task Models (optional - override for benchmark operations)
# GENERATE_MODEL=anthropic/claude-opus-4-5-20251101  # Dataset generation
# ANALYZE_MODEL=anthropic/claude-sonnet-4-20250514   # Failure analysis
# HYPOTHESIZE_MODEL=anthropic/claude-opus-4-5-20251101  # Hypothesis generation
# OPTIMIZE_MODEL=anthropic/claude-sonnet-4-20250514  # Optimization loop

# ============================================================================
# LMStudio / Local Model Support
# ============================================================================
# Use OpenAI-compatible format for model names with OPENAI_API_BASE
# OPENAI_API_BASE=http://localhost:1234/v1
# POLICY_EVAL_MODEL=openai/local-model-name
# CLASSIFIER_MODEL=openai/llama-3-8b
# ANALYZE_MODEL=openai/mixtral-8x7b

# API Keys (LiteLLM uses standard env vars)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# ... rest of existing config ...
```

### Step 5: Update Documentation

#### Step 5a: Update README.md
**File**: `README.md`

**Section to update**: "Configuration" section (lines 24-58)

Add new subsection after the basic config table:

```markdown
### Multi-Level Model Configuration

PolicyFlow supports configuring different models at multiple levels:

**Node Type Defaults**: Configure different models for different node types
```env
CLASSIFIER_MODEL=anthropic/claude-sonnet-4-20250514
SENTIMENT_MODEL=anthropic/claude-haiku-3-5-20250318  # Use faster model for sentiment
DATA_EXTRACTOR_MODEL=anthropic/claude-opus-4-5-20251101  # Use powerful model for extraction
```

**CLI Task Defaults**: Configure different models for benchmark operations
```env
GENERATE_MODEL=anthropic/claude-opus-4-5-20251101  # Use powerful model for generation
ANALYZE_MODEL=anthropic/claude-sonnet-4-20250514   # Use balanced model for analysis
```

**Local Models (LMStudio)**: Use OpenAI-compatible local models
```env
OPENAI_API_BASE=http://localhost:1234/v1
CLASSIFIER_MODEL=openai/llama-3-8b
SENTIMENT_MODEL=openai/mistral-7b
```

**Model Selection Priority** (highest to lowest):
1. Explicit parameter in workflow.yaml or CLI `--model` flag
2. Type-specific env var (e.g., `CLASSIFIER_MODEL`, `GENERATE_MODEL`)
3. Global default (`POLICY_EVAL_MODEL`)
4. Hardcoded fallback (`anthropic/claude-sonnet-4-20250514`)
```

**Update the environment variables table** to add new rows:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLASSIFIER_MODEL` | `POLICY_EVAL_MODEL` | Default model for ClassifierNode |
| `DATA_EXTRACTOR_MODEL` | `POLICY_EVAL_MODEL` | Default model for DataExtractorNode |
| `SENTIMENT_MODEL` | `POLICY_EVAL_MODEL` | Default model for SentimentNode |
| `SAMPLER_MODEL` | `POLICY_EVAL_MODEL` | Default model for SamplerNode |
| `GENERATE_MODEL` | `POLICY_EVAL_MODEL` | Model for generate-dataset command |
| `ANALYZE_MODEL` | `POLICY_EVAL_MODEL` | Model for analyze command |
| `HYPOTHESIZE_MODEL` | `POLICY_EVAL_MODEL` | Model for hypothesize command |
| `OPTIMIZE_MODEL` | `POLICY_EVAL_MODEL` | Model for optimize command |
| `OPENAI_API_BASE` | - | OpenAI-compatible endpoint (for LMStudio) |

#### Step 5b: Update docs/cli-cheatsheet.md
**File**: `docs/cli-cheatsheet.md`

**Updates needed**:

1. Add `--model` flag to generate-dataset examples (line 71):
```bash
policyflow generate-dataset --policy n.yaml -o d.yaml --mode llm --model anthropic/claude-opus-4-5-20251101
```

2. Update analyze examples (line 106):
```bash
policyflow analyze -r report.yaml -w workflow.yaml --mode llm --model anthropic/claude-opus-4-5-20251101 -o analysis.yaml
```

3. Update hypothesize examples (line 121):
```bash
policyflow hypothesize -a analysis.yaml -w workflow.yaml --mode llm --model anthropic/claude-opus-4-5-20251101 -o hypotheses.yaml
```

4. Update optimize examples (line 136):
```bash
policyflow optimize -w workflow.yaml -d dataset.yaml --model anthropic/claude-opus-4-5-20251101 -o optimized.yaml
```

5. Add new rows to Environment Variables table (after line 200):

| Variable | Default | Description |
|----------|---------|-------------|
| `CLASSIFIER_MODEL` | `POLICY_EVAL_MODEL` | Default for ClassifierNode |
| `DATA_EXTRACTOR_MODEL` | `POLICY_EVAL_MODEL` | Default for DataExtractorNode |
| `SENTIMENT_MODEL` | `POLICY_EVAL_MODEL` | Default for SentimentNode |
| `SAMPLER_MODEL` | `POLICY_EVAL_MODEL` | Default for SamplerNode |
| `GENERATE_MODEL` | `POLICY_EVAL_MODEL` | Default for generate-dataset |
| `ANALYZE_MODEL` | `POLICY_EVAL_MODEL` | Default for analyze |
| `HYPOTHESIZE_MODEL` | `POLICY_EVAL_MODEL` | Default for hypothesize |
| `OPTIMIZE_MODEL` | `POLICY_EVAL_MODEL` | Default for optimize |

#### Step 5c: Update docs/USERGUIDE.md
**File**: `docs/USERGUIDE.md`

**Add new section** after "Configuration" section (after line 20):

```markdown
## Model Configuration

PolicyFlow supports flexible model configuration at multiple levels, allowing you to optimize cost and performance for different tasks.

### Configuration Levels

#### 1. Global Default
Set a default model for all operations:
```bash
export POLICY_EVAL_MODEL="anthropic/claude-sonnet-4-20250514"
```

#### 2. Node Type Defaults
Configure different models for specific node types in workflows:
```bash
export CLASSIFIER_MODEL="anthropic/claude-haiku-3-5-20250318"  # Fast model for classification
export SENTIMENT_MODEL="anthropic/claude-haiku-3-5-20250318"   # Fast model for sentiment
export DATA_EXTRACTOR_MODEL="anthropic/claude-opus-4-5-20251101"  # Powerful model for extraction
```

#### 3. CLI Task Defaults
Configure different models for benchmark operations:
```bash
export GENERATE_MODEL="anthropic/claude-opus-4-5-20251101"  # High quality test generation
export ANALYZE_MODEL="anthropic/claude-sonnet-4-20250514"   # Balanced analysis
export HYPOTHESIZE_MODEL="anthropic/claude-opus-4-5-20251101"  # Creative hypothesis generation
```

#### 4. Runtime Overrides
Override model for specific operations:
```bash
# Override model for dataset generation
policyflow generate-dataset --policy n.yaml -o dataset.yaml --model anthropic/claude-opus-4-5-20251101

# Override model for analysis
policyflow analyze -r report.yaml -w workflow.yaml --model anthropic/claude-opus-4-5-20251101
```

#### 5. Workflow-Level Overrides
Specify model for individual nodes in workflow.yaml:
```yaml
workflow:
  nodes:
    - id: classifier
      type: ClassifierNode
      params:
        categories: [approve, reject, review]
        model: anthropic/claude-opus-4-5-20251101  # Use powerful model for this classifier
```

### Model Selection Priority

When PolicyFlow needs a model, it uses this priority order (highest to lowest):

1. **Explicit parameter**: `model` in workflow.yaml or CLI `--model` flag
2. **Type-specific env var**: `CLASSIFIER_MODEL`, `GENERATE_MODEL`, etc.
3. **Global default**: `POLICY_EVAL_MODEL`
4. **Hardcoded fallback**: `anthropic/claude-sonnet-4-20250514`

### Using Local Models (LMStudio)

PolicyFlow supports local models via LMStudio's OpenAI-compatible API:

1. **Start LMStudio** and load a model
2. **Enable the OpenAI-compatible server** (usually on `http://localhost:1234`)
3. **Configure environment variables**:

```bash
export OPENAI_API_BASE="http://localhost:1234/v1"
export POLICY_EVAL_MODEL="openai/your-model-name"
```

Example with different local models:
```bash
export OPENAI_API_BASE="http://localhost:1234/v1"
export POLICY_EVAL_MODEL="openai/llama-3-8b"
export CLASSIFIER_MODEL="openai/llama-3-8b"
export ANALYZE_MODEL="openai/mixtral-8x7b"  # More powerful for analysis
```

### Cost Optimization Strategies

**Strategy 1: Fast models for simple tasks**
```bash
export POLICY_EVAL_MODEL="anthropic/claude-sonnet-4-20250514"  # Default
export CLASSIFIER_MODEL="anthropic/claude-haiku-3-5-20250318"  # Faster/cheaper
export SENTIMENT_MODEL="anthropic/claude-haiku-3-5-20250318"   # Faster/cheaper
```

**Strategy 2: Powerful models for complex tasks**
```bash
export POLICY_EVAL_MODEL="anthropic/claude-sonnet-4-20250514"  # Default
export DATA_EXTRACTOR_MODEL="anthropic/claude-opus-4-5-20251101"  # Complex extraction
export ANALYZE_MODEL="anthropic/claude-opus-4-5-20251101"  # Deep analysis
```

**Strategy 3: Local models for development**
```bash
export OPENAI_API_BASE="http://localhost:1234/v1"
export POLICY_EVAL_MODEL="openai/llama-3-8b"  # Local model for development
export GENERATE_MODEL="anthropic/claude-opus-4-5-20251101"  # Cloud model for production generation
```

### Available Model Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `POLICY_EVAL_MODEL` | Global default | `anthropic/claude-sonnet-4-20250514` |
| `CLASSIFIER_MODEL` | ClassifierNode | Falls back to global |
| `DATA_EXTRACTOR_MODEL` | DataExtractorNode | Falls back to global |
| `SENTIMENT_MODEL` | SentimentNode | Falls back to global |
| `SAMPLER_MODEL` | SamplerNode | Falls back to global |
| `GENERATE_MODEL` | generate-dataset command | Falls back to global |
| `ANALYZE_MODEL` | analyze command | Falls back to global |
| `HYPOTHESIZE_MODEL` | hypothesize command | Falls back to global |
| `OPTIMIZE_MODEL` | optimize command | Falls back to global |
| `OPENAI_API_BASE` | OpenAI-compatible endpoint | - |
```

## Backward Compatibility

✅ **Existing .env files**: Work without changes. If only `POLICY_EVAL_MODEL` is set, all operations use it.

✅ **Existing workflow.yaml files**: Continue to work. Nodes without explicit `model` get defaults from config.

✅ **Existing CLI usage**: Commands work without `--model` flag, using env var defaults or global default.

✅ **Code compatibility**: `LLMNode.default_model` class attribute remains for safety.

## Testing Recommendations

### Unit Tests
1. Test `ModelConfig.get_model_for_node_type()` with various env var combinations
2. Test `ModelConfig.get_model_for_task()` with various env var combinations
3. Test `LLMNode` model selection hierarchy

### Integration Tests
1. Test each CLI command with `--model` flag override
2. Test workflows with mixed node configurations
3. Test env var precedence

### Manual Testing
1. Configure LMStudio with `OPENAI_API_BASE` and verify routing
2. Run benchmark with different models for different tasks
3. Create workflow with per-node model overrides

## Critical Files to Modify

### Core Implementation
1. `src/policyflow/config.py` - Add ModelConfig class
2. `src/policyflow/nodes/llm_node.py` - Update model selection in __init__
3. `src/policyflow/benchmark/cli.py` - Add --model flags to all commands
4. `.env.example` - Document all new environment variables

### Documentation
5. `README.md` - Update configuration section with new model config options
6. `docs/cli-cheatsheet.md` - Add --model flags to benchmark command examples
7. `docs/USERGUIDE.md` - Add model configuration section with examples

## No Changes Needed

- `src/policyflow/benchmark/generator.py` - Already accepts optional model parameter
- `src/policyflow/benchmark/analyzer.py` - Already accepts optional model parameter
- `src/policyflow/benchmark/hypothesis.py` - Already accepts optional model parameter
- Individual node classes (ClassifierNode, SentimentNode, etc.) - Already call super().__init__ correctly
- `src/policyflow/llm.py` - No changes needed
- `src/policyflow/workflow_builder.py` - No changes needed

## Example Usage

### Using different models per task
```bash
# .env file
POLICY_EVAL_MODEL=anthropic/claude-sonnet-4-20250514
GENERATE_MODEL=anthropic/claude-opus-4-5-20251101
ANALYZE_MODEL=anthropic/claude-opus-4-5-20251101
```

### Using LMStudio for specific nodes
```bash
# .env file
OPENAI_API_BASE=http://localhost:1234/v1
POLICY_EVAL_MODEL=anthropic/claude-sonnet-4-20250514
CLASSIFIER_MODEL=openai/llama-3-8b
SENTIMENT_MODEL=openai/mistral-7b
```

### CLI override
```bash
# Override model for specific run
policyflow generate-dataset --policy policy.md --output dataset.yaml --model anthropic/claude-opus-4-5-20251101

# Or use env var default
policyflow generate-dataset --policy policy.md --output dataset.yaml
```

### Workflow YAML override
```yaml
workflow:
  nodes:
    - id: classifier
      type: ClassifierNode
      params:
        categories: [approve, reject]
        model: anthropic/claude-opus-4-5-20251101  # Override for this specific node
```
