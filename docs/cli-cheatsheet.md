# CLI Cheatsheet

Quick reference for all PolicyFlow CLI commands.

## Core Commands

### parse - Parse policy structure

```bash
policyflow parse -p policy.md                     # Display structure
policyflow parse -p policy.md --format yaml       # Output as YAML
policyflow parse -p policy.md --save-workflow w.yaml
policyflow parse -p policy.md --save-normalized n.yaml --save-workflow w.yaml
```

| Option | Short | Description |
|--------|-------|-------------|
| `--policy` | `-p` | Path to policy markdown (required) |
| `--model` | `-m` | LiteLLM model identifier |
| `--save-workflow` | | Save workflow YAML |
| `--save-normalized` | | Save normalized policy YAML |
| `--format` | | Output: `pretty` or `yaml` |

### eval - Evaluate text

```bash
policyflow eval -p policy.md -i "text to evaluate"
policyflow eval -p policy.md -f input.txt
policyflow eval -w workflow.yaml -i "text"        # Use cached workflow
policyflow eval -p policy.md -i "text" --format minimal
```

| Option | Short | Description |
|--------|-------|-------------|
| `--policy` | `-p` | Path to policy markdown |
| `--workflow` | `-w` | Path to workflow YAML |
| `--input` | `-i` | Text to evaluate (inline) |
| `--input-file` | `-f` | File containing text |
| `--model` | `-m` | LiteLLM model identifier |
| `--format` | | Output: `pretty`, `yaml`, `minimal` |
| `--save-workflow` | | Save parsed workflow |

### batch - Batch evaluate

```bash
policyflow batch -w workflow.yaml --inputs texts.yaml -o results.yaml
policyflow batch -p policy.md --inputs texts.yaml -o results.yaml
```

| Option | Short | Description |
|--------|-------|-------------|
| `--policy` | `-p` | Path to policy markdown |
| `--workflow` | `-w` | Path to workflow YAML |
| `--inputs` | | YAML file with inputs list (required) |
| `--output` | `-o` | Output YAML file (required) |
| `--model` | `-m` | LiteLLM model identifier |

**Input file format:**
```yaml
- "First text to evaluate"
- "Second text to evaluate"
```

## Benchmark Commands

### generate-dataset - Generate test cases

```bash
policyflow generate-dataset --policy normalized.yaml --output dataset.yaml
policyflow generate-dataset --policy n.yaml -o d.yaml --cases-per-criterion 5
policyflow generate-dataset --policy n.yaml -o d.yaml --include-edge-cases --mode llm
```

| Option | Description |
|--------|-------------|
| `--policy` | Normalized policy YAML (required) |
| `--output` | Output golden dataset (required) |
| `--cases-per-criterion` | Test cases per criterion (default: 3) |
| `--include-edge-cases` | Include edge cases |
| `--strategies` | Edge case strategies (comma-separated) |
| `--mode` | Generation: `template`, `llm`, `hybrid` |
| `--model` | LLM model for hybrid/llm mode |

### benchmark - Run benchmark

```bash
policyflow benchmark -w workflow.yaml -d dataset.yaml
policyflow benchmark -w workflow.yaml -d dataset.yaml -o report.yaml
policyflow benchmark -w workflow.yaml -d dataset.yaml --category compliance
```

| Option | Short | Description |
|--------|-------|-------------|
| `--workflow` | `-w` | Workflow YAML (required) |
| `--dataset` | `-d` | Golden dataset YAML (required) |
| `--output` | `-o` | Output report path |
| `--id` | | Workflow version identifier |
| `--category` | `-c` | Filter by test category |

### analyze - Analyze failures

```bash
policyflow analyze -r report.yaml -w workflow.yaml
policyflow analyze -r report.yaml -w workflow.yaml --mode llm -o analysis.yaml
```

| Option | Short | Description |
|--------|-------|-------------|
| `--report` | `-r` | Benchmark report (required) |
| `--workflow` | `-w` | Workflow YAML (required) |
| `--output` | `-o` | Output analysis path |
| `--mode` | | Analysis: `rule_based`, `llm`, `hybrid` |
| `--model` | | LLM model for hybrid/llm mode |

### hypothesize - Generate improvements

```bash
policyflow hypothesize -a analysis.yaml -w workflow.yaml
policyflow hypothesize -a analysis.yaml -w workflow.yaml -o hypotheses.yaml
```

| Option | Short | Description |
|--------|-------|-------------|
| `--analysis` | `-a` | Analysis report (required) |
| `--workflow` | `-w` | Workflow YAML (required) |
| `--output` | `-o` | Output hypotheses path |
| `--mode` | | Generation: `template`, `llm`, `hybrid` |
| `--model` | | LLM model for hybrid/llm mode |

### optimize - Auto-optimize workflow

```bash
policyflow optimize -w workflow.yaml -d dataset.yaml -o optimized.yaml
policyflow optimize -w workflow.yaml -d dataset.yaml --max-iterations 10 --target 0.95
```

| Option | Description |
|--------|-------------|
| `--workflow` | Workflow YAML (required) |
| `--dataset` | Golden dataset (required) |
| `--output` | Output optimized workflow |
| `--max-iterations` | Max iterations (default: 10) |
| `--target` | Target accuracy (0.0-1.0) |
| `--patience` | Stop after N iterations without improvement |
| `--model` | LLM model for analysis |

### improve - Full improvement loop

```bash
policyflow improve -w workflow.yaml -d dataset.yaml
policyflow improve -w workflow.yaml -d dataset.yaml --mode hybrid
```

| Option | Description |
|--------|-------------|
| `--workflow` | Workflow YAML (required) |
| `--dataset` | Golden dataset (required) |
| `--output` | Output improved workflow |
| `--max-iterations` | Max iterations (default: 5) |
| `--target` | Target accuracy |
| `--mode` | Analysis mode: `rule_based`, `llm`, `hybrid` |
| `--model` | LLM model |

## Experiment Commands

```bash
policyflow experiments list                    # List all experiments
policyflow experiments list --dir ./exps       # Custom directory
policyflow experiments best                    # Show best experiment
policyflow experiments compare exp_001 exp_002 # Compare two experiments
```

| Command | Description |
|---------|-------------|
| `list` | List all recorded experiments |
| `best` | Show best-performing experiment |
| `compare EXP1 EXP2` | Compare two experiments |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POLICY_EVAL_MODEL` | `anthropic/claude-sonnet-4-20250514` | LLM model |
| `POLICY_EVAL_TEMPERATURE` | `0.0` | LLM temperature |
| `POLICY_EVAL_MAX_RETRIES` | `3` | Max retry attempts |
| `POLICY_EVAL_RETRY_WAIT` | `2` | Seconds between retries |
| `POLICY_EVAL_CONFIDENCE_HIGH` | `0.8` | High confidence threshold |
| `POLICY_EVAL_CONFIDENCE_LOW` | `0.5` | Low confidence threshold |
| `POLICY_EVAL_CACHE_ENABLED` | `true` | Enable LLM caching |
| `POLICY_EVAL_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `POLICY_EVAL_CACHE_DIR` | `.cache` | Cache directory |
| `POLICY_EVAL_THROTTLE_ENABLED` | `false` | Enable rate limiting |
| `POLICY_EVAL_THROTTLE_RPM` | `60` | Requests per minute |
| `PHOENIX_ENABLED` | `false` | Enable Phoenix tracing |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6007` | Phoenix endpoint |
| `PHOENIX_PROJECT_NAME` | `policyflowuator` | Phoenix project |

## Common Workflows

### One-time evaluation

```bash
policyflow eval -p policy.md -i "Your text here"
```

### Production workflow (cache & reuse)

```bash
# Step 1: Parse once
policyflow parse -p policy.md --save-normalized n.yaml --save-workflow w.yaml

# Step 2: Evaluate many
policyflow eval -w w.yaml -i "Text 1"
policyflow eval -w w.yaml -i "Text 2"
policyflow batch -w w.yaml --inputs batch.yaml -o results.yaml
```

### Benchmarking workflow

```bash
# Generate test cases
policyflow generate-dataset --policy n.yaml -o dataset.yaml

# Run benchmark
policyflow benchmark -w w.yaml -d dataset.yaml -o report.yaml

# Analyze and improve
policyflow analyze -r report.yaml -w w.yaml -o analysis.yaml
policyflow hypothesize -a analysis.yaml -w w.yaml -o hypotheses.yaml

# Or use the all-in-one command
policyflow improve -w w.yaml -d dataset.yaml
```

### Optimization workflow

```bash
policyflow optimize \
    --workflow workflow.yaml \
    --dataset golden_dataset.yaml \
    --max-iterations 10 \
    --target 0.95 \
    --patience 3 \
    --output optimized_workflow.yaml
```
