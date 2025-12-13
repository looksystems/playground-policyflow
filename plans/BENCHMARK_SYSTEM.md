# Self-Verifying & Improving Evaluation System

## Implementation Status ✅

**All phases complete, production-ready** (Dec 2024)

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| models.py | ✅ Complete | 28 tests | Added `category` field to TestCaseResult, `metadata` to FailurePattern |
| protocols.py | ✅ Complete | 22 tests | |
| loader.py | ✅ Complete | 11 tests | |
| comparator.py | ✅ Complete | 7 tests | |
| metrics.py | ✅ Complete | 9 tests | |
| runner.py | ✅ Complete | 8 tests | Now populates category field |
| analyzer.py | ✅ Complete | 11 tests | Full LLM integration, uses metadata |
| hypothesis.py | ✅ Complete | 13 tests | Full LLM integration, uses metadata |
| tracker.py | ✅ Complete | 12 tests | Proper logging for exceptions |
| cli.py | ✅ Complete | 11 tests | All commands implemented |
| generator.py | ✅ Complete | 19 tests | LLM-enhanced test case generation |
| applier.py | ✅ Complete | 14 tests | Configurable route interception |
| optimizer.py | ✅ Complete | 22 tests | Uses real SimpleBenchmarkRunner |
| test_integration.py | ✅ Complete | 7 tests | Full integration tests |
| __init__.py exports | ✅ Complete | - | |

**Total: 197 benchmark-specific tests passing**

---

## Review & Critique (Dec 2024) - ALL RESOLVED ✅

### Critical Issues - ALL FIXED ✅

#### 1. ~~Broken Category Analysis~~ FIXED
**File:** `analyzer.py:279-281`
```python
def _get_category(self, result: TestCaseResult) -> str:
    return result.category  # Now returns actual category
```
**Fix:** Added `category` field to `TestCaseResult` model, updated runner to populate it.

#### 2. ~~Optimizer Uses Stub Benchmark~~ FIXED
**File:** `optimizer.py:283-295`
```python
def _run_benchmark(self, workflow, dataset) -> BenchmarkReport:
    config = BenchmarkConfig(workflow_id=workflow.title)
    runner = SimpleBenchmarkRunner(config)
    return runner.run(workflow, dataset.test_cases)
```
**Fix:** Now uses `SimpleBenchmarkRunner` to execute actual benchmarks.

#### 3. ~~Non-Deterministic Test Case IDs~~ FIXED
**File:** `generator.py:423-432`
```python
content = f"{category}:{criterion}:{index}"
unique = hashlib.sha256(content.encode()).hexdigest()[:8]
```
**Fix:** Uses hash-based deterministic IDs instead of UUIDs.

### CLI Commands - ALL IMPLEMENTED ✅

| Planned Command | Status |
|-----------------|--------|
| `policyflow generate-dataset` | ✅ Implemented |
| `policyflow optimize` | ✅ Implemented |
| `policyflow improve` | ✅ Implemented |

### LLM Integration Status - COMPLETE ✅

All three LLM components are now fully implemented:

| Component | Status |
|-----------|--------|
| `HybridDatasetGenerator` | ✅ Full LLM integration with graceful fallback |
| `LLMEnhancedAnalyzer` | ✅ Full LLM integration with graceful fallback |
| `LLMHypothesisGenerator` | ✅ Full LLM integration with graceful fallback |

**Features:**
- All components accept optional `model` parameter
- Graceful fallback to template/rule-based when no model configured
- Error handling with automatic fallback on LLM failures

### Test Coverage - COMPLETE ✅

- **Integration tests added** - `test_integration.py` with 7 tests
- **Edge cases covered:** empty datasets, single criterion policies, dataset idempotency
- **LLM tests:** parsing, fallback behavior, factory functions
- **Code quality tests:** metadata extraction, logging, route interception
- **Total tests:** 167 → 197 (+30 new tests)

### Success Criteria Status

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `generate-dataset` command | ✅ MET |
| 2 | All test categories generated | ✅ MET |
| 3 | Intermediate state expectations | ✅ MET |
| 4 | Benchmark with accuracy metrics | ✅ MET |
| 5 | Per-criterion P/R/F1 | ✅ MET |
| 6 | Identify problematic categories | ✅ MET |
| 7 | 3+ actionable hypotheses | ✅ MET |
| 8 | `optimize` with budget | ✅ MET |
| 9 | Convergence with clear reason | ✅ MET |
| 10 | Track/compare experiments | ✅ MET |
| 11 | Protocol-based interfaces | ✅ MET |
| 12 | Rule-based AND LLM modes | ✅ MET (full LLM integration) |
| 13 | CLI + Python API functional | ✅ MET |
| 14 | Future optimizers pluggable | ✅ MET |

**Score: 14/14 fully met**

### All Fixes Applied ✅

**Phase 1: Critical Bugs - COMPLETE ✅**
1. ~~Fix `_get_category()` to extract from test case metadata~~ DONE
2. ~~Implement real `_run_benchmark()` using `SimpleBenchmarkRunner`~~ DONE
3. ~~Use hash-based deterministic IDs instead of UUIDs~~ DONE

**Phase 2: Missing CLI - COMPLETE ✅**
4. ~~Add `generate-dataset` CLI command with all options~~ DONE
5. ~~Add `optimize` CLI command with budget options~~ DONE
6. ~~Add `improve` convenience command~~ DONE

**Phase 3: Code Quality - COMPLETE ✅**
7. ~~Add integration tests (generator → optimizer full loop)~~ DONE
8. ~~Fix silent failures in optimizer (log rejected hypotheses)~~ DONE
9. ~~Strengthen regex parsing in hypothesis extraction~~ DONE (uses metadata)
10. ~~Add logging for silent exceptions in tracker~~ DONE
11. ~~Fix incomplete rewiring in applier~~ DONE (supports any route)
12. ~~Fix brittle string parsing in analyzer~~ DONE (uses metadata)

**Phase 4: LLM Integration - COMPLETE ✅**
13. ~~Wire actual LLM calls in HybridDatasetGenerator~~ DONE
14. ~~Wire actual LLM calls in LLMEnhancedAnalyzer~~ DONE
15. ~~Wire actual LLM calls in LLMHypothesisGenerator~~ DONE

---

## Goal
Build a lightweight POC that:
1. **Benchmarks** workflow performance against `golden_dataset.yaml`
2. **Analyzes** failures to identify patterns
3. **Generates hypotheses** for workflow improvement
4. **Tracks experiments** to measure improvement over iterations

## Design Principles
- **Protocol-first**: Clean interfaces (Python Protocols) for easy swapping
- **Minimal data models**: Lean Pydantic classes that compose well
- **Dual-mode analysis**: Both rule-based AND LLM-powered analyzers available, selectable at runtime
- **Human-in-the-loop**: Review hypotheses before applying
- **Incremental**: Can run partial pipeline (just benchmark, or benchmark + analyze)
- **Dual interface**: CLI commands AND Python API from the start
- **YAML persistence**: Experiments tracked in simple YAML files

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION LOOP                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Golden   │───>│ Benchmark│───>│ Analyzer │───>│ Optimizer│──┐       │
│  │ Dataset  │    │ Runner   │    │          │    │          │  │       │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │       │
│                       │               │               │         │       │
│                       v               v               v         │       │
│                  ┌─────────┐    ┌─────────┐    ┌─────────┐     │       │
│                  │Benchmark│    │AnalysisReport│ │Hypotheses│    │       │
│                  │ Report  │    │         │    │         │     │       │
│                  └─────────┘    └─────────┘    └─────────┘     │       │
│                                                                 │       │
│  ┌──────────────────────────────────────────────────────────────┘       │
│  │                                                                       │
│  │  Human Review  ───>  Apply Changes  ───>  Re-run Benchmark           │
│  │                                                                       │
│  └───────────────────────────────────────────────────────────────────────│
│                                                                          │
│                    ┌──────────────┐                                      │
│                    │  Experiment  │  (tracks all iterations)             │
│                    │   Tracker    │                                      │
│                    └──────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Data Models (`src/policyflow/benchmark/models.py`)

### Test Case Loading
```python
@dataclass
class GoldenTestCase:
    id: str
    name: str
    input_text: str
    expected: ExpectedResult
    category: str
    notes: str

@dataclass
class ExpectedResult:
    policy_satisfied: bool
    criterion_results: dict[str, CriterionExpectation]

@dataclass
class CriterionExpectation:
    met: bool
    sub_results: dict[str, SubCriterionExpectation] | None = None
```

### Benchmark Results
```python
@dataclass
class TestCaseResult:
    test_id: str
    expected: ExpectedResult
    actual: EvaluationResult | None  # None if error
    error: str | None
    duration_ms: float

@dataclass
class BenchmarkReport:
    workflow_id: str
    timestamp: datetime
    results: list[TestCaseResult]
    metrics: BenchmarkMetrics
    config: dict  # snapshot of workflow config
```

### Metrics
```python
@dataclass
class BenchmarkMetrics:
    overall_accuracy: float
    criterion_metrics: dict[str, CriterionMetrics]
    category_accuracy: dict[str, float]  # by test category
    confidence_calibration: ConfidenceCalibration

@dataclass
class CriterionMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: ConfusionMatrix

@dataclass
class ConfidenceCalibration:
    high_confidence_accuracy: float
    medium_confidence_accuracy: float
    low_confidence_accuracy: float
```

### Analysis & Hypotheses
```python
@dataclass
class AnalysisReport:
    patterns: list[FailurePattern]
    problematic_criteria: list[ProblematicCriterion]
    recommendations: list[str]

@dataclass
class FailurePattern:
    pattern_type: str  # e.g., "category_cluster", "criterion_systematic"
    description: str
    affected_tests: list[str]
    severity: Literal["high", "medium", "low"]

@dataclass
class Hypothesis:
    id: str
    description: str
    change_type: Literal["node_param", "workflow_structure", "prompt_tuning"]
    target: str  # node_id or "workflow"
    suggested_change: dict
    rationale: str
    expected_impact: str
```

---

## Phase 2: Protocol Interfaces (`src/policyflow/benchmark/protocols.py`)

```python
from typing import Protocol

class BenchmarkRunner(Protocol):
    """Runs workflow against test cases."""
    def run(
        self,
        workflow: Flow,
        test_cases: list[GoldenTestCase]
    ) -> BenchmarkReport: ...

class ResultComparator(Protocol):
    """Compares actual vs expected results."""
    def compare(
        self,
        actual: EvaluationResult,
        expected: ExpectedResult
    ) -> ComparisonResult: ...

class MetricsCalculator(Protocol):
    """Calculates metrics from benchmark results."""
    def calculate(self, results: list[TestCaseResult]) -> BenchmarkMetrics: ...

class FailureAnalyzer(Protocol):
    """Analyzes failure patterns."""
    def analyze(
        self,
        report: BenchmarkReport,
        workflow: ParsedWorkflowPolicy
    ) -> AnalysisReport: ...

class HypothesisGenerator(Protocol):
    """Generates improvement hypotheses."""
    def generate(
        self,
        analysis: AnalysisReport,
        workflow: ParsedWorkflowPolicy
    ) -> list[Hypothesis]: ...

class ExperimentTracker(Protocol):
    """Tracks experiments over time."""
    def record(self, experiment: Experiment) -> None: ...
    def get_history(self) -> list[Experiment]: ...
    def get_best(self) -> Experiment | None: ...

class DatasetGenerator(Protocol):
    """Generates golden dataset from normalized policy."""
    def generate(
        self,
        policy: NormalizedPolicy,
        config: GeneratorConfig
    ) -> GoldenDataset: ...

    def generate_for_criterion(
        self,
        criterion: Clause,
        policy: NormalizedPolicy,
        count: int
    ) -> list[GoldenTestCase]: ...

    def augment(
        self,
        existing: GoldenDataset,
        policy: NormalizedPolicy,
        config: GeneratorConfig
    ) -> GoldenDataset: ...
```

---

## Phase 3: Implementation Components

### 3.1 Golden Dataset Loader (`loader.py`)
- Parse `golden_dataset.yaml` into `list[GoldenTestCase]`
- Validate schema
- Filter by category/ID if needed

### 3.2 Benchmark Runner (`runner.py`)
```python
class SimpleBenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self, workflow: Flow, test_cases: list[GoldenTestCase]) -> BenchmarkReport:
        results = []
        for tc in test_cases:
            start = time.time()
            try:
                actual = workflow.run({"input_text": tc.input_text})
                result = TestCaseResult(
                    test_id=tc.id,
                    expected=tc.expected,
                    actual=actual,
                    error=None,
                    duration_ms=(time.time() - start) * 1000
                )
            except Exception as e:
                result = TestCaseResult(
                    test_id=tc.id,
                    expected=tc.expected,
                    actual=None,
                    error=str(e),
                    duration_ms=(time.time() - start) * 1000
                )
            results.append(result)

        metrics = self.metrics_calculator.calculate(results)
        return BenchmarkReport(...)
```

### 3.3 Result Comparator (`comparator.py`)
- Deep comparison of nested criterion results
- Track which specific sub-criteria differ
- Generate structured diff

### 3.4 Metrics Calculator (`metrics.py`)
- Overall accuracy: `sum(policy_satisfied matches) / total`
- Per-criterion precision/recall/F1
- Confusion matrices
- Category breakdown
- Confidence calibration curves

### 3.5 Failure Analyzer (`analyzer.py`)

**Rule-based patterns:**
```python
class RuleBasedAnalyzer:
    def analyze(self, report: BenchmarkReport, workflow: ParsedWorkflowPolicy) -> AnalysisReport:
        patterns = []

        # Pattern 1: Category clusters
        failures_by_category = group_by(report.failures, lambda f: f.category)
        for cat, failures in failures_by_category.items():
            if len(failures) / total_in_category > 0.5:
                patterns.append(FailurePattern(
                    pattern_type="category_cluster",
                    description=f"High failure rate in {cat} category",
                    affected_tests=[f.test_id for f in failures],
                    severity="high"
                ))

        # Pattern 2: Systematic criterion failures
        criterion_failure_rates = compute_criterion_failure_rates(report)
        for crit, rate in criterion_failure_rates.items():
            if rate > 0.3:
                patterns.append(FailurePattern(
                    pattern_type="criterion_systematic",
                    description=f"Criterion {crit} fails systematically",
                    ...
                ))

        # Pattern 3: False positive vs false negative imbalance
        # Pattern 4: Confidence miscalibration
        # ...

        return AnalysisReport(patterns=patterns, ...)
```

**LLM-enhanced analyzer (optional):**
```python
class LLMEnhancedAnalyzer:
    def analyze(self, report: BenchmarkReport, workflow: ParsedWorkflowPolicy) -> AnalysisReport:
        # First, run rule-based
        base_analysis = self.rule_based.analyze(report, workflow)

        # Then, ask LLM to find deeper patterns
        prompt = f"""
        Analyze these benchmark failures and identify patterns:

        Failures: {format_failures(report.failures)}
        Workflow: {workflow.to_yaml()}
        Rule-based findings: {base_analysis}

        Identify:
        1. Subtle patterns the rules might have missed
        2. Root cause hypotheses
        3. Connections between different failure modes
        """
        llm_analysis = call_llm(prompt)

        return merge_analyses(base_analysis, llm_analysis)
```

### 3.6 Hypothesis Generator (`hypothesis.py`)

**Templates for common fixes:**
```python
HYPOTHESIS_TEMPLATES = {
    "criterion_systematic": [
        Hypothesis(
            change_type="prompt_tuning",
            description="Clarify criterion {criterion} prompt",
            suggested_change={"prompt": "..."}
        ),
        Hypothesis(
            change_type="node_param",
            description="Lower confidence threshold for {criterion}",
            suggested_change={"confidence_threshold": 0.6}
        ),
    ],
    "false_positive_heavy": [
        Hypothesis(
            change_type="workflow_structure",
            description="Add confidence gate after {node}",
            suggested_change={"add_node": {...}}
        ),
    ],
}

class TemplateBasedHypothesisGenerator:
    def generate(self, analysis: AnalysisReport, workflow: ...) -> list[Hypothesis]:
        hypotheses = []
        for pattern in analysis.patterns:
            templates = HYPOTHESIS_TEMPLATES.get(pattern.pattern_type, [])
            for template in templates:
                hypotheses.append(template.instantiate(pattern))
        return hypotheses
```

**LLM-powered generator:**
```python
class LLMHypothesisGenerator:
    def generate(self, analysis: AnalysisReport, workflow: ...) -> list[Hypothesis]:
        prompt = f"""
        Given this analysis of workflow failures:
        {analysis}

        And this workflow definition:
        {workflow.to_yaml()}

        Suggest 3-5 specific, actionable hypotheses for improvement.
        For each hypothesis, specify:
        - What to change (node param, workflow structure, prompt)
        - The exact change
        - Expected impact

        Available node types: {node_registry.list_all()}
        """
        return parse_hypotheses(call_llm(prompt))
```

### 3.7 Golden Dataset Generator (`generator.py`)

Generates comprehensive test cases from a normalized policy, including edge cases and intermediate state expectations.

**Core Concept:**
```
NormalizedPolicy → LLM + Rules → GoldenDataset with expected criterion states
```

**Generation Config:**
```python
@dataclass
class GeneratorConfig:
    # Scope control
    cases_per_criterion: int = 3          # How many test cases per criterion
    include_edge_cases: bool = True       # Generate edge cases
    include_partial_matches: bool = True  # Cases where some criteria pass, others fail

    # Category distribution
    categories: list[str] = field(default_factory=lambda: [
        "clear_pass",      # All criteria met
        "clear_fail",      # Obviously fails
        "edge_case",       # Boundary conditions
        "partial_match",   # Some criteria met
    ])

    # Edge case strategies
    edge_case_strategies: list[str] = field(default_factory=lambda: [
        "boundary",        # At threshold boundaries
        "negation",        # Negative phrasing of positive cases
        "ambiguous",       # Deliberately ambiguous language
        "implicit",        # Implicit rather than explicit signals
        "missing_element", # Missing one key element
    ])

    # Output control
    include_reasoning: bool = True        # Include expected reasoning
    include_intermediate_states: bool = True  # Clause-by-clause expectations

    # LLM settings
    mode: Literal["llm", "template", "hybrid"] = "hybrid"
    temperature: float = 0.7              # Higher for diversity
```

**Generator Protocol:**
```python
class DatasetGenerator(Protocol):
    def generate(
        self,
        policy: NormalizedPolicy,
        config: GeneratorConfig
    ) -> GoldenDataset: ...

class GoldenDataset:
    policy_file: str
    description: str
    test_cases: list[GoldenTestCase]
    generation_metadata: GenerationMetadata

@dataclass
class GenerationMetadata:
    generator_version: str
    config_used: GeneratorConfig
    timestamp: datetime
    policy_hash: str  # For tracking policy changes
```

**Generation Strategy:**

```python
class HybridDatasetGenerator:
    """Combines template rules with LLM generation."""

    def generate(self, policy: NormalizedPolicy, config: GeneratorConfig) -> GoldenDataset:
        test_cases = []

        # 1. Generate "clear pass" cases (all criteria met)
        test_cases.extend(self._generate_clear_passes(policy, config))

        # 2. Generate "clear fail" cases (per criterion)
        for criterion in policy.get_criteria():
            test_cases.extend(self._generate_criterion_fails(criterion, policy, config))

        # 3. Generate partial matches (combinatorial)
        if config.include_partial_matches:
            test_cases.extend(self._generate_partial_matches(policy, config))

        # 4. Generate edge cases
        if config.include_edge_cases:
            for strategy in config.edge_case_strategies:
                test_cases.extend(self._generate_edge_cases(policy, strategy, config))

        # 5. Deduplicate and validate
        test_cases = self._deduplicate(test_cases)
        test_cases = self._validate_expected_states(test_cases, policy)

        return GoldenDataset(test_cases=test_cases, ...)

    def _generate_clear_passes(self, policy, config) -> list[GoldenTestCase]:
        """LLM generates text that satisfies ALL criteria."""
        prompt = f"""
        Generate {config.cases_per_criterion} realistic examples that satisfy
        ALL of the following policy criteria:

        {policy.to_criteria_list()}

        For each example:
        1. Write realistic input text (2-4 sentences)
        2. Explain which elements satisfy each criterion

        Vary the examples: different contexts, writing styles, explicit vs implicit signals.
        """
        return self._parse_llm_response(call_llm(prompt), category="clear_pass")

    def _generate_criterion_fails(self, criterion, policy, config) -> list[GoldenTestCase]:
        """Generate cases that fail THIS criterion but could pass others."""
        prompt = f"""
        Generate {config.cases_per_criterion} realistic examples that:
        - FAIL criterion: {criterion.text}
        - Could plausibly PASS the other criteria

        Other criteria for context:
        {policy.get_other_criteria(criterion)}

        Make the failure clear but realistic (not contrived).
        """
        # Parse and set expected state: this criterion = False
        cases = self._parse_llm_response(call_llm(prompt), category="clear_fail")
        for case in cases:
            case.expected.criterion_results[criterion.id].met = False
        return cases

    def _generate_edge_cases(self, policy, strategy: str, config) -> list[GoldenTestCase]:
        """Generate edge cases using specific strategy."""
        strategy_prompts = {
            "boundary": "Generate cases AT THE BOUNDARY of satisfying criteria...",
            "negation": "Generate cases using negative phrasing that might confuse...",
            "ambiguous": "Generate deliberately ambiguous cases where criteria status is unclear...",
            "implicit": "Generate cases where criteria are satisfied IMPLICITLY rather than explicitly...",
            "missing_element": "Generate cases missing exactly ONE element needed for a criterion...",
        }
        prompt = strategy_prompts[strategy].format(policy=policy)
        return self._parse_llm_response(call_llm(prompt), category=f"edge_case_{strategy}")
```

**Intermediate State Expectations:**

For hierarchical policies like the personal recommendation definition, generate expected states at every level:

```python
@dataclass
class GoldenTestCase:
    id: str
    name: str
    input_text: str
    expected: ExpectedResult
    category: str
    notes: str

    # NEW: Intermediate expectations for debugging
    intermediate_expectations: dict[str, IntermediateState] | None = None

@dataclass
class IntermediateState:
    """Expected state at a specific clause/subclause."""
    clause_id: str
    expected_met: bool
    key_signals: list[str]  # What text signals this?
    reasoning: str
```

Example for the personal recommendation policy:
```yaml
- id: test_edge_001
  name: "Implicit suitability through context"
  input_text: >
    Given your retirement timeline and what we discussed about your
    risk tolerance, the balanced fund makes sense.
  expected:
    policy_satisfied: true
    criterion_results:
      criterion_1: { met: true, ... }
      criterion_2: { met: true, ... }
      criterion_3: { met: true, ... }
      criterion_4: { met: true, ... }
  intermediate_expectations:
    criterion_3a:
      expected_met: false
      key_signals: []
      reasoning: "No explicit 'suitable' language used"
    criterion_3b:
      expected_met: true
      key_signals: ["retirement timeline", "risk tolerance", "what we discussed"]
      reasoning: "Implicit reference to recipient's circumstances"
  category: edge_case_implicit
  notes: "Tests implicit circumstance consideration without explicit suitability statement"
```

**CLI Interface:**
```bash
# Generate from normalized policy
policyflow generate-dataset --policy normalized.yaml --output golden_dataset.yaml

# With configuration
policyflow generate-dataset --policy normalized.yaml \
    --cases-per-criterion 5 \
    --include-edge-cases \
    --strategies boundary,negation,implicit \
    --mode hybrid \
    --output golden_dataset.yaml

# Generate for specific criteria only
policyflow generate-dataset --policy normalized.yaml \
    --criteria criterion_1,criterion_3 \
    --output partial_dataset.yaml

# Augment existing dataset with more edge cases
policyflow generate-dataset --policy normalized.yaml \
    --augment golden_dataset.yaml \
    --categories edge_case \
    --count 10 \
    --output augmented_dataset.yaml
```

**Python API:**
```python
from policyflow.benchmark import (
    DatasetGenerator,
    GeneratorConfig,
    load_normalized_policy
)

# Load policy
policy = load_normalized_policy("normalized.yaml")

# Configure generation
config = GeneratorConfig(
    cases_per_criterion=5,
    include_edge_cases=True,
    edge_case_strategies=["boundary", "implicit", "ambiguous"],
    include_intermediate_states=True,
    mode="hybrid"
)

# Generate
generator = DatasetGenerator(config)
dataset = generator.generate(policy)

# Save
dataset.to_yaml("golden_dataset.yaml")

# Or generate incrementally
for criterion in policy.get_criteria():
    cases = generator.generate_for_criterion(criterion, count=3)
    dataset.add_cases(cases)
```

---

### 3.8 Experiment Tracker (`tracker.py`)
```python
@dataclass
class Experiment:
    id: str
    timestamp: datetime
    workflow_snapshot: str  # YAML
    hypothesis_applied: Hypothesis | None
    benchmark_report: BenchmarkReport
    parent_experiment_id: str | None  # for tracking lineage

class FileBasedExperimentTracker:
    def __init__(self, experiments_dir: Path):
        self.dir = experiments_dir

    def record(self, experiment: Experiment) -> None:
        path = self.dir / f"{experiment.id}.yaml"
        path.write_text(experiment.to_yaml())

    def get_history(self) -> list[Experiment]:
        return [Experiment.from_yaml(p.read_text()) for p in self.dir.glob("*.yaml")]

    def get_best(self) -> Experiment | None:
        history = self.get_history()
        return max(history, key=lambda e: e.benchmark_report.metrics.overall_accuracy)
```

---

## Phase 4: CLI Interface

```bash
# === DATASET GENERATION ===
# Generate golden dataset from normalized policy
policyflow generate-dataset --policy normalized.yaml --output golden_dataset.yaml

# With full configuration
policyflow generate-dataset --policy normalized.yaml \
    --cases-per-criterion 5 \
    --strategies boundary,negation,implicit \
    --include-intermediate-states \
    --mode hybrid \
    --output golden_dataset.yaml

# Generate for specific criteria only
policyflow generate-dataset --policy normalized.yaml \
    --criteria criterion_1,criterion_3 \
    --output partial_dataset.yaml

# Augment existing dataset
policyflow generate-dataset --policy normalized.yaml \
    --augment golden_dataset.yaml \
    --categories edge_case \
    --count 10 \
    --output augmented_dataset.yaml

# === BENCHMARKING ===
# Run benchmark
policyflow benchmark --workflow workflow.yaml --dataset golden_dataset.yaml --output report.yaml

# Analyze failures
policyflow analyze --report report.yaml --workflow workflow.yaml --output analysis.yaml

# Generate hypotheses
policyflow hypothesize --analysis analysis.yaml --workflow workflow.yaml --output hypotheses.yaml

# Full loop (benchmark + analyze + hypothesize)
policyflow improve --workflow workflow.yaml --dataset golden_dataset.yaml

# === EXPERIMENTS ===
# View experiment history
policyflow experiments list
policyflow experiments compare exp_001 exp_002
policyflow experiments best
```

---

## Phase 5: Programmatic API

```python
from policyflow.benchmark import (
    load_golden_dataset,
    SimpleBenchmarkRunner,
    RuleBasedAnalyzer,
    LLMHypothesisGenerator,
    FileBasedExperimentTracker,
)

# Load
dataset = load_golden_dataset("golden_dataset.yaml")
workflow = DynamicWorkflowBuilder.from_yaml("workflow.yaml").build()

# Benchmark
runner = SimpleBenchmarkRunner(BenchmarkConfig())
report = runner.run(workflow, dataset)
print(f"Accuracy: {report.metrics.overall_accuracy:.2%}")

# Analyze
analyzer = RuleBasedAnalyzer()
analysis = analyzer.analyze(report, workflow)
for pattern in analysis.patterns:
    print(f"[{pattern.severity}] {pattern.description}")

# Generate hypotheses
generator = LLMHypothesisGenerator(llm_config)
hypotheses = generator.generate(analysis, workflow)
for h in hypotheses:
    print(f"- {h.description}")

# Track experiment
tracker = FileBasedExperimentTracker(Path("experiments/"))
tracker.record(Experiment(
    id="exp_001",
    workflow_snapshot=workflow.to_yaml(),
    hypothesis_applied=None,
    benchmark_report=report,
))
```

---

## File Structure

```
src/policyflow/benchmark/
├── __init__.py          # Public exports
├── models.py            # Data classes (including OptimizationBudget, OptimizationResult)
├── protocols.py         # Protocol interfaces (including Optimizer, HypothesisApplier)
├── loader.py            # Golden dataset loading
├── generator.py         # Golden dataset GENERATION
├── runner.py            # Benchmark runner
├── comparator.py        # Result comparison
├── metrics.py           # Metrics calculation
├── analyzer.py          # Failure analysis (rule-based + LLM)
├── hypothesis.py        # Hypothesis generation (template + LLM)
├── applier.py           # Hypothesis applier (applies changes to workflows)
├── optimizer.py         # Optimization loop (HillClimbingOptimizer, ConvergenceTester)
├── tracker.py           # Experiment tracking
└── cli.py               # CLI commands (generate-dataset, benchmark, optimize, etc.)
```

---

## Implementation Order

1. **models.py** - Core data structures (GoldenTestCase, BenchmarkReport, Hypothesis, OptimizationBudget, etc.)
2. **protocols.py** - Protocol interfaces (BenchmarkRunner, FailureAnalyzer, DatasetGenerator, Optimizer, HypothesisApplier)
3. **loader.py** - Parse golden_dataset.yaml into typed structures
4. **generator.py** - Golden dataset generation (LLM + template hybrid)
5. **comparator.py** - Deep comparison of actual vs expected results
6. **metrics.py** - Calculate accuracy/precision/recall/F1 per criterion
7. **runner.py** - Execute workflow against test cases, collect results
8. **analyzer.py** - BOTH rule-based AND LLM-powered analyzers (switchable via config)
9. **hypothesis.py** - BOTH template-based AND LLM-powered generators (switchable)
10. **applier.py** - HypothesisApplier to modify workflows based on hypotheses
11. **optimizer.py** - HillClimbingOptimizer + ConvergenceTester
12. **tracker.py** - YAML file-based experiment tracking
13. **cli.py** - Full CLI commands (generate-dataset, benchmark, analyze, optimize, experiments)
14. **__init__.py exports** - Clean public API for programmatic use

---

## Key Files to Modify

- `src/policyflow/__init__.py` - Export benchmark API
- `src/policyflow/cli.py` - Add benchmark commands
- `pyproject.toml` - No new dependencies needed

---

## Automated Optimization Approaches

### Landscape of Optimization Strategies

| Approach | Description | How it would plug in |
|----------|-------------|---------------------|
| **DSPy** | Declarative prompt optimization with signatures, BootstrapFewShot, MIPRO | Wrap workflow nodes as DSPy modules, use metric from benchmark |
| **Opik** | Comet's eval/optimization platform | Export traces, use their optimization loop |
| **Agent Hospital** | Self-evolution through simulated patient interactions | Generate synthetic test cases, self-critique and improve |
| **TextGrad** | Gradient descent on prompts using LLM feedback | Use failure analysis as "gradients" to update prompts |
| **ADAS** | Automated Design of Agentic Systems | Meta-search over workflow structures |
| **Bayesian Optimization** | Parameter search with acquisition functions | Tune thresholds, temperatures, etc. |
| **Evolutionary/Genetic** | Population-based search over configs | Mutate workflows, select fittest |

### Pluggable Optimizer Protocol

```python
class Optimizer(Protocol):
    """Base protocol for all optimizers."""

    def optimize(
        self,
        workflow: ParsedWorkflowPolicy,
        dataset: GoldenDataset,
        budget: OptimizationBudget,
        metric: Callable[[BenchmarkReport], float]
    ) -> OptimizationResult: ...

    def step(
        self,
        workflow: ParsedWorkflowPolicy,
        report: BenchmarkReport
    ) -> ParsedWorkflowPolicy | None:
        """Single optimization step. Returns None if converged."""
        ...

@dataclass
class OptimizationBudget:
    """Controls when optimization stops."""
    max_iterations: int = 10
    max_llm_calls: int = 100
    max_time_seconds: float = 3600
    target_metric: float | None = None  # Stop if metric >= this
    min_improvement: float = 0.01       # Stop if improvement < this
    patience: int = 3                   # Stop after N iterations without improvement

@dataclass
class OptimizationResult:
    best_workflow: ParsedWorkflowPolicy
    best_metric: float
    history: list[OptimizationStep]
    converged: bool
    convergence_reason: str  # "target_reached", "budget_exhausted", "no_improvement"
    total_llm_calls: int
    total_time_seconds: float

@dataclass
class OptimizationStep:
    iteration: int
    workflow_snapshot: str
    metric: float
    changes_made: list[str]
    llm_calls: int
```

### Convergence & Budget Tester

```python
class ConvergenceTester:
    """Tracks optimization progress and determines when to stop."""

    def __init__(self, budget: OptimizationBudget):
        self.budget = budget
        self.history: list[float] = []
        self.iterations = 0
        self.llm_calls = 0
        self.start_time = time.time()
        self.best_metric = float('-inf')
        self.steps_without_improvement = 0

    def record_step(self, metric: float, llm_calls: int) -> None:
        self.history.append(metric)
        self.iterations += 1
        self.llm_calls += llm_calls

        if metric > self.best_metric + self.budget.min_improvement:
            self.best_metric = metric
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

    def should_stop(self) -> tuple[bool, str]:
        """Returns (should_stop, reason)."""
        elapsed = time.time() - self.start_time

        if self.budget.target_metric and self.best_metric >= self.budget.target_metric:
            return True, "target_reached"
        if self.iterations >= self.budget.max_iterations:
            return True, "max_iterations"
        if self.llm_calls >= self.budget.max_llm_calls:
            return True, "max_llm_calls"
        if elapsed >= self.budget.max_time_seconds:
            return True, "timeout"
        if self.steps_without_improvement >= self.budget.patience:
            return True, "no_improvement"

        return False, ""

    def get_summary(self) -> dict:
        return {
            "iterations": self.iterations,
            "llm_calls": self.llm_calls,
            "elapsed_seconds": time.time() - self.start_time,
            "best_metric": self.best_metric,
            "improvement_curve": self.history,
        }
```

### Basic Hill-Climbing Optimizer (POC Implementation)

```python
class HillClimbingOptimizer:
    """Simple optimizer that applies hypotheses one at a time."""

    def __init__(
        self,
        analyzer: FailureAnalyzer,
        hypothesis_generator: HypothesisGenerator,
        hypothesis_applier: HypothesisApplier,  # NEW: applies changes to workflow
    ):
        self.analyzer = analyzer
        self.generator = hypothesis_generator
        self.applier = hypothesis_applier

    def optimize(
        self,
        workflow: ParsedWorkflowPolicy,
        dataset: GoldenDataset,
        budget: OptimizationBudget,
        metric: Callable[[BenchmarkReport], float] = lambda r: r.metrics.overall_accuracy
    ) -> OptimizationResult:

        runner = SimpleBenchmarkRunner()
        tester = ConvergenceTester(budget)
        history = []

        current_workflow = workflow
        best_workflow = workflow
        best_metric = float('-inf')

        while True:
            # 1. Benchmark current workflow
            report = runner.run(current_workflow, dataset)
            current_metric = metric(report)
            tester.record_step(current_metric, report.llm_calls)

            # Track best
            if current_metric > best_metric:
                best_metric = current_metric
                best_workflow = current_workflow

            history.append(OptimizationStep(
                iteration=tester.iterations,
                workflow_snapshot=current_workflow.to_yaml(),
                metric=current_metric,
                changes_made=[],
                llm_calls=report.llm_calls
            ))

            # 2. Check convergence
            should_stop, reason = tester.should_stop()
            if should_stop:
                return OptimizationResult(
                    best_workflow=best_workflow,
                    best_metric=best_metric,
                    history=history,
                    converged=(reason == "target_reached"),
                    convergence_reason=reason,
                    **tester.get_summary()
                )

            # 3. Analyze failures and generate hypotheses
            analysis = self.analyzer.analyze(report, current_workflow)
            hypotheses = self.generator.generate(analysis, current_workflow)

            if not hypotheses:
                return OptimizationResult(
                    best_workflow=best_workflow,
                    best_metric=best_metric,
                    history=history,
                    converged=True,
                    convergence_reason="no_hypotheses",
                    **tester.get_summary()
                )

            # 4. Try top hypothesis (greedy)
            best_hypothesis = hypotheses[0]
            current_workflow = self.applier.apply(current_workflow, best_hypothesis)
            history[-1].changes_made = [best_hypothesis.description]
```

### Hypothesis Applier Protocol

```python
class HypothesisApplier(Protocol):
    """Applies a hypothesis to modify a workflow."""

    def apply(
        self,
        workflow: ParsedWorkflowPolicy,
        hypothesis: Hypothesis
    ) -> ParsedWorkflowPolicy: ...

class BasicHypothesisApplier:
    """Applies structured hypothesis changes to workflows."""

    def apply(self, workflow: ParsedWorkflowPolicy, hypothesis: Hypothesis) -> ParsedWorkflowPolicy:
        workflow = workflow.copy()

        match hypothesis.change_type:
            case "node_param":
                # Update node parameters
                node_id = hypothesis.target
                for key, value in hypothesis.suggested_change.items():
                    workflow.nodes[node_id].params[key] = value

            case "prompt_tuning":
                # Update prompt template
                node_id = hypothesis.target
                workflow.nodes[node_id].params["prompt"] = hypothesis.suggested_change["prompt"]

            case "workflow_structure":
                # Add/remove/rewire nodes
                if "add_node" in hypothesis.suggested_change:
                    self._add_node(workflow, hypothesis.suggested_change["add_node"])
                if "remove_node" in hypothesis.suggested_change:
                    self._remove_node(workflow, hypothesis.suggested_change["remove_node"])
                if "rewire" in hypothesis.suggested_change:
                    self._rewire(workflow, hypothesis.suggested_change["rewire"])

            case "threshold":
                # Update confidence/gate thresholds
                node_id = hypothesis.target
                workflow.nodes[node_id].params["threshold"] = hypothesis.suggested_change["threshold"]

        return workflow
```

### Future Optimizer Implementations (Pluggable)

```python
# DSPy-style few-shot optimization
class DSPyStyleOptimizer:
    """Optimizes by finding best few-shot examples for each node."""

    def optimize(self, workflow, dataset, budget, metric):
        # 1. For each LLM node, collect successful examples
        # 2. Use bootstrap to select best examples
        # 3. Update node prompts with examples
        ...

# Bayesian parameter optimization
class BayesianOptimizer:
    """Uses Bayesian optimization for continuous parameters."""

    def __init__(self, param_space: dict[str, tuple[float, float]]):
        self.param_space = param_space  # e.g., {"confidence_threshold": (0.3, 0.9)}

    def optimize(self, workflow, dataset, budget, metric):
        # 1. Use acquisition function to select next params
        # 2. Apply params to workflow
        # 3. Benchmark and update surrogate model
        ...

# Evolutionary/genetic optimization
class EvolutionaryOptimizer:
    """Population-based optimization over workflow variants."""

    def __init__(self, population_size: int = 10, mutation_rate: float = 0.1):
        ...

    def optimize(self, workflow, dataset, budget, metric):
        # 1. Initialize population with workflow variants
        # 2. Evaluate fitness (benchmark metric)
        # 3. Select, crossover, mutate
        # 4. Repeat until convergence
        ...

# LLM-as-optimizer (Agent Hospital style)
class LLMMetaOptimizer:
    """Uses LLM to propose and evaluate workflow changes."""

    def optimize(self, workflow, dataset, budget, metric):
        # 1. Show LLM the workflow, failures, and metric
        # 2. Ask LLM to propose improvements
        # 3. Apply and evaluate
        # 4. Feed results back to LLM for next iteration
        ...
```

### CLI for Optimization

```bash
# Run basic optimization
policyflow optimize --workflow workflow.yaml --dataset golden_dataset.yaml \
    --max-iterations 10 \
    --target-accuracy 0.95 \
    --output optimized_workflow.yaml

# With budget constraints
policyflow optimize --workflow workflow.yaml --dataset golden_dataset.yaml \
    --max-llm-calls 100 \
    --max-time 3600 \
    --patience 3 \
    --output optimized_workflow.yaml

# Specific optimizer (future)
policyflow optimize --workflow workflow.yaml --dataset golden_dataset.yaml \
    --optimizer bayesian \
    --param-space '{"confidence_threshold": [0.3, 0.9]}' \
    --output optimized_workflow.yaml
```

### Python API for Optimization

```python
from policyflow.benchmark import (
    HillClimbingOptimizer,
    OptimizationBudget,
    create_analyzer,
    create_hypothesis_generator,
    BasicHypothesisApplier,
)

# Configure budget
budget = OptimizationBudget(
    max_iterations=10,
    max_llm_calls=100,
    target_metric=0.95,
    patience=3
)

# Create optimizer
optimizer = HillClimbingOptimizer(
    analyzer=create_analyzer(mode="hybrid"),
    hypothesis_generator=create_hypothesis_generator(mode="hybrid"),
    hypothesis_applier=BasicHypothesisApplier()
)

# Run optimization
result = optimizer.optimize(
    workflow=load_workflow("workflow.yaml"),
    dataset=load_golden_dataset("golden_dataset.yaml"),
    budget=budget,
    metric=lambda r: r.metrics.overall_accuracy
)

# Results
print(f"Converged: {result.converged} ({result.convergence_reason})")
print(f"Best accuracy: {result.best_metric:.2%}")
print(f"Iterations: {len(result.history)}")
print(f"LLM calls: {result.total_llm_calls}")

# Save optimized workflow
result.best_workflow.to_yaml("optimized_workflow.yaml")

# View improvement curve
for step in result.history:
    print(f"  {step.iteration}: {step.metric:.2%} {step.changes_made}")
```

---

## Analyzer Mode Selection

Runtime configuration for switching between analysis modes:

```python
class AnalyzerConfig:
    mode: Literal["rule_based", "llm", "hybrid"] = "hybrid"
    # hybrid = rule-based first, then LLM enhancement

class HypothesisConfig:
    mode: Literal["template", "llm", "hybrid"] = "hybrid"

# CLI usage
policyflow analyze --mode rule_based  # Fast, deterministic
policyflow analyze --mode llm         # Richer insights
policyflow analyze --mode hybrid      # Best of both (default)

# Python API
analyzer = create_analyzer(mode="hybrid")  # Factory function
```

---

## Success Criteria for POC

1. Can run `policyflow generate-dataset` to create test cases from normalized policy
2. Generated datasets include clear_pass, clear_fail, edge_cases, and partial_matches
3. Intermediate state expectations are included for debugging
4. Can run `policyflow benchmark` and get accuracy metrics
5. Can see per-criterion breakdown (precision/recall/F1 per criterion)
6. Can identify which test categories are problematic
7. Can generate at least 3 actionable hypotheses
8. Can run `policyflow optimize` with budget constraints (iterations, LLM calls, time, patience)
9. Optimizer converges or stops appropriately with clear reason
10. Can track experiments and compare runs over time
11. All interfaces are Protocols (swappable implementations)
12. Both rule-based and LLM modes work for generation and analysis
13. CLI and Python API both fully functional
14. Future optimizers (DSPy, Bayesian, Evolutionary) can plug in via Optimizer protocol

---

## Complete Improvement Loop (End-to-End)

```bash
# 0. (Optional) Generate golden dataset from normalized policy
policyflow generate-dataset --policy normalized.yaml \
    --cases-per-criterion 5 \
    --include-intermediate-states \
    --output golden_dataset.yaml

# 1. Baseline benchmark
policyflow benchmark --workflow workflow.yaml --dataset golden_dataset.yaml \
    --output experiments/baseline.yaml

# 2. Analyze failures (hybrid mode)
policyflow analyze --report experiments/baseline.yaml --mode hybrid \
    --output analysis.yaml

# 3. Generate hypotheses
policyflow hypothesize --analysis analysis.yaml --workflow workflow.yaml \
    --output hypotheses.yaml

# 4. Human reviews hypotheses.yaml, picks one to try

# 5. Apply hypothesis (manual edit to workflow.yaml for POC)

# 6. Re-benchmark
policyflow benchmark --workflow workflow_v2.yaml --dataset golden_dataset.yaml \
    --output experiments/exp_001.yaml

# 7. Compare experiments
policyflow experiments compare baseline exp_001

# 8. (Optional) Augment dataset with more edge cases based on failures
policyflow generate-dataset --policy normalized.yaml \
    --augment golden_dataset.yaml \
    --strategies implicit,ambiguous \
    --count 5 \
    --output golden_dataset_v2.yaml

# 9. Repeat until satisfied
```

**Programmatic equivalent:**
```python
from policyflow.benchmark import (
    load_golden_dataset, load_normalized_policy,
    DatasetGenerator, GeneratorConfig,
    SimpleBenchmarkRunner,
    create_analyzer, create_hypothesis_generator,
    FileBasedExperimentTracker
)

# Option A: Load existing dataset
dataset = load_golden_dataset("golden_dataset.yaml")

# Option B: Generate dataset from normalized policy
policy = load_normalized_policy("normalized.yaml")
generator = DatasetGenerator(GeneratorConfig(
    cases_per_criterion=5,
    include_edge_cases=True,
    edge_case_strategies=["boundary", "implicit", "ambiguous"],
    include_intermediate_states=True,
    mode="hybrid"
))
dataset = generator.generate(policy)
dataset.to_yaml("golden_dataset.yaml")

# Load workflow and tracker
workflow = load_workflow("workflow.yaml")
tracker = FileBasedExperimentTracker("experiments/")

# Baseline benchmark
report = SimpleBenchmarkRunner().run(workflow, dataset)
tracker.record(Experiment(id="baseline", report=report, workflow_snapshot=workflow))

# Analyze + Hypothesize
analysis = create_analyzer(mode="hybrid").analyze(report, workflow)
hypotheses = create_hypothesis_generator(mode="hybrid").generate(analysis, workflow)

# Print for human review
for h in hypotheses:
    print(f"[{h.change_type}] {h.description}\n  -> {h.suggested_change}")

# (Optional) Augment dataset based on failure patterns
if analysis.has_systematic_failures():
    augmented = generator.augment(
        existing=dataset,
        policy=policy,
        config=GeneratorConfig(
            categories=["edge_case"],
            edge_case_strategies=analysis.get_weak_strategies(),
            cases_per_criterion=3
        )
    )
    augmented.to_yaml("golden_dataset_augmented.yaml")

# Human applies chosen hypothesis, re-runs...
```
