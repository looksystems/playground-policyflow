"""Benchmark runner for executing workflows against test cases."""

from __future__ import annotations

import time
from datetime import datetime

from pydantic import BaseModel, Field

from policyflow.benchmark.metrics import SimpleMetricsCalculator
from policyflow.benchmark.models import (
    BenchmarkReport,
    GoldenTestCase,
    TestCaseResult,
)
from policyflow.config import WorkflowConfig
from policyflow.models import ClauseResult, EvaluationResult, ParsedWorkflowPolicy
from policyflow.workflow_builder import DynamicWorkflowBuilder


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark runs."""

    workflow_id: str = Field(
        default="default",
        description="Identifier for this workflow version",
    )
    max_iterations: int = Field(
        default=100,
        description="Maximum workflow iterations per test case",
    )
    workflow_config: WorkflowConfig | None = Field(
        default=None,
        description="Workflow configuration for LLM nodes",
    )


def _extract_clause_result(data: dict) -> ClauseResult:
    """Extract a ClauseResult from a dictionary.

    Args:
        data: Dictionary with clause result data

    Returns:
        ClauseResult instance
    """
    sub_results = []
    if "sub_results" in data and data["sub_results"]:
        sub_results = [_extract_clause_result(sr) for sr in data["sub_results"]]

    return ClauseResult(
        clause_id=data.get("clause_id", ""),
        clause_name=data.get("clause_name", ""),
        met=data.get("met", False),
        reasoning=data.get("reasoning", ""),
        confidence=data.get("confidence", 0.5),
        sub_results=sub_results,
    )


def _extract_evaluation_result(shared: dict) -> EvaluationResult:
    """Extract EvaluationResult from workflow shared store.

    Args:
        shared: The shared store after workflow execution

    Returns:
        EvaluationResult extracted from shared data
    """
    clause_results = []
    raw_results = shared.get("clause_results", [])

    for cr in raw_results:
        clause_results.append(_extract_clause_result(cr))

    return EvaluationResult(
        policy_satisfied=shared.get("policy_satisfied", False),
        policy_title=shared.get("policy_title", ""),
        clause_results=clause_results,
        overall_reasoning=shared.get("overall_reasoning", ""),
        overall_confidence=shared.get("overall_confidence", 0.5),
        input_text=shared.get("input_text", ""),
    )


class SimpleBenchmarkRunner:
    """Simple benchmark runner that executes workflows against test cases.

    This runner:
    1. Iterates through test cases
    2. Executes the workflow for each
    3. Collects results and timing
    4. Computes aggregate metrics
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.metrics_calculator = SimpleMetricsCalculator()

    def run(
        self,
        workflow: ParsedWorkflowPolicy,
        test_cases: list[GoldenTestCase],
    ) -> BenchmarkReport:
        """Run benchmark against all test cases.

        Args:
            workflow: The workflow to benchmark
            test_cases: Test cases to run

        Returns:
            Complete benchmark report with results and metrics
        """
        # Build the workflow builder once
        workflow_config = self.config.workflow_config or WorkflowConfig()
        builder = DynamicWorkflowBuilder(workflow, workflow_config)

        # Run each test case
        results: list[TestCaseResult] = []
        for test_case in test_cases:
            result = self._run_single_test(builder, test_case)
            results.append(result)

        # Build category mapping for metrics
        test_categories = {tc.id: tc.category for tc in test_cases}

        # Calculate metrics
        metrics = self.metrics_calculator.calculate(results, test_categories)

        # Create report
        return BenchmarkReport(
            workflow_id=self.config.workflow_id,
            timestamp=datetime.now(),
            results=results,
            metrics=metrics,
            config=self.config.model_dump(),
        )

    def _run_single_test(
        self,
        builder: DynamicWorkflowBuilder,
        test_case: GoldenTestCase,
    ) -> TestCaseResult:
        """Run a single test case through the workflow.

        Args:
            builder: The workflow builder to use
            test_case: The test case to run

        Returns:
            Result of running the test case
        """
        start_time = time.time()

        try:
            # Run workflow
            shared = builder.run(
                test_case.input_text,
                max_iterations=self.config.max_iterations,
            )

            # Extract result from shared store
            actual = _extract_evaluation_result(shared)

            duration_ms = (time.time() - start_time) * 1000

            return TestCaseResult(
                test_id=test_case.id,
                category=test_case.category,
                expected=test_case.expected,
                actual=actual,
                error=None,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            return TestCaseResult(
                test_id=test_case.id,
                category=test_case.category,
                expected=test_case.expected,
                actual=None,
                error=str(e),
                duration_ms=duration_ms,
            )
