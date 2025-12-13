"""Failure analyzer for benchmark results.

Provides both rule-based and LLM-enhanced analysis of failure patterns.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from policyflow.benchmark.models import (
    AnalysisReport,
    BenchmarkReport,
    FailurePattern,
    ProblematicCriterion,
    TestCaseResult,
)
from policyflow.models import ParsedWorkflowPolicy


class RuleBasedAnalyzer:
    """Rule-based failure pattern analyzer.

    Identifies common failure patterns using statistical rules:
    - Category clusters (high failure rate in specific categories)
    - Systematic criterion failures
    - False positive/negative imbalances
    - Confidence miscalibration
    """

    def __init__(
        self,
        category_failure_threshold: float = 0.5,
        criterion_failure_threshold: float = 0.3,
        fp_imbalance_threshold: float = 0.35,
        fn_imbalance_threshold: float = 0.35,
        confidence_miscalibration_threshold: float = 0.3,
    ):
        """Initialize the analyzer with configurable thresholds.

        Args:
            category_failure_threshold: Failure rate above which a category is flagged
            criterion_failure_threshold: Failure rate above which a criterion is flagged
            fp_imbalance_threshold: False positive rate above which to flag
            fn_imbalance_threshold: False negative rate above which to flag
            confidence_miscalibration_threshold: Diff between confidence and accuracy to flag
        """
        self.category_threshold = category_failure_threshold
        self.criterion_threshold = criterion_failure_threshold
        self.fp_threshold = fp_imbalance_threshold
        self.fn_threshold = fn_imbalance_threshold
        self.miscalibration_threshold = confidence_miscalibration_threshold

    def analyze(
        self,
        report: BenchmarkReport,
        workflow: ParsedWorkflowPolicy,
    ) -> AnalysisReport:
        """Analyze benchmark failures to identify patterns.

        Args:
            report: The benchmark report to analyze
            workflow: The workflow that was benchmarked

        Returns:
            Analysis report with patterns and recommendations
        """
        patterns: list[FailurePattern] = []
        problematic_criteria: list[ProblematicCriterion] = []
        recommendations: list[str] = []

        # Pattern 1: Category clusters
        category_patterns = self._detect_category_clusters(report)
        patterns.extend(category_patterns)

        # Pattern 2: Systematic criterion failures
        criterion_patterns, prob_criteria = self._detect_criterion_failures(report)
        patterns.extend(criterion_patterns)
        problematic_criteria.extend(prob_criteria)

        # Pattern 3: False positive/negative imbalances
        imbalance_patterns = self._detect_fp_fn_imbalances(report)
        patterns.extend(imbalance_patterns)

        # Pattern 4: Confidence miscalibration
        miscal_patterns = self._detect_confidence_miscalibration(report)
        patterns.extend(miscal_patterns)

        # Generate recommendations based on patterns
        recommendations = self._generate_recommendations(patterns, problematic_criteria)

        return AnalysisReport(
            patterns=patterns,
            problematic_criteria=problematic_criteria,
            recommendations=recommendations,
        )

    def _detect_category_clusters(
        self, report: BenchmarkReport
    ) -> list[FailurePattern]:
        """Detect categories with high failure rates."""
        patterns = []

        for category, accuracy in report.metrics.category_accuracy.items():
            failure_rate = 1.0 - accuracy
            if failure_rate >= self.category_threshold:
                # Find affected tests
                affected = [
                    r.test_id for r in report.results
                    if not r.passed and self._get_category(r) == category
                ]

                severity = self._rate_to_severity(failure_rate)
                patterns.append(
                    FailurePattern(
                        pattern_type="category_cluster",
                        description=f"High failure rate ({failure_rate:.0%}) in '{category}' category",
                        affected_tests=affected,
                        severity=severity,
                        metadata={"category": category},  # Store structured data
                    )
                )

        return patterns

    def _detect_criterion_failures(
        self, report: BenchmarkReport
    ) -> tuple[list[FailurePattern], list[ProblematicCriterion]]:
        """Detect criteria with systematic failures."""
        patterns = []
        problematic = []

        for crit_id, metrics in report.metrics.criterion_metrics.items():
            failure_rate = 1.0 - metrics.accuracy

            if failure_rate >= self.criterion_threshold:
                # Calculate FP/FN rates
                confusion = metrics.confusion
                total = confusion.total
                fp_rate = confusion.fp / total if total > 0 else 0.0
                fn_rate = confusion.fn / total if total > 0 else 0.0

                problematic.append(
                    ProblematicCriterion(
                        criterion_id=crit_id,
                        failure_rate=failure_rate,
                        false_positive_rate=fp_rate,
                        false_negative_rate=fn_rate,
                        common_failure_patterns=[],
                    )
                )

                severity = self._rate_to_severity(failure_rate)
                patterns.append(
                    FailurePattern(
                        pattern_type="criterion_systematic",
                        description=f"Criterion '{crit_id}' fails systematically ({failure_rate:.0%} failure rate)",
                        affected_tests=[],  # Would need to track per-criterion failures
                        severity=severity,
                        metadata={"criterion": crit_id},  # Store structured data
                    )
                )

        return patterns, problematic

    def _detect_fp_fn_imbalances(
        self, report: BenchmarkReport
    ) -> list[FailurePattern]:
        """Detect false positive or false negative imbalances."""
        patterns = []

        for crit_id, metrics in report.metrics.criterion_metrics.items():
            confusion = metrics.confusion
            total = confusion.total
            if total == 0:
                continue

            fp_rate = confusion.fp / total
            fn_rate = confusion.fn / total

            if fp_rate >= self.fp_threshold:
                patterns.append(
                    FailurePattern(
                        pattern_type="false_positive_heavy",
                        description=f"Criterion '{crit_id}' has high false positive rate ({fp_rate:.0%})",
                        affected_tests=[],
                        severity="medium" if fp_rate < 0.5 else "high",
                        metadata={"criterion": crit_id},  # Store structured data
                    )
                )

            if fn_rate >= self.fn_threshold:
                patterns.append(
                    FailurePattern(
                        pattern_type="false_negative_heavy",
                        description=f"Criterion '{crit_id}' has high false negative rate ({fn_rate:.0%})",
                        affected_tests=[],
                        severity="medium" if fn_rate < 0.5 else "high",
                        metadata={"criterion": crit_id},  # Store structured data
                    )
                )

        return patterns

    def _detect_confidence_miscalibration(
        self, report: BenchmarkReport
    ) -> list[FailurePattern]:
        """Detect confidence scores that don't match actual accuracy."""
        patterns = []
        cal = report.metrics.confidence_calibration

        # High confidence should mean high accuracy
        # If high_conf_accuracy is low, that's a problem
        if cal.high_confidence_accuracy < (0.8 - self.miscalibration_threshold):
            patterns.append(
                FailurePattern(
                    pattern_type="confidence_miscalibration",
                    description=(
                        f"High confidence predictions have low accuracy "
                        f"({cal.high_confidence_accuracy:.0%}). "
                        "Confidence scores are not well calibrated."
                    ),
                    affected_tests=[],
                    severity="high",
                )
            )

        # Low confidence should have lower accuracy - if it's high, might be too conservative
        if cal.low_confidence_accuracy > 0.8:
            patterns.append(
                FailurePattern(
                    pattern_type="confidence_miscalibration",
                    description=(
                        f"Low confidence predictions have high accuracy "
                        f"({cal.low_confidence_accuracy:.0%}). "
                        "Model may be under-confident."
                    ),
                    affected_tests=[],
                    severity="low",
                )
            )

        return patterns

    def _generate_recommendations(
        self,
        patterns: list[FailurePattern],
        problematic_criteria: list[ProblematicCriterion],
    ) -> list[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []

        for pattern in patterns:
            if pattern.pattern_type == "category_cluster":
                # Use metadata if available, fallback to parsing description
                category = pattern.metadata.get("category")
                if not category:
                    # Fallback: try to extract from description (backward compat)
                    parts = pattern.description.split("'")
                    category = parts[1] if len(parts) > 1 else "unknown"
                recommendations.append(
                    f"Review test cases in '{category}' category for common characteristics"
                )

            elif pattern.pattern_type == "criterion_systematic":
                # Use metadata if available, fallback to parsing description
                crit = pattern.metadata.get("criterion")
                if not crit:
                    # Fallback: try to extract from description (backward compat)
                    parts = pattern.description.split("'")
                    crit = parts[1] if len(parts) > 1 else "unknown"
                recommendations.append(
                    f"Consider refining the prompt or logic for criterion '{crit}'"
                )

            elif pattern.pattern_type == "false_positive_heavy":
                recommendations.append(
                    "Add more negative examples or increase specificity thresholds"
                )

            elif pattern.pattern_type == "false_negative_heavy":
                recommendations.append(
                    "Broaden matching criteria or add more positive examples"
                )

            elif pattern.pattern_type == "confidence_miscalibration":
                recommendations.append(
                    "Adjust confidence thresholds or add confidence calibration node"
                )

        return list(set(recommendations))  # Remove duplicates

    def _get_category(self, result: TestCaseResult) -> str:
        """Get category for a test result."""
        return result.category

    def _rate_to_severity(self, rate: float) -> Literal["high", "medium", "low"]:
        """Convert a failure rate to severity level."""
        if rate >= 0.7:
            return "high"
        elif rate >= 0.4:
            return "medium"
        else:
            return "low"


class LLMEnhancedAnalyzer:
    """LLM-enhanced failure analyzer.

    First runs rule-based analysis, then uses LLM to find deeper patterns.
    """

    def __init__(
        self,
        model: str | None = None,
        rule_based: RuleBasedAnalyzer | None = None,
    ):
        """Initialize with optional LLM model and rule-based analyzer.

        Args:
            model: LLM model identifier for enhancement
            rule_based: Rule-based analyzer to use as foundation
        """
        self.model = model
        self.rule_based = rule_based or RuleBasedAnalyzer()

    def analyze(
        self,
        report: BenchmarkReport,
        workflow: ParsedWorkflowPolicy,
    ) -> AnalysisReport:
        """Analyze using rules first, then optionally enhance with LLM.

        If no model is configured, returns only rule-based analysis.
        """
        # Start with rule-based analysis
        base_analysis = self.rule_based.analyze(report, workflow)

        # If no model configured, return rule-based analysis
        if not self.model:
            return base_analysis

        try:
            return self._enhance_with_llm(base_analysis, report, workflow)
        except Exception:
            # Fall back to rule-based on LLM errors
            return base_analysis

    def _enhance_with_llm(
        self,
        base_analysis: AnalysisReport,
        report: BenchmarkReport,
        workflow: ParsedWorkflowPolicy,
    ) -> AnalysisReport:
        """Enhance rule-based analysis with LLM insights."""
        from policyflow.llm import call_llm

        # Build summary of failures
        failed_tests = [r for r in report.results if not r.passed]
        failure_summary = "\n".join(
            f"- {r.test_id}: expected={r.expected_satisfied}, actual={r.actual_satisfied}"
            for r in failed_tests[:20]  # Limit to first 20 for context
        )

        # Build existing patterns summary
        patterns_summary = "\n".join(
            f"- {p.pattern_type}: {p.description}"
            for p in base_analysis.patterns
        )

        prompt = f"""Analyze these benchmark failures to identify additional patterns and insights.

WORKFLOW: {workflow.title}
OVERALL ACCURACY: {report.metrics.overall_accuracy:.1%}

EXISTING PATTERNS (from rule-based analysis):
{patterns_summary or "None detected"}

SAMPLE FAILURES (first 20):
{failure_summary or "None"}

CRITERION METRICS:
{chr(10).join(f'- {cid}: accuracy={m.accuracy:.1%}' for cid, m in list(report.metrics.criterion_metrics.items())[:10])}

Identify any additional patterns not captured by the rule-based analysis.
Focus on:
1. Correlations between criteria failures
2. Input characteristics that predict failure
3. Potential root causes of systematic errors

Respond in YAML format:
```yaml
additional_patterns:
  - pattern_type: category_cluster|criterion_systematic|false_positive_heavy|false_negative_heavy|confidence_miscalibration|correlation
    description: Human-readable description
    severity: high|medium|low
    metadata:
      criterion: optional_criterion_id
      category: optional_category
additional_recommendations:
  - Actionable recommendation text
insights:
  - Key insight about failure patterns
```"""

        system_prompt = """You are an expert at analyzing ML/AI system failures.
Focus on actionable patterns that can guide improvement efforts.
Be specific about which criteria or categories are affected."""

        result = call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.model,
            yaml_response=True,
            span_name="failure_analysis",
        )

        return self._merge_analyses(base_analysis, result)

    def _merge_analyses(
        self, base: AnalysisReport, llm_result: dict
    ) -> AnalysisReport:
        """Merge LLM analysis results with base rule-based analysis."""
        additional_patterns = []
        additional_recommendations = []

        if isinstance(llm_result, dict):
            # Parse additional patterns
            for p in llm_result.get("additional_patterns", []):
                try:
                    additional_patterns.append(
                        FailurePattern(
                            pattern_type=p.get("pattern_type", "llm_detected"),
                            description=p.get("description", "LLM-detected pattern"),
                            affected_tests=[],
                            severity=p.get("severity", "medium"),
                            metadata=p.get("metadata", {}),
                        )
                    )
                except Exception:
                    continue

            # Parse additional recommendations
            additional_recommendations = llm_result.get("additional_recommendations", [])
            if not isinstance(additional_recommendations, list):
                additional_recommendations = []

        return AnalysisReport(
            patterns=base.patterns + additional_patterns,
            problematic_criteria=base.problematic_criteria,
            recommendations=list(set(base.recommendations + additional_recommendations)),
        )


def create_analyzer(
    mode: Literal["rule_based", "llm", "hybrid"] = "hybrid",
    model: str | None = None,
) -> RuleBasedAnalyzer | LLMEnhancedAnalyzer:
    """Factory function to create an analyzer.

    Args:
        mode: Analysis mode
            - "rule_based": Only use statistical rules
            - "llm": Use LLM for all analysis (requires model)
            - "hybrid": Rule-based first, then LLM enhancement
        model: LLM model identifier for "llm" and "hybrid" modes

    Returns:
        Configured analyzer instance

    Raises:
        ValueError: If mode is invalid
    """
    if mode == "rule_based":
        return RuleBasedAnalyzer()
    elif mode in ("llm", "hybrid"):
        return LLMEnhancedAnalyzer(model=model)
    else:
        raise ValueError(f"Invalid analyzer mode: {mode}")
