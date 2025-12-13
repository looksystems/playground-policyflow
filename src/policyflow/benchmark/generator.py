"""Golden dataset generator for benchmarking.

Generates comprehensive test cases from a normalized policy, including
clear passes, clear fails, edge cases, and partial matches.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from policyflow.benchmark.models import (
    CriterionExpectation,
    ExpectedResult,
    GenerationMetadata,
    GeneratorConfig,
    GoldenDataset,
    GoldenTestCase,
    IntermediateState,
)
from policyflow.models import Clause, NormalizedPolicy


# Template patterns for generating test cases
PASS_TEMPLATES = [
    "This content fully satisfies the requirement: {criterion}. It demonstrates compliance in every aspect.",
    "The following clearly meets {criterion}. All conditions are properly addressed.",
    "Here is an example that passes {criterion} without any issues.",
]

FAIL_TEMPLATES = [
    "This content does NOT meet {criterion}. It violates the core requirement.",
    "The following fails to satisfy {criterion}. The key condition is missing.",
    "Here is an example that clearly fails {criterion}.",
]

EDGE_CASE_TEMPLATES = {
    "boundary": [
        "This is exactly at the boundary of {criterion}. It could go either way.",
        "The content is borderline for {criterion}. The requirement is barely met.",
    ],
    "negation": [
        "This content does not NOT satisfy {criterion}. The double negative creates ambiguity.",
        "I wouldn't say it doesn't meet {criterion}. The phrasing is confusing.",
    ],
    "ambiguous": [
        "The content might or might not satisfy {criterion}. It's unclear.",
        "Depending on interpretation, {criterion} may or may not be met.",
    ],
    "implicit": [
        "While not explicitly stated, {criterion} is implicitly satisfied through context.",
        "The requirement {criterion} is met indirectly through related actions.",
    ],
    "missing_element": [
        "Most of {criterion} is satisfied, but one key element is missing.",
        "Nearly complete for {criterion}, except for one requirement.",
    ],
}


class TemplateBasedGenerator:
    """Generates golden datasets using predefined templates.

    This generator uses template patterns to create test cases without
    requiring LLM calls. It's fast and deterministic, suitable for
    initial dataset creation and testing.
    """

    def __init__(self):
        """Initialize the template-based generator."""
        self.pass_templates = PASS_TEMPLATES
        self.fail_templates = FAIL_TEMPLATES
        self.edge_templates = EDGE_CASE_TEMPLATES

    def generate(
        self,
        policy: NormalizedPolicy,
        config: GeneratorConfig,
    ) -> GoldenDataset:
        """Generate a complete golden dataset.

        Args:
            policy: The normalized policy to generate tests for
            config: Generation configuration

        Returns:
            Complete golden dataset
        """
        test_cases: list[GoldenTestCase] = []

        # Get all criteria from the policy
        criteria = policy.get_all_clauses()

        # 1. Generate clear pass cases
        if "clear_pass" in config.categories:
            test_cases.extend(self._generate_clear_passes(policy, config))

        # 2. Generate clear fail cases (per criterion)
        if "clear_fail" in config.categories:
            for criterion in criteria:
                test_cases.extend(
                    self._generate_criterion_fails(criterion, policy, config)
                )

        # 3. Generate partial matches
        if config.include_partial_matches and "partial_match" in config.categories:
            test_cases.extend(self._generate_partial_matches(policy, config))

        # 4. Generate edge cases
        if config.include_edge_cases and "edge_case" in config.categories:
            for strategy in config.edge_case_strategies:
                test_cases.extend(
                    self._generate_edge_cases(policy, strategy, config)
                )

        # Deduplicate by ID
        seen_ids: set[str] = set()
        unique_cases: list[GoldenTestCase] = []
        for tc in test_cases:
            if tc.id not in seen_ids:
                unique_cases.append(tc)
                seen_ids.add(tc.id)

        return GoldenDataset(
            policy_file=policy.title,
            description=f"Generated dataset for {policy.title}",
            test_cases=unique_cases,
            generation_metadata=GenerationMetadata(
                generator_version="1.0.0",
                config_used=config,
                timestamp=datetime.now(),
                policy_hash=self._compute_policy_hash(policy),
            ),
        )

    def generate_for_criterion(
        self,
        criterion: Clause,
        policy: NormalizedPolicy,
        count: int,
    ) -> list[GoldenTestCase]:
        """Generate test cases for a specific criterion.

        Args:
            criterion: The criterion to generate tests for
            policy: The full policy for context
            count: Number of test cases to generate

        Returns:
            List of generated test cases
        """
        cases: list[GoldenTestCase] = []
        all_criteria = policy.get_all_clauses()

        # Generate pass cases
        for i in range(count // 2 + count % 2):
            template = self.pass_templates[i % len(self.pass_templates)]
            input_text = template.format(criterion=criterion.text)

            cases.append(
                GoldenTestCase(
                    id=self._generate_id("pass", criterion.number, i),
                    name=f"Pass case for criterion {criterion.number}",
                    input_text=input_text,
                    expected=self._build_expected_result(
                        policy_satisfied=True,
                        criteria=all_criteria,
                        failing_criteria=[],
                    ),
                    category="clear_pass",
                    notes=f"Generated pass case for criterion {criterion.number}",
                )
            )

        # Generate fail cases
        for i in range(count // 2):
            template = self.fail_templates[i % len(self.fail_templates)]
            input_text = template.format(criterion=criterion.text)

            cases.append(
                GoldenTestCase(
                    id=self._generate_id("fail", criterion.number, i),
                    name=f"Fail case for criterion {criterion.number}",
                    input_text=input_text,
                    expected=self._build_expected_result(
                        policy_satisfied=False,
                        criteria=all_criteria,
                        failing_criteria=[criterion],
                    ),
                    category="clear_fail",
                    notes=f"Generated fail case for criterion {criterion.number}",
                )
            )

        return cases

    def augment(
        self,
        existing: GoldenDataset,
        policy: NormalizedPolicy,
        config: GeneratorConfig,
    ) -> GoldenDataset:
        """Augment an existing dataset with more test cases.

        Args:
            existing: The existing dataset to augment
            policy: The normalized policy
            config: Generation configuration

        Returns:
            Augmented dataset
        """
        # Generate new cases
        new_dataset = self.generate(policy, config)

        # Combine with existing, avoiding duplicates
        existing_ids = {tc.id for tc in existing.test_cases}
        new_cases = [
            tc for tc in new_dataset.test_cases if tc.id not in existing_ids
        ]

        return GoldenDataset(
            policy_file=existing.policy_file,
            description=f"{existing.description} (augmented)",
            test_cases=existing.test_cases + new_cases,
            generation_metadata=GenerationMetadata(
                generator_version="1.0.0",
                config_used=config,
                timestamp=datetime.now(),
                policy_hash=self._compute_policy_hash(policy),
            ),
        )

    def _generate_clear_passes(
        self, policy: NormalizedPolicy, config: GeneratorConfig
    ) -> list[GoldenTestCase]:
        """Generate clear pass test cases."""
        cases: list[GoldenTestCase] = []
        criteria = policy.get_all_clauses()

        for i in range(config.cases_per_criterion):
            template = self.pass_templates[i % len(self.pass_templates)]

            # Combine all criteria into input
            criteria_text = "; ".join(c.text for c in criteria[:3])  # Top 3
            input_text = template.format(criterion=criteria_text)

            cases.append(
                GoldenTestCase(
                    id=self._generate_id("clear_pass", "all", i),
                    name=f"Clear pass case {i + 1}",
                    input_text=input_text,
                    expected=self._build_expected_result(
                        policy_satisfied=True,
                        criteria=criteria,
                        failing_criteria=[],
                    ),
                    category="clear_pass",
                    notes="All criteria satisfied",
                )
            )

        return cases

    def _generate_criterion_fails(
        self,
        criterion: Clause,
        policy: NormalizedPolicy,
        config: GeneratorConfig,
    ) -> list[GoldenTestCase]:
        """Generate clear fail cases for a specific criterion."""
        cases: list[GoldenTestCase] = []
        all_criteria = policy.get_all_clauses()

        for i in range(config.cases_per_criterion):
            template = self.fail_templates[i % len(self.fail_templates)]
            input_text = template.format(criterion=criterion.text)

            cases.append(
                GoldenTestCase(
                    id=self._generate_id("clear_fail", criterion.number, i),
                    name=f"Fail case for criterion {criterion.number} #{i + 1}",
                    input_text=input_text,
                    expected=self._build_expected_result(
                        policy_satisfied=False,
                        criteria=all_criteria,
                        failing_criteria=[criterion],
                    ),
                    category="clear_fail",
                    notes=f"Criterion {criterion.number} fails",
                )
            )

        return cases

    def _generate_partial_matches(
        self, policy: NormalizedPolicy, config: GeneratorConfig
    ) -> list[GoldenTestCase]:
        """Generate partial match cases where some criteria pass, others fail."""
        cases: list[GoldenTestCase] = []
        criteria = policy.get_all_clauses()

        if len(criteria) < 2:
            return cases

        # Generate cases where first criterion passes but second fails
        for i in range(config.cases_per_criterion):
            passing = criteria[0]
            failing = criteria[1] if len(criteria) > 1 else criteria[0]

            input_text = (
                f"This content satisfies {passing.text} but does NOT satisfy {failing.text}."
            )

            cases.append(
                GoldenTestCase(
                    id=self._generate_id("partial", f"{passing.number}_{failing.number}", i),
                    name=f"Partial match: {passing.number} passes, {failing.number} fails",
                    input_text=input_text,
                    expected=self._build_expected_result(
                        policy_satisfied=False,  # Overall fails if any criterion fails
                        criteria=criteria,
                        failing_criteria=[failing],
                    ),
                    category="partial_match",
                    notes=f"Criterion {passing.number} passes, {failing.number} fails",
                )
            )

        return cases

    def _generate_edge_cases(
        self,
        policy: NormalizedPolicy,
        strategy: str,
        config: GeneratorConfig,
    ) -> list[GoldenTestCase]:
        """Generate edge cases using a specific strategy."""
        cases: list[GoldenTestCase] = []
        criteria = policy.get_all_clauses()

        if strategy not in self.edge_templates:
            return cases

        templates = self.edge_templates[strategy]

        for i, criterion in enumerate(criteria[:config.cases_per_criterion]):
            template = templates[i % len(templates)]
            input_text = template.format(criterion=criterion.text)

            # Edge cases have uncertain expected results - typically we assume pass
            # but mark them for manual review
            cases.append(
                GoldenTestCase(
                    id=self._generate_id(f"edge_{strategy}", criterion.number, i),
                    name=f"Edge case ({strategy}) for {criterion.number}",
                    input_text=input_text,
                    expected=self._build_expected_result(
                        policy_satisfied=True,  # Assume pass for edge cases
                        criteria=criteria,
                        failing_criteria=[],
                    ),
                    category=f"edge_case_{strategy}",
                    notes=f"Edge case using {strategy} strategy",
                    intermediate_expectations=self._build_intermediate_expectations(
                        criterion, strategy
                    ) if config.include_intermediate_states else None,
                )
            )

        return cases

    def _build_expected_result(
        self,
        policy_satisfied: bool,
        criteria: list[Clause],
        failing_criteria: list[Clause],
    ) -> ExpectedResult:
        """Build an ExpectedResult with criterion expectations."""
        failing_numbers = {c.number for c in failing_criteria}

        criterion_results: dict[str, CriterionExpectation] = {}
        for criterion in criteria:
            met = criterion.number not in failing_numbers

            # Handle sub-clauses
            sub_results = None
            if criterion.sub_clauses:
                sub_results = {
                    sc.number: CriterionExpectation(
                        met=met,  # Inherit from parent for simplicity
                    )
                    for sc in criterion.sub_clauses
                }

            criterion_results[criterion.number] = CriterionExpectation(
                met=met,
                sub_results=sub_results,
            )

        return ExpectedResult(
            policy_satisfied=policy_satisfied,
            criterion_results=criterion_results,
        )

    def _build_intermediate_expectations(
        self, criterion: Clause, strategy: str
    ) -> dict[str, IntermediateState]:
        """Build intermediate state expectations for debugging."""
        expectations: dict[str, IntermediateState] = {}

        expectations[criterion.number] = IntermediateState(
            clause_id=criterion.number,
            expected_met=True,  # Assume met for edge cases
            key_signals=[criterion.text[:50]],  # First 50 chars as signal
            reasoning=f"Edge case ({strategy}): may require manual review",
        )

        return expectations

    def _generate_id(self, category: str, criterion: str, index: int) -> str:
        """Generate a unique, deterministic test case ID.

        Uses a hash-based approach to ensure the same inputs always produce
        the same ID, enabling reproducibility and proper duplicate detection.
        """
        # Create deterministic hash from inputs
        content = f"{category}:{criterion}:{index}"
        unique = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"test_{category}_{criterion}_{index}_{unique}"

    def _compute_policy_hash(self, policy: NormalizedPolicy) -> str:
        """Compute a hash of the policy for change tracking."""
        content = f"{policy.title}:{policy.description}:{policy.raw_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class HybridDatasetGenerator:
    """Hybrid generator combining templates with LLM generation.

    Uses template-based generation as the foundation, with optional
    LLM enhancement for more sophisticated test cases.
    """

    def __init__(
        self,
        model: str | None = None,
        template_based: TemplateBasedGenerator | None = None,
    ):
        """Initialize with optional LLM model and template-based generator.

        Args:
            model: LLM model identifier for enhancement
            template_based: Template generator for base cases
        """
        self.model = model
        self.template_based = template_based or TemplateBasedGenerator()

    def generate(
        self,
        policy: NormalizedPolicy,
        config: GeneratorConfig,
    ) -> GoldenDataset:
        """Generate a complete golden dataset.

        Uses template-based generation as foundation, optionally enhanced
        with LLM-generated test cases for richer coverage.

        Args:
            policy: The normalized policy to generate tests for
            config: Generation configuration

        Returns:
            Complete golden dataset
        """
        # Start with template-based generation
        dataset = self.template_based.generate(policy, config)

        # Enhance with LLM if model is configured
        if self.model:
            try:
                dataset = self._enhance_with_llm(dataset, policy, config)
            except Exception:
                # Fall back to template-only on LLM errors
                pass

        return dataset

    def _enhance_with_llm(
        self,
        dataset: GoldenDataset,
        policy: NormalizedPolicy,
        config: GeneratorConfig,
    ) -> GoldenDataset:
        """Enhance dataset with LLM-generated test cases."""
        from policyflow.llm import call_llm

        criteria = policy.get_all_clauses()
        criteria_text = "\n".join(
            f"- {c.number}: {c.text}" for c in criteria[:5]  # First 5 criteria
        )

        prompt = f"""Generate challenging test cases for a policy evaluation system.

POLICY: {policy.title}
{policy.description or ""}

CRITERIA:
{criteria_text}

Generate 3 realistic, challenging test cases that would be difficult to evaluate correctly.
Each test case should:
1. Be realistic content that might actually be submitted
2. Test edge cases or ambiguous scenarios
3. Have clear expected results based on the criteria

Respond in YAML format:
```yaml
test_cases:
  - name: Descriptive name
    input_text: The actual content to evaluate
    expected_policy_satisfied: true or false
    failing_criteria: [list of criterion numbers that should fail, or empty]
    category: edge_case_llm
    notes: Why this case is challenging
```"""

        system_prompt = """You are an expert at generating test cases for policy evaluation systems.
Focus on realistic edge cases that test the boundaries of the criteria.
Be specific and provide concrete examples."""

        result = call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.model,
            yaml_response=True,
            span_name="dataset_generation",
        )

        return self._merge_datasets(dataset, result, policy, config)

    def _merge_datasets(
        self,
        dataset: GoldenDataset,
        llm_result: dict,
        policy: NormalizedPolicy,
        config: GeneratorConfig,
    ) -> GoldenDataset:
        """Merge LLM-generated test cases with existing dataset."""
        new_cases = []
        criteria = policy.get_all_clauses()

        if not isinstance(llm_result, dict):
            return dataset

        for i, tc in enumerate(llm_result.get("test_cases", [])):
            try:
                # Build criterion expectations
                failing = set(tc.get("failing_criteria", []))
                criterion_results = {}
                for criterion in criteria:
                    criterion_results[criterion.number] = CriterionExpectation(
                        met=criterion.number not in failing
                    )

                case_id = self.template_based._generate_id(
                    "llm_edge", "generated", len(dataset.test_cases) + i
                )

                new_cases.append(
                    GoldenTestCase(
                        id=case_id,
                        name=tc.get("name", f"LLM-generated case {i+1}"),
                        input_text=tc.get("input_text", ""),
                        expected=ExpectedResult(
                            policy_satisfied=tc.get("expected_policy_satisfied", True),
                            criterion_results=criterion_results,
                        ),
                        category=tc.get("category", "edge_case_llm"),
                        notes=tc.get("notes", "Generated by LLM"),
                    )
                )
            except Exception:
                continue

        # Combine datasets
        return GoldenDataset(
            policy_file=dataset.policy_file,
            description=f"{dataset.description} (LLM-enhanced)",
            test_cases=dataset.test_cases + new_cases,
            generation_metadata=dataset.generation_metadata,
        )

    def generate_for_criterion(
        self,
        criterion: Clause,
        policy: NormalizedPolicy,
        count: int,
    ) -> list[GoldenTestCase]:
        """Generate test cases for a specific criterion."""
        return self.template_based.generate_for_criterion(criterion, policy, count)

    def augment(
        self,
        existing: GoldenDataset,
        policy: NormalizedPolicy,
        config: GeneratorConfig,
    ) -> GoldenDataset:
        """Augment an existing dataset with more test cases."""
        return self.template_based.augment(existing, policy, config)


def create_generator(
    mode: Literal["template", "llm", "hybrid"] = "hybrid",
    model: str | None = None,
) -> TemplateBasedGenerator | HybridDatasetGenerator:
    """Factory function to create a dataset generator.

    Args:
        mode: Generation mode
            - "template": Only use predefined templates
            - "llm": Use LLM for generation (requires model)
            - "hybrid": Combine template and LLM approaches
        model: LLM model identifier for "llm" and "hybrid" modes

    Returns:
        Configured generator instance

    Raises:
        ValueError: If mode is invalid
    """
    if mode == "template":
        return TemplateBasedGenerator()
    elif mode in ("llm", "hybrid"):
        return HybridDatasetGenerator(model=model)
    else:
        raise ValueError(f"Invalid generator mode: {mode}")
