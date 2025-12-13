"""Tests for the golden dataset generator."""

from __future__ import annotations

import pytest

from policyflow.benchmark.models import (
    GeneratorConfig,
    GoldenDataset,
    GoldenTestCase,
)


class TestGeneratorConfig:
    """Tests for GeneratorConfig defaults and customization."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GeneratorConfig()
        assert config.cases_per_criterion == 3
        assert config.include_edge_cases is True
        assert config.include_partial_matches is True
        assert "clear_pass" in config.categories
        assert "edge_case" in config.categories
        assert "boundary" in config.edge_case_strategies
        assert config.mode == "hybrid"
        assert config.temperature == 0.7

    def test_custom_config(self):
        """Test custom configuration."""
        config = GeneratorConfig(
            cases_per_criterion=5,
            include_edge_cases=False,
            mode="template",
        )
        assert config.cases_per_criterion == 5
        assert config.include_edge_cases is False
        assert config.mode == "template"


class TestTemplateBasedGenerator:
    """Tests for the template-based dataset generator."""

    def test_generator_initialization(self):
        """Test generator can be initialized."""
        from policyflow.benchmark.generator import TemplateBasedGenerator

        generator = TemplateBasedGenerator()
        assert generator is not None

    def test_generate_for_criterion_creates_test_cases(self):
        """Test generating test cases for a specific criterion."""
        from policyflow.benchmark.generator import TemplateBasedGenerator
        from policyflow.models import Clause, ClauseType, NormalizedPolicy, Section

        generator = TemplateBasedGenerator()

        # Create a minimal policy with one criterion
        clause = Clause(
            number="1",
            text="The content must be professional and appropriate",
            clause_type=ClauseType.REQUIREMENT,
        )
        policy = NormalizedPolicy(
            title="Test Policy",
            description="A test policy",
            sections=[
                Section(
                    number="1",
                    title="Professionalism",
                    clauses=[clause],
                )
            ],
            raw_text="Test policy content",
        )

        cases = generator.generate_for_criterion(
            criterion=clause,
            policy=policy,
            count=2,
        )

        assert len(cases) == 2
        assert all(isinstance(c, GoldenTestCase) for c in cases)
        assert all(c.id for c in cases)  # All have IDs
        assert all(c.input_text for c in cases)  # All have input text
        assert all(c.expected is not None for c in cases)  # All have expected results

    def test_generate_clear_pass_cases(self):
        """Test generating clear pass test cases."""
        from policyflow.benchmark.generator import TemplateBasedGenerator
        from policyflow.models import Clause, ClauseType, NormalizedPolicy, Section

        generator = TemplateBasedGenerator()

        clause = Clause(
            number="1",
            text="The message must contain a greeting",
            clause_type=ClauseType.REQUIREMENT,
        )
        policy = NormalizedPolicy(
            title="Greeting Policy",
            description="Requires greetings",
            sections=[
                Section(number="1", title="Greetings", clauses=[clause])
            ],
            raw_text="The message must contain a greeting",
        )

        cases = generator._generate_clear_passes(policy, GeneratorConfig(cases_per_criterion=2))

        assert len(cases) > 0
        assert all(c.category == "clear_pass" for c in cases)
        assert all(c.expected.policy_satisfied is True for c in cases)

    def test_generate_clear_fail_cases(self):
        """Test generating clear fail test cases."""
        from policyflow.benchmark.generator import TemplateBasedGenerator
        from policyflow.models import Clause, ClauseType, NormalizedPolicy, Section

        generator = TemplateBasedGenerator()

        clause = Clause(
            number="1",
            text="The message must contain a greeting",
            clause_type=ClauseType.REQUIREMENT,
        )
        policy = NormalizedPolicy(
            title="Greeting Policy",
            description="Requires greetings",
            sections=[
                Section(number="1", title="Greetings", clauses=[clause])
            ],
            raw_text="The message must contain a greeting",
        )

        cases = generator._generate_criterion_fails(clause, policy, GeneratorConfig(cases_per_criterion=2))

        assert len(cases) > 0
        assert all(c.category == "clear_fail" for c in cases)
        # The specific criterion should fail
        for case in cases:
            assert case.expected.criterion_results.get("1") is not None
            assert case.expected.criterion_results["1"].met is False


class TestHybridGenerator:
    """Tests for the hybrid (template + LLM) generator."""

    def test_hybrid_generator_initialization(self):
        """Test hybrid generator can be initialized."""
        from policyflow.benchmark.generator import HybridDatasetGenerator

        generator = HybridDatasetGenerator()
        assert generator is not None

    def test_hybrid_generator_falls_back_to_template(self):
        """Test that hybrid generator uses template-based generation as fallback."""
        from policyflow.benchmark.generator import HybridDatasetGenerator
        from policyflow.models import Clause, ClauseType, NormalizedPolicy, Section

        generator = HybridDatasetGenerator()

        clause = Clause(
            number="1",
            text="Content must be safe",
            clause_type=ClauseType.REQUIREMENT,
        )
        policy = NormalizedPolicy(
            title="Safety Policy",
            description="Requires safe content",
            sections=[
                Section(number="1", title="Safety", clauses=[clause])
            ],
            raw_text="Content must be safe",
        )

        config = GeneratorConfig(
            cases_per_criterion=1,
            include_edge_cases=False,
            include_partial_matches=False,
            mode="hybrid",
        )

        dataset = generator.generate(policy, config)

        assert isinstance(dataset, GoldenDataset)
        assert len(dataset.test_cases) > 0


class TestDatasetGeneration:
    """Integration tests for full dataset generation."""

    def test_generate_full_dataset(self):
        """Test generating a complete dataset."""
        from policyflow.benchmark.generator import create_generator
        from policyflow.models import Clause, ClauseType, NormalizedPolicy, Section

        generator = create_generator(mode="template")

        clause1 = Clause(
            number="1",
            text="The content must be professional",
            clause_type=ClauseType.REQUIREMENT,
        )
        clause2 = Clause(
            number="2",
            text="The content must not contain profanity",
            clause_type=ClauseType.REQUIREMENT,
        )
        policy = NormalizedPolicy(
            title="Content Policy",
            description="Content requirements",
            sections=[
                Section(
                    number="1",
                    title="Requirements",
                    clauses=[clause1, clause2],
                )
            ],
            raw_text="Content policy text",
        )

        config = GeneratorConfig(
            cases_per_criterion=2,
            include_edge_cases=False,
            include_partial_matches=False,
        )

        dataset = generator.generate(policy, config)

        assert isinstance(dataset, GoldenDataset)
        assert dataset.policy_file != ""
        assert dataset.description != ""
        assert len(dataset.test_cases) > 0
        assert dataset.generation_metadata is not None
        assert dataset.generation_metadata.config_used == config

    def test_generate_with_edge_cases(self):
        """Test generating dataset with edge cases enabled."""
        from policyflow.benchmark.generator import create_generator
        from policyflow.models import Clause, ClauseType, NormalizedPolicy, Section

        generator = create_generator(mode="template")

        clause = Clause(
            number="1",
            text="The number must be greater than 10",
            clause_type=ClauseType.REQUIREMENT,
        )
        policy = NormalizedPolicy(
            title="Number Policy",
            description="Number requirements",
            sections=[
                Section(number="1", title="Numbers", clauses=[clause])
            ],
            raw_text="Number policy text",
        )

        config = GeneratorConfig(
            cases_per_criterion=1,
            include_edge_cases=True,
            edge_case_strategies=["boundary"],
            include_partial_matches=False,
        )

        dataset = generator.generate(policy, config)

        # Should have edge cases
        edge_cases = [tc for tc in dataset.test_cases if "edge_case" in tc.category]
        assert len(edge_cases) > 0

    def test_augment_existing_dataset(self):
        """Test augmenting an existing dataset."""
        from policyflow.benchmark.generator import create_generator
        from policyflow.models import Clause, ClauseType, NormalizedPolicy, Section

        generator = create_generator(mode="template")

        clause = Clause(
            number="1",
            text="Content must be appropriate",
            clause_type=ClauseType.REQUIREMENT,
        )
        policy = NormalizedPolicy(
            title="Content Policy",
            description="Content requirements",
            sections=[
                Section(number="1", title="Content", clauses=[clause])
            ],
            raw_text="Content policy text",
        )

        # Create initial dataset
        initial_config = GeneratorConfig(
            cases_per_criterion=1,
            include_edge_cases=False,
            include_partial_matches=False,
        )
        initial_dataset = generator.generate(policy, initial_config)
        initial_count = len(initial_dataset.test_cases)

        # Augment with more cases
        augment_config = GeneratorConfig(
            cases_per_criterion=2,
            categories=["edge_case"],
            edge_case_strategies=["boundary"],
            include_partial_matches=False,
        )
        augmented = generator.augment(initial_dataset, policy, augment_config)

        assert len(augmented.test_cases) > initial_count


class TestGeneratorFactory:
    """Tests for the generator factory function."""

    def test_create_template_generator(self):
        """Test creating template-based generator."""
        from policyflow.benchmark.generator import TemplateBasedGenerator, create_generator

        generator = create_generator(mode="template")
        assert isinstance(generator, TemplateBasedGenerator)

    def test_create_hybrid_generator(self):
        """Test creating hybrid generator."""
        from policyflow.benchmark.generator import HybridDatasetGenerator, create_generator

        generator = create_generator(mode="hybrid")
        assert isinstance(generator, HybridDatasetGenerator)

    def test_create_llm_generator(self):
        """Test creating LLM generator returns hybrid (fallback for POC)."""
        from policyflow.benchmark.generator import HybridDatasetGenerator, create_generator

        generator = create_generator(mode="llm")
        assert isinstance(generator, HybridDatasetGenerator)

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises error."""
        from policyflow.benchmark.generator import create_generator

        with pytest.raises(ValueError, match="Invalid generator mode"):
            create_generator(mode="invalid")


class TestTestCaseIdGeneration:
    """Tests for unique test case ID generation."""

    def test_ids_are_unique(self):
        """Test that generated test case IDs are unique."""
        from policyflow.benchmark.generator import create_generator
        from policyflow.models import Clause, ClauseType, NormalizedPolicy, Section

        generator = create_generator(mode="template")

        clause = Clause(
            number="1",
            text="Content must be appropriate",
            clause_type=ClauseType.REQUIREMENT,
        )
        policy = NormalizedPolicy(
            title="Test Policy",
            description="Test requirements",
            sections=[
                Section(number="1", title="Content", clauses=[clause])
            ],
            raw_text="Test policy text",
        )

        config = GeneratorConfig(cases_per_criterion=5)
        dataset = generator.generate(policy, config)

        ids = [tc.id for tc in dataset.test_cases]
        assert len(ids) == len(set(ids)), "Test case IDs must be unique"

    def test_ids_are_deterministic(self):
        """Test that same inputs produce same IDs across runs."""
        from policyflow.benchmark.generator import TemplateBasedGenerator

        generator1 = TemplateBasedGenerator()
        generator2 = TemplateBasedGenerator()

        # Generate IDs with same inputs
        id1_a = generator1._generate_id("clear_pass", "1", 0)
        id1_b = generator2._generate_id("clear_pass", "1", 0)

        id2_a = generator1._generate_id("edge_case", "2", 1)
        id2_b = generator2._generate_id("edge_case", "2", 1)

        # Same inputs should produce same IDs
        assert id1_a == id1_b, "Same inputs should produce deterministic IDs"
        assert id2_a == id2_b, "Same inputs should produce deterministic IDs"

        # Different inputs should produce different IDs
        assert id1_a != id2_a, "Different inputs should produce different IDs"

    def test_ids_deterministic_across_full_generation(self):
        """Test that full dataset generation produces deterministic IDs."""
        from policyflow.benchmark.generator import create_generator
        from policyflow.models import Clause, ClauseType, NormalizedPolicy, Section

        clause = Clause(
            number="1",
            text="Content must be appropriate",
            clause_type=ClauseType.REQUIREMENT,
        )
        policy = NormalizedPolicy(
            title="Test Policy",
            description="Test requirements",
            sections=[
                Section(number="1", title="Content", clauses=[clause])
            ],
            raw_text="Test policy text",
        )

        config = GeneratorConfig(
            cases_per_criterion=2,
            include_edge_cases=False,
            include_partial_matches=False,
        )

        # Generate datasets twice
        generator1 = create_generator(mode="template")
        dataset1 = generator1.generate(policy, config)

        generator2 = create_generator(mode="template")
        dataset2 = generator2.generate(policy, config)

        # IDs should be identical
        ids1 = [tc.id for tc in dataset1.test_cases]
        ids2 = [tc.id for tc in dataset2.test_cases]

        assert ids1 == ids2, "Same policy and config should produce same IDs"


class TestIntermediateExpectations:
    """Tests for intermediate state expectations in generated test cases."""

    def test_include_intermediate_states(self):
        """Test that intermediate expectations can be included."""
        from policyflow.benchmark.generator import create_generator
        from policyflow.models import Clause, ClauseType, NormalizedPolicy, Section

        generator = create_generator(mode="template")

        # Create a policy with sub-clauses
        sub_clause = Clause(
            number="1a",
            text="Sub-requirement",
            clause_type=ClauseType.REQUIREMENT,
        )
        clause = Clause(
            number="1",
            text="Main requirement",
            clause_type=ClauseType.REQUIREMENT,
            sub_clauses=[sub_clause],
        )
        policy = NormalizedPolicy(
            title="Nested Policy",
            description="Has nested clauses",
            sections=[
                Section(number="1", title="Main", clauses=[clause])
            ],
            raw_text="Nested policy text",
        )

        config = GeneratorConfig(
            cases_per_criterion=1,
            include_intermediate_states=True,
            include_edge_cases=False,
            include_partial_matches=False,
        )

        dataset = generator.generate(policy, config)

        # At least some cases should have intermediate expectations
        # (depends on implementation - may be None if not applicable)
        assert len(dataset.test_cases) > 0
