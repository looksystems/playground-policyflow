"""Pydantic data models for policy evaluation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Self

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .nodes.criterion import CriterionResult


class YAMLMixin:
    """Mixin class providing YAML serialization methods."""

    def to_yaml(self) -> str:
        """Serialize model to YAML string."""
        return yaml.dump(
            self.model_dump(mode="json"),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    def save_yaml(self, path: str | Path) -> None:
        """Save model to a YAML file."""
        path = Path(path)
        with path.open("w") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Self:
        """Load model from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    @classmethod
    def load_yaml(cls, path: str | Path) -> Self:
        """Load model from a YAML file."""
        path = Path(path)
        with path.open() as f:
            return cls.from_yaml(f.read())


class LogicOperator(str, Enum):
    """How criteria are combined."""

    ALL = "all"  # AND logic - all must match
    ANY = "any"  # OR logic - at least one must match


class ClauseType(str, Enum):
    """Type of clause content in a normalized policy."""

    REQUIREMENT = "requirement"  # Evaluatable requirement
    DEFINITION = "definition"  # Defines terms
    CONDITION = "condition"  # Conditional logic (if/then)
    EXCEPTION = "exception"  # Exception to a rule
    REFERENCE = "reference"  # Reference to external document


class Criterion(BaseModel):
    """A single criterion extracted from a policy."""

    id: str = Field(description="Unique identifier, e.g., 'criterion_1'")
    name: str = Field(description="Short name for the criterion")
    description: str = Field(description="Full text of the criterion")
    sub_criteria: list["Criterion"] = Field(
        default_factory=list,
        description="Nested sub-criteria (for OR within AND, etc.)",
    )
    sub_logic: LogicOperator | None = Field(
        default=None,
        description="Logic for combining sub-criteria",
    )


class ParsedPolicy(YAMLMixin, BaseModel):
    """Complete parsed policy structure."""

    title: str = Field(description="Policy title or name")
    description: str = Field(description="Overall policy description")
    criteria: list[Criterion] = Field(description="Top-level criteria")
    logic: LogicOperator = Field(
        default=LogicOperator.ALL,
        description="How top-level criteria are combined",
    )
    raw_text: str = Field(description="Original policy markdown text")


# ============================================================================
# Normalized Policy Models (Two-Step Parser)
# ============================================================================


class Clause(BaseModel):
    """A single clause in a normalized policy document."""

    number: str = Field(
        description="Hierarchical clause number (e.g., '1', '1.1', '1.1.a')"
    )
    title: str = Field(default="", description="Optional short title for the clause")
    text: str = Field(description="Original text of the clause")
    clause_type: ClauseType = Field(
        default=ClauseType.REQUIREMENT,
        description="Type of clause for processing hints",
    )
    sub_clauses: list["Clause"] = Field(
        default_factory=list,
        description="Nested sub-clauses",
    )
    logic: LogicOperator | None = Field(
        default=None,
        description="How sub-clauses combine (all/any)",
    )

    @property
    def node_id(self) -> str:
        """Generate workflow node ID from clause number.

        Examples:
            '1' -> 'clause_1'
            '1.1' -> 'clause_1_1'
            '1.1.a' -> 'clause_1_1_a'
        """
        normalized = self.number.replace(".", "_")
        return f"clause_{normalized}"

    @property
    def depth(self) -> int:
        """Return the nesting depth of this clause."""
        return self.number.count(".")


class Section(BaseModel):
    """A top-level section in a normalized policy document."""

    number: str = Field(description="Section number (e.g., '1', '2')")
    title: str = Field(description="Section title")
    description: str = Field(
        default="",
        description="Optional section description/preamble",
    )
    clauses: list[Clause] = Field(
        default_factory=list,
        description="Clauses within this section",
    )
    logic: LogicOperator = Field(
        default=LogicOperator.ALL,
        description="How clauses in this section combine",
    )


class NormalizedPolicy(YAMLMixin, BaseModel):
    """A policy document normalized into hierarchical sections and clauses.

    This is the intermediate format between raw markdown and workflow generation.
    It can be persisted to YAML for review before workflow generation.
    """

    title: str = Field(description="Policy title")
    version: str = Field(default="1.0", description="Policy version")
    effective_date: str | None = Field(
        default=None,
        description="When the policy takes effect",
    )
    description: str = Field(description="Overall policy description")
    sections: list[Section] = Field(
        default_factory=list,
        description="Top-level sections",
    )
    logic: LogicOperator = Field(
        default=LogicOperator.ALL,
        description="How top-level sections combine",
    )
    raw_text: str = Field(description="Original policy markdown")
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (source, author, etc.)",
    )

    def get_all_clauses(self) -> list[Clause]:
        """Flatten all clauses including nested ones."""
        result: list[Clause] = []

        def collect(clauses: list[Clause]) -> None:
            for clause in clauses:
                result.append(clause)
                if clause.sub_clauses:
                    collect(clause.sub_clauses)

        for section in self.sections:
            collect(section.clauses)

        return result

    def get_clause_by_number(self, number: str) -> Clause | None:
        """Find a clause by its number."""
        for clause in self.get_all_clauses():
            if clause.number == number:
                return clause
        return None


class ConfidenceLevel(str, Enum):
    """Confidence level classification."""

    HIGH = "high"  # Above high threshold
    MEDIUM = "medium"  # Between thresholds
    LOW = "low"  # Below low threshold


class EvaluationResult(YAMLMixin, BaseModel):
    """Complete evaluation result."""

    policy_satisfied: bool = Field(
        description="Whether the overall policy is satisfied"
    )
    input_text: str = Field(description="The text that was evaluated")
    policy_title: str = Field(description="Title of the policy used")
    criterion_results: list[CriterionResult] = Field(
        description="Per-criterion evaluation results"
    )
    overall_reasoning: str = Field(description="Summary of the evaluation")
    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score",
    )
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Classified confidence level",
    )
    needs_review: bool = Field(
        default=False,
        description="Whether human review is recommended",
    )
    low_confidence_criteria: list[str] = Field(
        default_factory=list,
        description="IDs of criteria with low confidence scores",
    )


# ============================================================================
# Dynamic Workflow Models
# ============================================================================


class NodeConfig(BaseModel):
    """Configuration for a single node in a dynamic workflow."""

    id: str = Field(description="Unique node identifier")
    type: str = Field(description="Node class name (e.g., 'PatternMatchNode')")
    params: dict = Field(
        default_factory=dict,
        description="Node constructor parameters",
    )
    routes: dict[str, str] = Field(
        default_factory=dict,
        description="Action -> next node ID mapping",
    )


class WorkflowDefinition(BaseModel):
    """Definition of a workflow's node graph."""

    nodes: list[NodeConfig] = Field(description="List of node configurations")
    start_node: str = Field(description="ID of the starting node")


class ParsedWorkflowPolicy(YAMLMixin, BaseModel):
    """Policy parsed into a dynamic workflow definition."""

    title: str = Field(description="Policy title")
    description: str = Field(description="Policy description")
    workflow: WorkflowDefinition = Field(description="Workflow configuration")
    raw_text: str = Field(default="", description="Original policy markdown")


# ============================================================================
# Hierarchical Workflow Models (Two-Step Parser - Step 2)
# ============================================================================


class NodeGroup(BaseModel):
    """A group of nodes representing a hierarchical clause."""

    clause_number: str = Field(
        description="The clause number this group represents"
    )
    clause_text: str = Field(
        default="",
        description="Original clause text for reference",
    )
    nodes: list[str] = Field(
        default_factory=list,
        description="Node IDs belonging to this group",
    )
    sub_groups: list["NodeGroup"] = Field(
        default_factory=list,
        description="Nested groups for sub-clauses",
    )
    logic: LogicOperator | None = Field(
        default=None,
        description="How nodes/sub-groups combine",
    )


class HierarchicalWorkflowDefinition(BaseModel):
    """Workflow definition that preserves document hierarchy."""

    nodes: list[NodeConfig] = Field(description="All node configurations")
    start_node: str = Field(description="ID of the starting node")
    hierarchy: list[NodeGroup] = Field(
        default_factory=list,
        description="Hierarchical grouping of nodes by clause",
    )

    def get_nodes_for_clause(self, clause_number: str) -> list[NodeConfig]:
        """Get all nodes that evaluate a specific clause."""

        def find_group(groups: list[NodeGroup]) -> NodeGroup | None:
            for group in groups:
                if group.clause_number == clause_number:
                    return group
                found = find_group(group.sub_groups)
                if found:
                    return found
            return None

        group = find_group(self.hierarchy)
        if not group:
            return []

        return [n for n in self.nodes if n.id in group.nodes]


class ParsedWorkflowPolicyV2(YAMLMixin, BaseModel):
    """Enhanced policy with hierarchical workflow and clause mapping.

    This is the output of the two-step parsing process, containing:
    - The workflow definition with all nodes
    - Hierarchical mapping of nodes to clause numbers
    - Reference to the normalized policy it was generated from
    """

    title: str = Field(description="Policy title")
    description: str = Field(description="Policy description")
    workflow: HierarchicalWorkflowDefinition = Field(
        description="Workflow with hierarchy"
    )
    normalized_policy_ref: str | None = Field(
        default=None,
        description="Path to the normalized policy YAML this was generated from",
    )
    raw_text: str = Field(default="", description="Original policy markdown")


def _rebuild_models():
    """Rebuild models that use forward references from node modules."""
    from .nodes.criterion import CriterionResult
    EvaluationResult.model_rebuild()


_rebuild_models()
