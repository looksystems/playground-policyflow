"""Schema definitions for node self-documentation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeParameter:
    """Describes a single node constructor parameter."""

    name: str
    type: str  # e.g., "list[str]", "dict[str, float]", "int"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class NodeSchema:
    """Schema describing a node for the parser.

    This schema is used to dynamically generate parser prompts
    that include documentation for available nodes.
    """

    name: str  # Class name, e.g., "PatternMatchNode"
    description: str  # Brief description for LLM
    category: str  # "deterministic" | "llm" | "routing" | "internal"
    parameters: list[NodeParameter] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)  # Possible return actions
    yaml_example: str = ""  # Minimal YAML config example
    parser_exposed: bool = True  # Whether to expose to parser
