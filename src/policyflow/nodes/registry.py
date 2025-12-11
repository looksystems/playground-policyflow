"""Node registry for dynamic discovery and instantiation."""

from typing import Type

from pocketflow import Node

from .schema import NodeSchema


# Global registry of node classes
_node_registry: dict[str, Type[Node]] = {}


def register_node(cls: Type[Node]) -> Type[Node]:
    """Register a node class in the global registry.

    Can be used as a decorator or called directly.

    Args:
        cls: The node class to register

    Returns:
        The same class (for decorator use)
    """
    _node_registry[cls.__name__] = cls
    return cls


def get_node_class(name: str) -> Type[Node] | None:
    """Get a node class by its name.

    Args:
        name: The class name (e.g., "PatternMatchNode")

    Returns:
        The node class, or None if not found
    """
    return _node_registry.get(name)


def get_all_nodes() -> dict[str, Type[Node]]:
    """Get all registered nodes.

    Returns:
        Dictionary mapping class names to node classes
    """
    return _node_registry.copy()


def get_parser_exposed_nodes() -> list[Type[Node]]:
    """Get nodes that have parser_exposed=True in their schema.

    Returns:
        List of node classes that should be exposed to the parser
    """
    exposed = []
    for cls in _node_registry.values():
        schema = getattr(cls, "parser_schema", None)
        if schema and schema.parser_exposed:
            exposed.append(cls)
    return exposed


def get_parser_schemas() -> list[NodeSchema]:
    """Get schemas for all parser-exposed nodes.

    Returns:
        List of NodeSchema objects for nodes with parser_exposed=True
    """
    schemas = []
    for cls in get_parser_exposed_nodes():
        schemas.append(cls.parser_schema)
    return schemas
