"""Clause numbering utilities for policy documents.

This module provides utilities for generating and manipulating hierarchical
clause numbers in the format: 1, 1.1, 1.1.1, 1.1.1.a, 1.1.1.b, etc.

Numbering scheme:
- Depth 0 (sections): "1", "2", "3"
- Depth 1 (clauses): "1.1", "1.2", "2.1"
- Depth 2 (sub-clauses): "1.1.1", "1.1.2"
- Depth 3+ (deep sub-clauses): "1.1.1.a", "1.1.1.b", "1.1.1.aa"
"""

from __future__ import annotations


def generate_clause_number(
    parent_number: str | None,
    index: int,
    depth: int,
) -> str:
    """Generate a clause number based on parent and position.

    Args:
        parent_number: The parent clause number (None for top-level)
        index: 0-based index within siblings
        depth: Current nesting depth (0 = top-level section)

    Returns:
        The generated clause number

    Examples:
        >>> generate_clause_number(None, 0, 0)
        '1'
        >>> generate_clause_number('1', 0, 1)
        '1.1'
        >>> generate_clause_number('1.1', 0, 2)
        '1.1.1'
        >>> generate_clause_number('1.1.1', 0, 3)
        '1.1.1.a'
        >>> generate_clause_number('1.1.1', 25, 3)
        '1.1.1.z'
        >>> generate_clause_number('1.1.1', 26, 3)
        '1.1.1.aa'
    """
    if depth <= 2:
        # Numeric: 1, 1.1, 1.1.1
        suffix = str(index + 1)
    else:
        # Alphabetic for depth 3+
        suffix = _index_to_alpha(index)

    if parent_number:
        return f"{parent_number}.{suffix}"
    return suffix


def _index_to_alpha(index: int) -> str:
    """Convert 0-based index to alphabetic string.

    0 -> 'a', 25 -> 'z', 26 -> 'aa', 27 -> 'ab', etc.

    Args:
        index: 0-based index

    Returns:
        Alphabetic string representation

    Examples:
        >>> _index_to_alpha(0)
        'a'
        >>> _index_to_alpha(25)
        'z'
        >>> _index_to_alpha(26)
        'aa'
        >>> _index_to_alpha(27)
        'ab'
    """
    result = []
    index += 1  # Convert to 1-based
    while index > 0:
        index -= 1
        result.append(chr(ord("a") + index % 26))
        index //= 26
    return "".join(reversed(result))


def _alpha_to_index(alpha: str) -> int:
    """Convert alphabetic string to 0-based index.

    'a' -> 0, 'z' -> 25, 'aa' -> 26, 'ab' -> 27, etc.

    Args:
        alpha: Alphabetic string (lowercase)

    Returns:
        0-based index

    Examples:
        >>> _alpha_to_index('a')
        0
        >>> _alpha_to_index('z')
        25
        >>> _alpha_to_index('aa')
        26
    """
    result = 0
    for char in alpha.lower():
        result = result * 26 + (ord(char) - ord("a") + 1)
    return result - 1


def clause_number_to_node_id(clause_number: str) -> str:
    """Convert clause number to valid node ID.

    Replaces dots with underscores and adds 'clause_' prefix.

    Args:
        clause_number: Hierarchical clause number

    Returns:
        Valid node ID string

    Examples:
        >>> clause_number_to_node_id('1')
        'clause_1'
        >>> clause_number_to_node_id('1.1')
        'clause_1_1'
        >>> clause_number_to_node_id('1.1.a')
        'clause_1_1_a'
        >>> clause_number_to_node_id('1.1.1.aa')
        'clause_1_1_1_aa'
    """
    safe_id = clause_number.replace(".", "_")
    return f"clause_{safe_id}"


def node_id_to_clause_number(node_id: str) -> str | None:
    """Extract clause number from node ID.

    Args:
        node_id: Node ID string

    Returns:
        Clause number if the node ID follows the clause pattern, None otherwise

    Examples:
        >>> node_id_to_clause_number('clause_1')
        '1'
        >>> node_id_to_clause_number('clause_1_1')
        '1.1'
        >>> node_id_to_clause_number('clause_1_1_a')
        '1.1.a'
        >>> node_id_to_clause_number('preprocess')
        None
    """
    if not node_id.startswith("clause_"):
        return None

    suffix = node_id[7:]  # Remove 'clause_'
    if not suffix:
        return None

    # Split and reconstruct with dots
    parts = suffix.split("_")
    return ".".join(parts)


def parse_clause_depth(clause_number: str) -> int:
    """Determine the depth of a clause number.

    Args:
        clause_number: Hierarchical clause number

    Returns:
        Depth level (0 for top-level)

    Examples:
        >>> parse_clause_depth('1')
        0
        >>> parse_clause_depth('1.1')
        1
        >>> parse_clause_depth('1.1.a')
        2
        >>> parse_clause_depth('1.1.1.a')
        3
    """
    return clause_number.count(".")


def get_parent_clause_number(clause_number: str) -> str | None:
    """Get the parent clause number.

    Args:
        clause_number: Hierarchical clause number

    Returns:
        Parent clause number, or None if top-level

    Examples:
        >>> get_parent_clause_number('1')
        None
        >>> get_parent_clause_number('1.1')
        '1'
        >>> get_parent_clause_number('1.1.a')
        '1.1'
    """
    if "." not in clause_number:
        return None
    return clause_number.rsplit(".", 1)[0]


def clause_sort_key(clause_number: str) -> tuple:
    """Generate sort key for clause numbers.

    Ensures proper ordering: 1 < 1.1 < 1.1.a < 1.1.b < 1.2 < 2

    Args:
        clause_number: Hierarchical clause number

    Returns:
        Tuple suitable for sorting

    Examples:
        >>> sorted(['1.2', '1.1.a', '1.1', '2', '1'], key=clause_sort_key)
        ['1', '1.1', '1.1.a', '1.2', '2']
    """
    parts = clause_number.replace(".", " ").split()
    result = []
    for part in parts:
        if part.isdigit():
            # Numeric parts: sort by value
            result.append((0, int(part), ""))
        else:
            # Alphabetic parts: sort alphabetically
            result.append((1, _alpha_to_index(part), part))
    return tuple(result)


def is_ancestor_of(ancestor: str, descendant: str) -> bool:
    """Check if one clause is an ancestor of another.

    Args:
        ancestor: Potential ancestor clause number
        descendant: Potential descendant clause number

    Returns:
        True if ancestor is a proper ancestor of descendant

    Examples:
        >>> is_ancestor_of('1', '1.1')
        True
        >>> is_ancestor_of('1', '1.1.a')
        True
        >>> is_ancestor_of('1.1', '1.1.a')
        True
        >>> is_ancestor_of('1.1', '1.2')
        False
        >>> is_ancestor_of('1', '2')
        False
    """
    if ancestor == descendant:
        return False
    return descendant.startswith(ancestor + ".")
