"""Pattern matching node for deterministic regex/keyword checks."""

import re
from typing import Literal

from pocketflow import Node
from pydantic import BaseModel, Field

from .schema import NodeParameter, NodeSchema


class PatternMatchResult(BaseModel):
    """Result from PatternMatchNode."""

    matched: bool = Field(description="Whether patterns matched based on mode")
    matched_patterns: list[str] = Field(
        default_factory=list, description="List of patterns that matched"
    )
    match_details: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Pattern -> list of matched strings",
    )


class PatternMatchNode(Node):
    """
    Deterministic node that checks input against regex/keyword patterns.
    No LLM involved - uses pure regex matching.

    Shared Store:
        Reads: shared["input_text"]
        Writes: shared["pattern_match_result"]

    Actions:
        - "matched": At least one pattern matched (mode="any"), all matched (mode="all"),
                     or none matched (mode="none")
        - "not_matched": Conditions not met based on mode
    """

    parser_schema = NodeSchema(
        name="PatternMatchNode",
        description="Check input against regex/keyword patterns (deterministic, no LLM)",
        category="deterministic",
        parameters=[
            NodeParameter(
                "patterns",
                "list[str]",
                "Regex patterns to match against input",
                required=True,
            ),
            NodeParameter(
                "mode",
                "str",
                "'any' (match if any pattern matches), 'all' (all must match), or 'none' (none should match)",
                required=False,
                default="any",
            ),
        ],
        actions=["matched", "not_matched"],
        yaml_example="""- type: PatternMatchNode
  id: check_pii
  params:
    patterns: ["\\\\b\\\\d{3}-\\\\d{2}-\\\\d{4}\\\\b", "\\\\bpassword\\\\b"]
    mode: any
  routes:
    matched: flag_sensitive
    not_matched: continue_processing""",
        parser_exposed=True,
    )

    def __init__(
        self,
        patterns: list[str],
        mode: Literal["any", "all", "none"] = "any",
    ):
        r"""
        Initialize pattern matcher.

        Args:
            patterns: List of regex patterns to match against input.
                     Examples: [r"\b(password|secret)\b", r"\d{3}-\d{2}-\d{4}"]
            mode: Matching mode:
                  - "any": Match if ANY pattern matches (OR logic)
                  - "all": Match if ALL patterns match (AND logic)
                  - "none": Match if NO patterns match (NOT logic)
        """
        super().__init__()
        self.patterns = patterns
        self.mode = mode
        self.compiled_patterns: list[tuple[str, re.Pattern | None]] = []

        # Compile patterns once for efficiency
        for pattern in patterns:
            try:
                compiled = re.compile(pattern)
                self.compiled_patterns.append((pattern, compiled))
            except re.error as e:
                # Store None for invalid patterns, will be handled in exec
                self.compiled_patterns.append((pattern, None))
                print(f"Warning: Invalid regex pattern '{pattern}': {e}")

    def prep(self, shared: dict) -> dict:
        """Prepare input text for pattern matching."""
        return {
            "input_text": shared.get("input_text", ""),
        }

    def exec(self, prep_res: dict) -> dict:
        """Execute pattern matching against input text."""
        input_text = prep_res["input_text"]
        matched_patterns = []
        failed_patterns = []

        for pattern_str, compiled_pattern in self.compiled_patterns:
            if compiled_pattern is None:
                # Invalid pattern - treat as not matched
                failed_patterns.append({
                    "pattern": pattern_str,
                    "error": "Invalid regex pattern",
                })
                continue

            # Search for pattern in input
            matches = compiled_pattern.finditer(input_text)
            match_list = list(matches)

            if match_list:
                # Pattern matched - collect details
                match_details = []
                for match in match_list:
                    match_details.append({
                        "text": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                        "groups": match.groups(),
                    })

                matched_patterns.append({
                    "pattern": pattern_str,
                    "match_count": len(match_list),
                    "matches": match_details,
                })

        # Determine if overall match condition is met based on mode
        total_patterns = len(self.compiled_patterns)
        valid_patterns = total_patterns - len(failed_patterns)
        matched_count = len(matched_patterns)

        if self.mode == "any":
            # Match if at least one pattern matched
            is_matched = matched_count > 0
        elif self.mode == "all":
            # Match if all valid patterns matched
            is_matched = matched_count == valid_patterns and valid_patterns > 0
        elif self.mode == "none":
            # Match if no patterns matched
            is_matched = matched_count == 0
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return {
            "is_matched": is_matched,
            "mode": self.mode,
            "matched_patterns": matched_patterns,
            "failed_patterns": failed_patterns,
            "total_patterns": total_patterns,
            "matched_count": matched_count,
            "valid_patterns": valid_patterns,
        }

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store pattern match result and return routing action."""
        shared["pattern_match_result"] = exec_res

        if exec_res["is_matched"]:
            return "matched"
        else:
            return "not_matched"
