"""Text transformation node for PocketFlow."""

import re
from pocketflow import Node

from .schema import NodeParameter, NodeSchema


class TransformNode(Node):
    """
    Transforms/preprocesses input text through a series of transformations.
    This is a deterministic node (no LLM).

    Shared Store:
        Reads: shared[input_key] (default: "input_text")
        Writes: shared[output_key] (default: "input_text")
    """

    parser_schema = NodeSchema(
        name="TransformNode",
        description="Preprocess/transform text (lowercase, strip HTML, truncate, etc.)",
        category="deterministic",
        parameters=[
            NodeParameter(
                "transforms",
                "list[str]",
                "Transform operations: lowercase, uppercase, strip_html, normalize_whitespace, strip_urls, strip_emails, truncate:N, trim",
                required=True,
            ),
            NodeParameter(
                "input_key",
                "str",
                "Shared store key to read from",
                required=False,
                default="input_text",
            ),
            NodeParameter(
                "output_key",
                "str",
                "Shared store key to write to",
                required=False,
                default="input_text",
            ),
        ],
        actions=["default"],
        yaml_example="""- type: TransformNode
  id: preprocess
  params:
    transforms:
      - lowercase
      - strip_html
      - normalize_whitespace
      - truncate:5000
  routes:
    default: next_node""",
        parser_exposed=True,
    )

    def __init__(
        self,
        transforms: list[str],
        input_key: str = "input_text",
        output_key: str = "input_text",
    ):
        """
        Initialize transform node.

        Args:
            transforms: List of transformation operations to apply in order.
                Supported: "lowercase", "uppercase", "strip_html",
                "normalize_whitespace", "strip_urls", "strip_emails",
                "truncate:N", "trim"
            input_key: Key to read from shared store
            output_key: Key to write to shared store
        """
        super().__init__()
        self.transforms = transforms
        self.input_key = input_key
        self.output_key = output_key

    def prep(self, shared: dict) -> dict:
        """Retrieve input text from shared store."""
        return {
            "text": shared.get(self.input_key, ""),
        }

    def exec(self, prep_res: dict) -> dict:
        """Apply all transformations in sequence."""
        text = prep_res["text"]

        for transform in self.transforms:
            text = self._apply_transform(text, transform)

        return {"transformed_text": text}

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store transformed text in shared store."""
        shared[self.output_key] = exec_res["transformed_text"]
        return "default"

    def _apply_transform(self, text: str, transform: str) -> str:
        """
        Apply a single transformation to the text.

        Args:
            text: Input text
            transform: Transform specification (e.g., "lowercase", "truncate:1000")

        Returns:
            Transformed text
        """
        # Handle parameterized transforms
        if ":" in transform:
            transform_name, param = transform.split(":", 1)
        else:
            transform_name = transform
            param = None

        # Apply transformation
        if transform_name == "lowercase":
            return text.lower()
        elif transform_name == "uppercase":
            return text.upper()
        elif transform_name == "strip_html":
            # Remove HTML tags using regex
            return re.sub(r"<[^>]+>", "", text)
        elif transform_name == "normalize_whitespace":
            # Collapse multiple whitespace to single space
            return re.sub(r"\s+", " ", text)
        elif transform_name == "strip_urls":
            # Remove URLs (http/https)
            return re.sub(
                r"https?://[^\s]+", "", text
            )
        elif transform_name == "strip_emails":
            # Remove email addresses
            return re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text
            )
        elif transform_name == "truncate":
            # Truncate to N characters
            if param:
                max_length = int(param)
                return text[:max_length]
            return text
        elif transform_name == "trim":
            # Strip leading/trailing whitespace
            return text.strip()
        else:
            # Unknown transform, return text unchanged
            return text
