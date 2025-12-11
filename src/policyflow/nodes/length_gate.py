"""Length gate node for routing based on input length thresholds."""

from pocketflow import Node
from pydantic import BaseModel, Field

from .schema import NodeParameter, NodeSchema


class LengthInfo(BaseModel):
    """Result from LengthGateNode."""

    char_count: int = Field(description="Character count")
    word_count: int = Field(description="Word count")
    bucket: str = Field(description="Length bucket name")


class LengthGateNode(Node):
    """
    Routes workflow based on input text length.

    Shared Store:
        Reads: shared["input_text"]
        Writes: shared["length_info"]

    Actions:
        Returns bucket name based on character count thresholds
        (e.g., "short", "medium", "long")
    """

    parser_schema = NodeSchema(
        name="LengthGateNode",
        description="Route based on input text length (deterministic, no LLM)",
        category="deterministic",
        parameters=[
            NodeParameter(
                "thresholds",
                "dict[str, int]",
                "Map bucket names to character count thresholds",
                required=True,
            ),
        ],
        actions=["<bucket_name>"],
        yaml_example="""- type: LengthGateNode
  id: length_check
  params:
    thresholds:
      short: 100
      medium: 1000
      long: 5000
  routes:
    short: quick_process
    medium: standard_process
    long: detailed_process""",
        parser_exposed=True,
    )

    def __init__(self, thresholds: dict[str, int]):
        """
        Initialize length gate with thresholds.

        Args:
            thresholds: Dictionary mapping bucket names to character count thresholds.
                       Example: {"short": 100, "medium": 1000, "long": 5000}
                       Input <= 100 chars → "short", <= 1000 → "medium", etc.
        """
        super().__init__()
        # Sort thresholds by value for efficient lookup
        self.sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])

    def prep(self, shared: dict) -> dict:
        """Extract input text for length analysis."""
        return {
            "input_text": shared.get("input_text", ""),
        }

    def exec(self, prep_res: dict) -> dict:
        """Calculate length metrics and determine bucket."""
        input_text = prep_res["input_text"]

        char_count = len(input_text)
        word_count = len(input_text.split())

        # Find appropriate bucket based on character count
        bucket = None
        for bucket_name, threshold in self.sorted_thresholds:
            if char_count <= threshold:
                bucket = bucket_name
                break

        # If no threshold matched, use the last (highest) bucket
        if bucket is None and self.sorted_thresholds:
            bucket = self.sorted_thresholds[-1][0]

        return {
            "char_count": char_count,
            "word_count": word_count,
            "bucket": bucket,
        }

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store length info and return bucket name as action."""
        shared["length_info"] = exec_res

        return exec_res["bucket"]
