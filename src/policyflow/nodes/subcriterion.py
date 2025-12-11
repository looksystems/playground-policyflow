"""Sub-criterion evaluation node for PocketFlow."""

from pocketflow import Node
from pydantic import BaseModel, Field

from .schema import NodeSchema
from ..models import Criterion, LogicOperator
from ..config import WorkflowConfig
from ..llm import call_llm
from ..prompts import build_subcriterion_prompt


class SubCriterionResult(BaseModel):
    """Evaluation result for a single sub-criterion."""

    sub_criterion_id: str = Field(description="ID of the sub-criterion")
    sub_criterion_name: str = Field(description="Name of the sub-criterion")
    met: bool = Field(description="Whether the sub-criterion is satisfied")
    reasoning: str = Field(description="Explanation for the evaluation")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score 0.0-1.0",
    )


class SubCriterionNode(Node):
    """
    Evaluates a single sub-criterion against input text.
    Supports early termination based on sub_logic (ANY/ALL).

    Shared Store:
        Reads: shared["input_text"], shared["policy_context"]
        Writes: shared["sub_criterion_results"][parent_id][sub_id]

    Actions:
        - "satisfied": Sub-criterion met AND logic is ANY (skip remaining)
        - "failed": Sub-criterion not met AND logic is ALL (skip remaining)
        - "default": Continue to next sub-criterion
    """

    # Class-level default model - can be overridden per instance
    default_model: str = "anthropic/claude-sonnet-4-20250514"

    parser_schema = NodeSchema(
        name="SubCriterionNode",
        description="Internal node for evaluating sub-criteria",
        category="internal",
        parameters=[],
        actions=["satisfied", "failed", "default"],
        parser_exposed=False,
    )

    def __init__(
        self,
        parent_criterion: Criterion,
        sub_criterion: Criterion,
        sub_logic: LogicOperator,
        config: WorkflowConfig | None = None,
        model: str | None = None,
    ):
        super().__init__(max_retries=config.max_retries if config else 3)
        self.parent_criterion = parent_criterion
        self.sub_criterion = sub_criterion
        self.sub_logic = sub_logic
        self.config = config or WorkflowConfig()
        self.model = model if model is not None else self.default_model

    def prep(self, shared: dict) -> dict:
        """Prepare evaluation context."""
        return {
            "input_text": shared["input_text"],
            "parent_criterion": self.parent_criterion,
            "sub_criterion": self.sub_criterion,
            "policy_context": shared.get("policy_context", ""),
        }

    def exec(self, prep_res: dict) -> dict:
        """Evaluate the sub-criterion using LLM."""
        prompt = build_subcriterion_prompt(
            parent_criterion=prep_res["parent_criterion"],
            sub_criterion=prep_res["sub_criterion"],
            policy_context=prep_res["policy_context"],
        )

        return call_llm(
            prompt=f"Evaluate this text:\n\n{prep_res['input_text']}",
            system_prompt=prompt,
            model=self.model,
            config=self.config,
            yaml_response=True,
            span_name=f"subcriterion_{self.parent_criterion.id}_{self.sub_criterion.id}",
        )

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store sub-criterion result and determine next action."""
        result = SubCriterionResult(
            sub_criterion_id=self.sub_criterion.id,
            sub_criterion_name=self.sub_criterion.name,
            met=exec_res.get("met", False),
            reasoning=exec_res.get("reasoning", ""),
            confidence=exec_res.get("confidence", 0.0),
        )

        # Initialize nested dict structure if needed
        parent_id = self.parent_criterion.id
        if "sub_criterion_results" not in shared:
            shared["sub_criterion_results"] = {}
        if parent_id not in shared["sub_criterion_results"]:
            shared["sub_criterion_results"][parent_id] = {}

        shared["sub_criterion_results"][parent_id][self.sub_criterion.id] = result

        # Determine action based on sub_logic
        if self.sub_logic == LogicOperator.ANY and result.met:
            # OR logic: one match is enough, skip remaining
            return "satisfied"
        elif self.sub_logic == LogicOperator.ALL and not result.met:
            # AND logic: one failure is enough, skip remaining
            return "failed"
        else:
            # Continue to next sub-criterion
            return "default"
