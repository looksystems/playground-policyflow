"""Criterion evaluation node for PocketFlow."""

from pocketflow import Node
from pydantic import BaseModel, Field

from .schema import NodeSchema
from .subcriterion import SubCriterionResult
from ..models import Criterion
from ..config import WorkflowConfig
from ..llm import call_llm
from ..prompts import build_criterion_prompt


class CriterionResult(BaseModel):
    """Evaluation result for a single criterion."""

    criterion_id: str = Field(description="ID of the evaluated criterion")
    criterion_name: str = Field(description="Name of the criterion")
    met: bool = Field(description="Whether the criterion is satisfied")
    reasoning: str = Field(description="Explanation for the evaluation")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score 0.0-1.0",
    )
    sub_results: list[SubCriterionResult] = Field(
        default_factory=list,
        description="Results for sub-criteria if any",
    )


class CriterionEvaluationNode(Node):
    """
    Evaluates a single criterion against input text.
    Dynamically created for each criterion in the policy.

    Shared Store:
        Reads: shared["input_text"], shared["parsed_policy"]
        Writes: shared["criterion_results"][criterion_id]
    """

    # Class-level default model - can be overridden per instance
    default_model: str = "anthropic/claude-sonnet-4-20250514"

    parser_schema = NodeSchema(
        name="CriterionEvaluationNode",
        description="Internal node for evaluating policy criteria",
        category="internal",
        parameters=[],
        actions=["default"],
        parser_exposed=False,
    )

    def __init__(
        self,
        criterion: Criterion,
        config: WorkflowConfig | None = None,
        model: str | None = None,
    ):
        super().__init__(max_retries=config.max_retries if config else 3)
        self.criterion = criterion
        self.config = config or WorkflowConfig()
        self.model = model if model is not None else self.default_model

    def prep(self, shared: dict) -> dict:
        """Prepare evaluation context."""
        return {
            "input_text": shared["input_text"],
            "criterion": self.criterion,
            "policy_context": shared["parsed_policy"].description,
        }

    def exec(self, prep_res: dict) -> dict:
        """Evaluate the criterion using LLM."""
        prompt = build_criterion_prompt(
            criterion=prep_res["criterion"],
            policy_context=prep_res["policy_context"],
        )

        return call_llm(
            prompt=f"Evaluate this text:\n\n{prep_res['input_text']}",
            system_prompt=prompt,
            model=self.model,
            config=self.config,
            yaml_response=True,
            span_name=f"criterion_{self.criterion.id}",
        )

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store criterion result."""
        result = CriterionResult(
            criterion_id=self.criterion.id,
            criterion_name=self.criterion.name,
            met=exec_res.get("met", False),
            reasoning=exec_res.get("reasoning", ""),
            confidence=exec_res.get("confidence", 0.0),
        )

        if "criterion_results" not in shared:
            shared["criterion_results"] = {}
        shared["criterion_results"][self.criterion.id] = result

        return "default"
