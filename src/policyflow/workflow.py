"""Dynamic workflow generation for policy evaluation."""

from pathlib import Path

import yaml
from pocketflow import Flow, Node

from .models import (
    ParsedPolicy,
    Criterion,
    EvaluationResult,
    LogicOperator,
    ConfidenceLevel,
)
from .nodes.criterion import CriterionResult
from .nodes.subcriterion import SubCriterionResult
from .config import WorkflowConfig
from .nodes import (
    CriterionEvaluationNode,
    ResultAggregatorNode,
    SubCriterionNode,
    ConfidenceGateNode,
)


class SubCriterionAggregatorNode(Node):
    """
    Aggregates sub-criterion results into a criterion result.
    Used after evaluating all sub-criteria for a criterion.
    """

    def __init__(self, criterion: Criterion, config: WorkflowConfig | None = None):
        super().__init__()
        self.criterion = criterion
        self.config = config or WorkflowConfig()

    def prep(self, shared: dict) -> dict:
        """Gather sub-criterion results for this criterion."""
        parent_id = self.criterion.id
        sub_results = shared.get("sub_criterion_results", {}).get(parent_id, {})
        return {
            "criterion": self.criterion,
            "sub_results": sub_results,
        }

    def exec(self, prep_res: dict) -> dict:
        """Aggregate sub-criterion results based on sub_logic."""
        criterion = prep_res["criterion"]
        sub_results: dict[str, SubCriterionResult] = prep_res["sub_results"]

        if not sub_results:
            return {
                "met": False,
                "reasoning": "No sub-criteria evaluated",
                "confidence": 0.0,
                "sub_results": [],
            }

        results_list = list(sub_results.values())
        met_values = [r.met for r in results_list]
        confidences = [r.confidence for r in results_list]

        # Apply sub_logic
        sub_logic = criterion.sub_logic or LogicOperator.ANY
        if sub_logic == LogicOperator.ANY:
            met = any(met_values)
            reasoning_prefix = "At least one sub-criterion met" if met else "No sub-criteria met"
        else:  # ALL
            met = all(met_values)
            reasoning_prefix = "All sub-criteria met" if met else "Not all sub-criteria met"

        # Build reasoning
        met_subs = [r for r in results_list if r.met]
        unmet_subs = [r for r in results_list if not r.met]

        reasoning_parts = [reasoning_prefix]
        if met_subs:
            reasoning_parts.append(f"Met: {', '.join(r.sub_criterion_name for r in met_subs)}")
        if unmet_subs:
            reasoning_parts.append(f"Not met: {', '.join(r.sub_criterion_name for r in unmet_subs)}")

        return {
            "met": met,
            "reasoning": ". ".join(reasoning_parts),
            "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "sub_results": results_list,
        }

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store criterion result."""
        result = CriterionResult(
            criterion_id=self.criterion.id,
            criterion_name=self.criterion.name,
            met=exec_res["met"],
            reasoning=exec_res["reasoning"],
            confidence=exec_res["confidence"],
            sub_results=exec_res["sub_results"],
        )

        if "criterion_results" not in shared:
            shared["criterion_results"] = {}
        shared["criterion_results"][self.criterion.id] = result

        return "default"


class PolicyEvaluationWorkflow:
    """
    Dynamically generates and executes an evaluation workflow
    based on parsed policy criteria.

    Workflow structure:
    - For criteria with sub-criteria: SubCriterionNode chain → SubCriterionAggregator
    - For simple criteria: CriterionEvaluationNode
    - After all criteria: ConfidenceGateNode → ResultAggregatorNode
    """

    def __init__(
        self,
        parsed_policy: ParsedPolicy,
        config: WorkflowConfig | None = None,
    ):
        self.policy = parsed_policy
        self.config = config or WorkflowConfig()
        self.flow = self._build_flow()

    def _build_criterion_subflow(self, criterion: Criterion) -> tuple[Node, Node]:
        """
        Build a subflow for a criterion with sub-criteria.
        Returns (first_node, last_node) tuple.
        """
        sub_logic = criterion.sub_logic or LogicOperator.ANY

        # Create SubCriterionNode for each sub-criterion
        sub_nodes = []
        for sub_criterion in criterion.sub_criteria:
            node = SubCriterionNode(
                parent_criterion=criterion,
                sub_criterion=sub_criterion,
                sub_logic=sub_logic,
                config=self.config,
            )
            sub_nodes.append(node)

        # Create aggregator for this criterion
        aggregator = SubCriterionAggregatorNode(criterion, self.config)

        if not sub_nodes:
            return aggregator, aggregator

        # Chain sub-criterion nodes
        # default -> next, satisfied/failed -> aggregator (skip remaining)
        for i, node in enumerate(sub_nodes):
            if i < len(sub_nodes) - 1:
                # Link to next sub-criterion on default
                node - "default" >> sub_nodes[i + 1]
            else:
                # Last node always goes to aggregator
                node - "default" >> aggregator

            # Early termination routes to aggregator
            node - "satisfied" >> aggregator
            node - "failed" >> aggregator

        return sub_nodes[0], aggregator

    def _build_flow(self) -> Flow:
        """Build the complete evaluation flow."""
        all_nodes = []  # List of (first_node, last_node) for each criterion

        for criterion in self.policy.criteria:
            if criterion.sub_criteria:
                # Use SubCriterionNode chain
                first, last = self._build_criterion_subflow(criterion)
                all_nodes.append((first, last))
            else:
                # Use simple CriterionEvaluationNode
                node = CriterionEvaluationNode(
                    criterion=criterion,
                    config=self.config,
                )
                all_nodes.append((node, node))

        # Create confidence gate and aggregator
        confidence_gate = ConfidenceGateNode(self.config)
        aggregator = ResultAggregatorNode()

        if not all_nodes:
            # Edge case: no criteria
            confidence_gate - "high_confidence" >> aggregator
            confidence_gate - "needs_review" >> aggregator
            confidence_gate - "low_confidence" >> aggregator
            return Flow(start=confidence_gate)

        # Chain criterion evaluation nodes/subflows
        for i in range(len(all_nodes) - 1):
            _, current_last = all_nodes[i]
            next_first, _ = all_nodes[i + 1]
            current_last - "default" >> next_first

        # Last criterion connects to confidence gate
        _, last_criterion_node = all_nodes[-1]
        last_criterion_node - "default" >> confidence_gate

        # Confidence gate routes to aggregator (all paths lead to aggregator)
        confidence_gate - "high_confidence" >> aggregator
        confidence_gate - "needs_review" >> aggregator
        confidence_gate - "low_confidence" >> aggregator

        first_node, _ = all_nodes[0]
        return Flow(start=first_node)

    def run(self, input_text: str) -> EvaluationResult:
        """Execute the workflow on input text."""
        shared = {
            "input_text": input_text,
            "parsed_policy": self.policy,
            "policy_context": self.policy.description,
            "criterion_results": {},
            "sub_criterion_results": {},
        }

        self.flow.run(shared)

        result = shared.get("evaluation_result")

        # Enhance result with confidence gate info
        if result and "confidence_gate_result" in shared:
            gate_result = shared["confidence_gate_result"]
            result = EvaluationResult(
                policy_satisfied=result.policy_satisfied,
                input_text=result.input_text,
                policy_title=result.policy_title,
                criterion_results=result.criterion_results,
                overall_reasoning=result.overall_reasoning,
                overall_confidence=result.overall_confidence,
                confidence_level=gate_result.get("confidence_level", ConfidenceLevel.MEDIUM),
                needs_review=gate_result.get("needs_review", False),
                low_confidence_criteria=gate_result.get("low_confidence_criteria", []),
            )

        return result

    def save(self, path: str | Path) -> None:
        """
        Save the parsed policy workflow to a YAML file.

        Args:
            path: Path to save the workflow YAML file
        """
        path = Path(path)
        data = self.policy.model_dump(mode="json")
        with path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: WorkflowConfig | None = None,
    ) -> "PolicyEvaluationWorkflow":
        """
        Load a workflow from a saved YAML file.

        Args:
            path: Path to the workflow YAML file
            config: Optional workflow configuration

        Returns:
            PolicyEvaluationWorkflow instance
        """
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)

        parsed_policy = ParsedPolicy.model_validate(data)
        return cls(parsed_policy=parsed_policy, config=config)
