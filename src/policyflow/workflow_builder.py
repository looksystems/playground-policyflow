"""Dynamic workflow builder from parsed workflow policies."""

import warnings

from pocketflow import Flow, Node

from .models import ParsedWorkflowPolicy, NodeConfig
from .nodes.registry import get_node_class
from .config import WorkflowConfig


class DynamicWorkflowBuilder:
    """Builds a PocketFlow workflow from a ParsedWorkflowPolicy.

    This builder takes a workflow definition (parsed from a policy)
    and constructs an executable PocketFlow workflow by:
    1. Instantiating all configured nodes
    2. Wiring up routes between nodes based on actions
    3. Returning a Flow starting from the designated start node
    """

    def __init__(
        self,
        policy: ParsedWorkflowPolicy,
        config: WorkflowConfig | None = None,
    ):
        """Initialize the workflow builder.

        Args:
            policy: Parsed workflow policy with node configurations
            config: Optional workflow configuration (for LLM nodes)
        """
        self.policy = policy
        self.config = config or WorkflowConfig()

    def build(self) -> Flow:
        """Build and return the executable workflow.

        Returns:
            A PocketFlow Flow ready to execute

        Raises:
            ValueError: If node types are unknown or start_node is invalid
        """
        nodes: dict[str, Node] = {}

        # Phase 1: Instantiate all nodes
        for node_config in self.policy.workflow.nodes:
            node = self._create_node(node_config)
            nodes[node_config.id] = node

        # Phase 2: Wire up routes between nodes
        for node_config in self.policy.workflow.nodes:
            source = nodes[node_config.id]
            for action, target_id in node_config.routes.items():
                if target_id in nodes:
                    source - action >> nodes[target_id]
                # Note: If target_id is not in nodes, that route will
                # simply not be connected (could be an end state)

        # Validate workflow structure
        self._validate_workflow(nodes)

        # Phase 3: Create and return Flow
        start_node_id = self.policy.workflow.start_node
        if start_node_id not in nodes:
            raise ValueError(
                f"Start node '{start_node_id}' not found in workflow nodes"
            )

        return Flow(start=nodes[start_node_id])

    def _create_node(self, config: NodeConfig) -> Node:
        """Create a node instance from configuration.

        Args:
            config: Node configuration with type, params, and routes

        Returns:
            Instantiated node

        Raises:
            ValueError: If the node type is unknown
        """
        cls = get_node_class(config.type)
        if cls is None:
            raise ValueError(f"Unknown node type: {config.type}")

        # Copy params to avoid mutating the original config
        params = config.params.copy()

        # Check if this is an LLM-based node that needs config injection
        schema = getattr(cls, "parser_schema", None)
        if schema and schema.category == "llm":
            # LLM nodes need WorkflowConfig passed in
            params["config"] = self.config

        return cls(**params)

    def _validate_workflow(self, nodes: dict[str, Node]) -> None:
        """Validate workflow has terminal nodes and check for cycles.

        Args:
            nodes: Dictionary mapping node IDs to instantiated nodes
        """
        # Check for terminal nodes (nodes with empty routes)
        has_terminal = any(
            not node_config.routes for node_config in self.policy.workflow.nodes
        )
        if not has_terminal:
            warnings.warn(
                "Workflow has no terminal nodes (nodes with empty routes). "
                "This may cause infinite execution."
            )

        # Check for cycles using DFS
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            node_config = next(
                (n for n in self.policy.workflow.nodes if n.id == node_id), None
            )
            if node_config:
                for target_id in node_config.routes.values():
                    if target_id not in visited:
                        if has_cycle(target_id):
                            return True
                    elif target_id in rec_stack:
                        return True
            rec_stack.remove(node_id)
            return False

        start_id = self.policy.workflow.start_node
        if has_cycle(start_id):
            warnings.warn(
                "Workflow contains cycles. This may cause infinite execution."
            )

    def run(self, input_text: str, max_iterations: int = 100) -> dict:
        """Build workflow and run it with the given input.

        Convenience method that builds the workflow and executes it.

        Args:
            input_text: Text to process through the workflow
            max_iterations: Maximum node executions before raising error

        Returns:
            The shared store after workflow execution

        Raises:
            RuntimeError: If max_iterations is exceeded
        """
        flow = self.build()
        shared = {"input_text": input_text}

        # Wrap node _run methods to track iterations
        iteration_count = {"count": 0}
        nodes_to_patch = self._collect_all_nodes(flow.start)

        for node in nodes_to_patch:
            original_run = node._run

            def make_counted_run(orig):
                def counted_run(shared):
                    iteration_count["count"] += 1
                    if iteration_count["count"] > max_iterations:
                        raise RuntimeError(
                            f"Workflow exceeded {max_iterations} iterations. "
                            "Possible infinite loop detected."
                        )
                    return orig(shared)

                return counted_run

            node._run = make_counted_run(original_run)

        flow.run(shared)
        return shared

    def _collect_all_nodes(self, start_node: Node) -> list[Node]:
        """Collect all nodes reachable from the start node.

        Args:
            start_node: The starting node of the flow

        Returns:
            List of all reachable nodes
        """
        visited: set[int] = set()
        nodes: list[Node] = []

        def visit(node: Node) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            nodes.append(node)
            # Traverse all successors
            if hasattr(node, "successors") and node.successors:
                for successor in node.successors.values():
                    visit(successor)

        visit(start_node)
        return nodes


def build_workflow_from_policy(
    policy: ParsedWorkflowPolicy,
    config: WorkflowConfig | None = None,
) -> Flow:
    """Build a workflow from a parsed workflow policy.

    Convenience function for building workflows.

    Args:
        policy: Parsed workflow policy
        config: Optional workflow configuration

    Returns:
        Executable PocketFlow workflow
    """
    builder = DynamicWorkflowBuilder(policy, config)
    return builder.build()
