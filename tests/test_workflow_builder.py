"""Tests for workflow builder, including infinite loop protection."""

import warnings

import pytest
from pocketflow import Node

from policyflow.workflow_builder import DynamicWorkflowBuilder
from policyflow.models import ParsedWorkflowPolicy, WorkflowDefinition, NodeConfig
from policyflow.nodes.registry import register_node


@register_node
class SimpleNode(Node):
    """Simple test node that returns a configurable action."""

    def __init__(self, action: str = "default"):
        super().__init__()
        self.action = action

    def prep(self, shared):
        return shared

    def exec(self, prep_res):
        return prep_res

    def post(self, shared, prep_res, exec_res):
        return self.action


class TestWorkflowValidation:
    """Tests for workflow validation."""

    def test_warns_on_no_terminal_nodes(self):
        """Should warn when workflow has no terminal nodes."""
        policy = ParsedWorkflowPolicy(
            title="Test",
            description="Test policy",
            workflow=WorkflowDefinition(
                nodes=[
                    NodeConfig(
                        id="node_a",
                        type="SimpleNode",
                        params={"action": "next"},
                        routes={"next": "node_b"},
                    ),
                    NodeConfig(
                        id="node_b",
                        type="SimpleNode",
                        params={"action": "next"},
                        routes={"next": "node_a"},  # Cycles back to A
                    ),
                ],
                start_node="node_a",
            ),
        )

        builder = DynamicWorkflowBuilder(policy)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.build()

            # Should have warnings about no terminal nodes and cycles
            warning_messages = [str(warning.message) for warning in w]
            assert any("terminal nodes" in msg for msg in warning_messages)
            assert any("cycles" in msg for msg in warning_messages)

    def test_warns_on_cycle(self):
        """Should warn when workflow contains a cycle."""
        policy = ParsedWorkflowPolicy(
            title="Test",
            description="Test policy",
            workflow=WorkflowDefinition(
                nodes=[
                    NodeConfig(
                        id="node_a",
                        type="SimpleNode",
                        params={"action": "next"},
                        routes={"next": "node_b"},
                    ),
                    NodeConfig(
                        id="node_b",
                        type="SimpleNode",
                        params={"action": "next"},
                        routes={"next": "node_c"},
                    ),
                    NodeConfig(
                        id="node_c",
                        type="SimpleNode",
                        params={"action": "back"},
                        routes={"back": "node_a"},  # Cycle
                    ),
                ],
                start_node="node_a",
            ),
        )

        builder = DynamicWorkflowBuilder(policy)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.build()

            warning_messages = [str(warning.message) for warning in w]
            assert any("cycles" in msg for msg in warning_messages)

    def test_no_warning_for_valid_workflow(self):
        """Should not warn when workflow has terminal nodes and no cycles."""
        policy = ParsedWorkflowPolicy(
            title="Test",
            description="Test policy",
            workflow=WorkflowDefinition(
                nodes=[
                    NodeConfig(
                        id="node_a",
                        type="SimpleNode",
                        params={"action": "next"},
                        routes={"next": "node_b"},
                    ),
                    NodeConfig(
                        id="node_b",
                        type="SimpleNode",
                        params={"action": "done"},
                        routes={},  # Terminal node
                    ),
                ],
                start_node="node_a",
            ),
        )

        builder = DynamicWorkflowBuilder(policy)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.build()

            # Filter for our specific warnings
            our_warnings = [
                warning
                for warning in w
                if "terminal" in str(warning.message) or "cycles" in str(warning.message)
            ]
            assert len(our_warnings) == 0


class TestMaxIterations:
    """Tests for max iterations safety."""

    def test_raises_on_max_iterations_exceeded(self):
        """Should raise RuntimeError when max iterations is exceeded."""
        policy = ParsedWorkflowPolicy(
            title="Test",
            description="Test policy",
            workflow=WorkflowDefinition(
                nodes=[
                    NodeConfig(
                        id="node_a",
                        type="SimpleNode",
                        params={"action": "next"},
                        routes={"next": "node_b"},
                    ),
                    NodeConfig(
                        id="node_b",
                        type="SimpleNode",
                        params={"action": "next"},
                        routes={"next": "node_a"},  # Cycles back - infinite loop
                    ),
                ],
                start_node="node_a",
            ),
        )

        builder = DynamicWorkflowBuilder(policy)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore validation warnings

            with pytest.raises(RuntimeError) as exc_info:
                builder.run("test input", max_iterations=10)

            assert "exceeded 10 iterations" in str(exc_info.value)
            assert "infinite loop" in str(exc_info.value).lower()

    def test_completes_within_max_iterations(self):
        """Should complete normally when within max iterations."""
        policy = ParsedWorkflowPolicy(
            title="Test",
            description="Test policy",
            workflow=WorkflowDefinition(
                nodes=[
                    NodeConfig(
                        id="node_a",
                        type="SimpleNode",
                        params={"action": "next"},
                        routes={"next": "node_b"},
                    ),
                    NodeConfig(
                        id="node_b",
                        type="SimpleNode",
                        params={"action": "done"},
                        routes={},  # Terminal node
                    ),
                ],
                start_node="node_a",
            ),
        )

        builder = DynamicWorkflowBuilder(policy)
        result = builder.run("test input", max_iterations=100)

        assert result["input_text"] == "test input"

    def test_custom_max_iterations(self):
        """Should respect custom max_iterations parameter."""
        policy = ParsedWorkflowPolicy(
            title="Test",
            description="Test policy",
            workflow=WorkflowDefinition(
                nodes=[
                    NodeConfig(
                        id="node_a",
                        type="SimpleNode",
                        params={"action": "next"},
                        routes={"next": "node_a"},  # Self-loop
                    ),
                ],
                start_node="node_a",
            ),
        )

        builder = DynamicWorkflowBuilder(policy)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Should fail at 5 iterations
            with pytest.raises(RuntimeError) as exc_info:
                builder.run("test input", max_iterations=5)

            assert "exceeded 5 iterations" in str(exc_info.value)
