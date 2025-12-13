"""Hypothesis applier for modifying workflows based on improvement hypotheses.

Applies hypothesis changes to workflows, creating modified versions
for testing improvement strategies.
"""

from __future__ import annotations

import copy
from typing import Any

from policyflow.benchmark.models import Hypothesis
from policyflow.models import (
    HierarchicalWorkflowDefinition,
    NodeConfig,
    ParsedWorkflowPolicy,
)


class BasicHypothesisApplier:
    """Applies structured hypothesis changes to workflows.

    Handles different change types:
    - node_param: Update node parameters
    - prompt_tuning: Update prompt templates
    - workflow_structure: Add/remove/rewire nodes
    - threshold: Update confidence/gate thresholds
    """

    def apply(
        self,
        workflow: ParsedWorkflowPolicy,
        hypothesis: Hypothesis,
    ) -> ParsedWorkflowPolicy:
        """Apply a hypothesis to create a modified workflow.

        Args:
            workflow: The workflow to modify
            hypothesis: The hypothesis to apply

        Returns:
            Modified workflow (original is not mutated)

        Raises:
            ValueError: If the target node is not found
        """
        # Deep copy to avoid mutating the original
        modified = self._deep_copy_workflow(workflow)

        match hypothesis.change_type:
            case "node_param":
                self._apply_node_param(modified, hypothesis)
            case "prompt_tuning":
                self._apply_prompt_tuning(modified, hypothesis)
            case "workflow_structure":
                self._apply_workflow_structure(modified, hypothesis)
            case "threshold":
                self._apply_threshold(modified, hypothesis)
            case _:
                raise ValueError(f"Unknown change type: {hypothesis.change_type}")

        return modified

    def _deep_copy_workflow(
        self, workflow: ParsedWorkflowPolicy
    ) -> ParsedWorkflowPolicy:
        """Create a deep copy of the workflow."""
        # Use model_copy with deep=True for Pydantic models
        return workflow.model_copy(deep=True)

    def _find_node(
        self, workflow: ParsedWorkflowPolicy, node_id: str
    ) -> NodeConfig | None:
        """Find a node by ID in the workflow."""
        for node in workflow.workflow.nodes:
            if node.id == node_id:
                return node
        return None

    def _apply_node_param(
        self, workflow: ParsedWorkflowPolicy, hypothesis: Hypothesis
    ) -> None:
        """Apply node parameter changes."""
        node = self._find_node(workflow, hypothesis.target)
        if node is None:
            raise ValueError(f"Node '{hypothesis.target}' not found in workflow")

        # Apply all suggested changes to the node params
        for key, value in hypothesis.suggested_change.items():
            node.params[key] = value

    def _apply_prompt_tuning(
        self, workflow: ParsedWorkflowPolicy, hypothesis: Hypothesis
    ) -> None:
        """Apply prompt tuning changes."""
        node = self._find_node(workflow, hypothesis.target)
        if node is None:
            raise ValueError(f"Node '{hypothesis.target}' not found in workflow")

        # Update the prompt parameter
        if "prompt" in hypothesis.suggested_change:
            node.params["prompt"] = hypothesis.suggested_change["prompt"]
        else:
            # Apply all changes as params (backward compatibility)
            for key, value in hypothesis.suggested_change.items():
                node.params[key] = value

    def _apply_workflow_structure(
        self, workflow: ParsedWorkflowPolicy, hypothesis: Hypothesis
    ) -> None:
        """Apply workflow structure changes."""
        change = hypothesis.suggested_change

        if "add_node" in change:
            self._add_node(workflow, change["add_node"])

        if "remove_node" in change:
            self._remove_node(workflow, change["remove_node"])

        if "rewire" in change:
            self._rewire(workflow, change["rewire"])

    def _apply_threshold(
        self, workflow: ParsedWorkflowPolicy, hypothesis: Hypothesis
    ) -> None:
        """Apply threshold changes."""
        node = self._find_node(workflow, hypothesis.target)
        if node is None:
            raise ValueError(f"Node '{hypothesis.target}' not found in workflow")

        # Apply threshold-related changes
        for key, value in hypothesis.suggested_change.items():
            node.params[key] = value

    def _add_node(
        self, workflow: ParsedWorkflowPolicy, node_spec: dict[str, Any]
    ) -> None:
        """Add a new node to the workflow.

        Supports specifying which route to intercept via 'intercept_route'.
        Defaults to 'complete' for backward compatibility.
        """
        new_node = NodeConfig(
            id=node_spec["id"],
            type=node_spec["type"],
            params=node_spec.get("params", {}),
            routes=node_spec.get("routes", {}),
        )

        # Insert after specified node if provided
        after_node_id = node_spec.get("after")
        if after_node_id:
            # Determine which route to intercept (default to "complete" for backward compatibility)
            intercept_route = node_spec.get("intercept_route", "complete")

            # Find the node to insert after
            insert_index = None
            for i, node in enumerate(workflow.workflow.nodes):
                if node.id == after_node_id:
                    insert_index = i + 1
                    # Rewire the specified route to point to the new node
                    if intercept_route in node.routes:
                        old_target = node.routes[intercept_route]
                        node.routes[intercept_route] = new_node.id
                        # Set the new node to point to the old target via "complete"
                        new_node.routes["complete"] = old_target
                    break

            if insert_index is not None:
                workflow.workflow.nodes.insert(insert_index, new_node)
            else:
                # If after node not found, append
                workflow.workflow.nodes.append(new_node)
        else:
            workflow.workflow.nodes.append(new_node)

    def _remove_node(
        self, workflow: ParsedWorkflowPolicy, node_id: str
    ) -> None:
        """Remove a node from the workflow."""
        # Find the node to remove
        node_to_remove = None
        node_index = None
        for i, node in enumerate(workflow.workflow.nodes):
            if node.id == node_id:
                node_to_remove = node
                node_index = i
                break

        if node_to_remove is None:
            raise ValueError(f"Node '{node_id}' not found in workflow")

        # Find where this node routes to
        target = node_to_remove.routes.get("complete")

        # Update any nodes that route to this node
        for node in workflow.workflow.nodes:
            for action, target_id in node.routes.items():
                if target_id == node_id:
                    # Route to this node's target instead
                    node.routes[action] = target or "end"

        # Remove the node
        workflow.workflow.nodes.pop(node_index)

    def _rewire(
        self, workflow: ParsedWorkflowPolicy, rewire_spec: dict[str, Any]
    ) -> None:
        """Rewire node connections."""
        source_id = rewire_spec["source"]
        action = rewire_spec.get("action", "complete")
        new_target = rewire_spec["target"]

        node = self._find_node(workflow, source_id)
        if node is None:
            raise ValueError(f"Source node '{source_id}' not found")

        node.routes[action] = new_target


def create_applier() -> BasicHypothesisApplier:
    """Factory function to create a hypothesis applier.

    Returns:
        Configured applier instance
    """
    return BasicHypothesisApplier()
