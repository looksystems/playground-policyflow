"""Tests for the hypothesis applier."""

from __future__ import annotations

import pytest

from policyflow.benchmark.models import Hypothesis
from policyflow.models import (
    HierarchicalWorkflowDefinition,
    NodeConfig,
    NodeGroup,
    ParsedWorkflowPolicy,
)


def create_test_workflow() -> ParsedWorkflowPolicy:
    """Create a minimal test workflow."""
    return ParsedWorkflowPolicy(
        title="Test Policy",
        description="Test workflow for applier tests",
        workflow=HierarchicalWorkflowDefinition(
            nodes=[
                NodeConfig(
                    id="node_1",
                    type="LLMEvaluatorNode",
                    params={
                        "prompt": "Evaluate the content",
                        "confidence_threshold": 0.7,
                    },
                    routes={"complete": "node_2"},
                ),
                NodeConfig(
                    id="node_2",
                    type="AggregatorNode",
                    params={"strategy": "all"},
                    routes={"complete": "end"},
                ),
            ],
            start_node="node_1",
            hierarchy=[
                NodeGroup(
                    clause_number="1",
                    clause_text="Test clause",
                    nodes=["node_1", "node_2"],
                )
            ],
        ),
    )


class TestBasicHypothesisApplier:
    """Tests for the basic hypothesis applier."""

    def test_applier_initialization(self):
        """Test applier can be initialized."""
        from policyflow.benchmark.applier import BasicHypothesisApplier

        applier = BasicHypothesisApplier()
        assert applier is not None

    def test_apply_node_param_change(self):
        """Test applying a node parameter change."""
        from policyflow.benchmark.applier import BasicHypothesisApplier

        applier = BasicHypothesisApplier()
        workflow = create_test_workflow()

        hypothesis = Hypothesis(
            id="hyp_001",
            description="Lower confidence threshold",
            change_type="node_param",
            target="node_1",
            suggested_change={"confidence_threshold": 0.5},
            rationale="More permissive threshold",
            expected_impact="Better recall",
        )

        modified = applier.apply(workflow, hypothesis)

        # Verify the original was not mutated
        assert workflow.workflow.nodes[0].params["confidence_threshold"] == 0.7

        # Verify the modified workflow has the change
        node_1 = next(n for n in modified.workflow.nodes if n.id == "node_1")
        assert node_1.params["confidence_threshold"] == 0.5

    def test_apply_threshold_change(self):
        """Test applying a threshold change."""
        from policyflow.benchmark.applier import BasicHypothesisApplier

        applier = BasicHypothesisApplier()
        workflow = create_test_workflow()

        hypothesis = Hypothesis(
            id="hyp_002",
            description="Increase threshold",
            change_type="threshold",
            target="node_1",
            suggested_change={"threshold": 0.85},
            rationale="Stricter threshold",
            expected_impact="Better precision",
        )

        modified = applier.apply(workflow, hypothesis)

        node_1 = next(n for n in modified.workflow.nodes if n.id == "node_1")
        assert node_1.params["threshold"] == 0.85

    def test_apply_prompt_tuning(self):
        """Test applying a prompt tuning change."""
        from policyflow.benchmark.applier import BasicHypothesisApplier

        applier = BasicHypothesisApplier()
        workflow = create_test_workflow()

        new_prompt = "Carefully evaluate the content for compliance"
        hypothesis = Hypothesis(
            id="hyp_003",
            description="Improve prompt clarity",
            change_type="prompt_tuning",
            target="node_1",
            suggested_change={"prompt": new_prompt},
            rationale="Clearer instructions",
            expected_impact="More accurate evaluations",
        )

        modified = applier.apply(workflow, hypothesis)

        node_1 = next(n for n in modified.workflow.nodes if n.id == "node_1")
        assert node_1.params["prompt"] == new_prompt

    def test_apply_workflow_structure_add_node(self):
        """Test applying a workflow structure change to add a node."""
        from policyflow.benchmark.applier import BasicHypothesisApplier

        applier = BasicHypothesisApplier()
        workflow = create_test_workflow()
        original_node_count = len(workflow.workflow.nodes)

        hypothesis = Hypothesis(
            id="hyp_004",
            description="Add confidence gate",
            change_type="workflow_structure",
            target="workflow",
            suggested_change={
                "add_node": {
                    "id": "confidence_gate",
                    "type": "ConfidenceGateNode",
                    "params": {"high_threshold": 0.85, "low_threshold": 0.6},
                    "after": "node_1",
                }
            },
            rationale="Route uncertain predictions",
            expected_impact="Better handling of edge cases",
        )

        modified = applier.apply(workflow, hypothesis)

        # Should have one more node
        assert len(modified.workflow.nodes) == original_node_count + 1

        # The new node should exist
        node_ids = [n.id for n in modified.workflow.nodes]
        assert "confidence_gate" in node_ids

    def test_apply_returns_new_workflow(self):
        """Test that apply returns a new workflow, not mutating the original."""
        from policyflow.benchmark.applier import BasicHypothesisApplier

        applier = BasicHypothesisApplier()
        workflow = create_test_workflow()

        hypothesis = Hypothesis(
            id="hyp_005",
            description="Test change",
            change_type="node_param",
            target="node_1",
            suggested_change={"new_param": "value"},
            rationale="Test",
            expected_impact="Test",
        )

        modified = applier.apply(workflow, hypothesis)

        # They should be different objects
        assert modified is not workflow
        assert modified.workflow is not workflow.workflow

    def test_apply_preserves_other_nodes(self):
        """Test that applying changes preserves nodes not targeted."""
        from policyflow.benchmark.applier import BasicHypothesisApplier

        applier = BasicHypothesisApplier()
        workflow = create_test_workflow()

        hypothesis = Hypothesis(
            id="hyp_006",
            description="Change node_1 only",
            change_type="node_param",
            target="node_1",
            suggested_change={"new_param": "value"},
            rationale="Test",
            expected_impact="Test",
        )

        modified = applier.apply(workflow, hypothesis)

        # node_2 should be unchanged
        original_node_2 = next(n for n in workflow.workflow.nodes if n.id == "node_2")
        modified_node_2 = next(n for n in modified.workflow.nodes if n.id == "node_2")

        assert original_node_2.params == modified_node_2.params

    def test_apply_nonexistent_target_raises(self):
        """Test that applying to nonexistent target raises error."""
        from policyflow.benchmark.applier import BasicHypothesisApplier

        applier = BasicHypothesisApplier()
        workflow = create_test_workflow()

        hypothesis = Hypothesis(
            id="hyp_007",
            description="Target nonexistent node",
            change_type="node_param",
            target="nonexistent_node",
            suggested_change={"param": "value"},
            rationale="Test",
            expected_impact="Test",
        )

        with pytest.raises(ValueError, match="not found"):
            applier.apply(workflow, hypothesis)


class TestApplierEdgeCases:
    """Test edge cases for the hypothesis applier."""

    def test_apply_empty_suggested_change(self):
        """Test applying with empty suggested_change does nothing."""
        from policyflow.benchmark.applier import BasicHypothesisApplier

        applier = BasicHypothesisApplier()
        workflow = create_test_workflow()

        hypothesis = Hypothesis(
            id="hyp_008",
            description="No-op change",
            change_type="node_param",
            target="node_1",
            suggested_change={},
            rationale="Test",
            expected_impact="None",
        )

        modified = applier.apply(workflow, hypothesis)

        # Should be unchanged
        original_node_1 = next(n for n in workflow.workflow.nodes if n.id == "node_1")
        modified_node_1 = next(n for n in modified.workflow.nodes if n.id == "node_1")

        assert original_node_1.params == modified_node_1.params

    def test_apply_multiple_param_changes(self):
        """Test applying multiple parameter changes at once."""
        from policyflow.benchmark.applier import BasicHypothesisApplier

        applier = BasicHypothesisApplier()
        workflow = create_test_workflow()

        hypothesis = Hypothesis(
            id="hyp_009",
            description="Multiple changes",
            change_type="node_param",
            target="node_1",
            suggested_change={
                "confidence_threshold": 0.5,
                "new_param": "new_value",
            },
            rationale="Test",
            expected_impact="Test",
        )

        modified = applier.apply(workflow, hypothesis)

        node_1 = next(n for n in modified.workflow.nodes if n.id == "node_1")
        assert node_1.params["confidence_threshold"] == 0.5
        assert node_1.params["new_param"] == "new_value"


class TestApplierFactory:
    """Tests for applier factory function."""

    def test_create_basic_applier(self):
        """Test creating basic applier."""
        from policyflow.benchmark.applier import BasicHypothesisApplier, create_applier

        applier = create_applier()
        assert isinstance(applier, BasicHypothesisApplier)


class TestApplierProtocolConformance:
    """Test that applier conforms to the HypothesisApplier protocol."""

    def test_satisfies_protocol(self):
        """Test that BasicHypothesisApplier satisfies the protocol."""
        from policyflow.benchmark.applier import BasicHypothesisApplier
        from policyflow.benchmark.protocols import HypothesisApplier

        applier = BasicHypothesisApplier()

        # Should have the apply method with correct signature
        assert hasattr(applier, "apply")
        assert callable(applier.apply)


class TestAddNodeRewiring:
    """Tests for the _add_node rewiring functionality."""

    def test_add_node_rewires_all_routes(self):
        """Test that adding a node correctly rewires all routes, not just 'complete'."""
        from policyflow.benchmark.applier import BasicHypothesisApplier

        applier = BasicHypothesisApplier()

        # Create a workflow with a node that has multiple routes
        workflow = ParsedWorkflowPolicy(
            title="Test Policy",
            description="Test workflow with multiple routes",
            workflow=HierarchicalWorkflowDefinition(
                nodes=[
                    NodeConfig(
                        id="classifier",
                        type="ClassifierNode",
                        params={"prompt": "Classify the content"},
                        routes={
                            "pass": "pass_handler",
                            "fail": "fail_handler",
                            "default": "default_handler",
                        },
                    ),
                    NodeConfig(
                        id="pass_handler",
                        type="HandlerNode",
                        params={},
                        routes={"complete": "end"},
                    ),
                    NodeConfig(
                        id="fail_handler",
                        type="HandlerNode",
                        params={},
                        routes={"complete": "end"},
                    ),
                    NodeConfig(
                        id="default_handler",
                        type="HandlerNode",
                        params={},
                        routes={"complete": "end"},
                    ),
                ],
                start_node="classifier",
                hierarchy=[
                    NodeGroup(
                        clause_number="1",
                        clause_text="Test clause",
                        nodes=["classifier", "pass_handler", "fail_handler", "default_handler"],
                    )
                ],
            ),
        )

        # Add a node after classifier, which should rewire the "pass" route
        hypothesis = Hypothesis(
            id="hyp_010",
            description="Add confidence gate after classifier",
            change_type="workflow_structure",
            target="workflow",
            suggested_change={
                "add_node": {
                    "id": "confidence_gate",
                    "type": "ConfidenceGateNode",
                    "params": {"threshold": 0.8},
                    "after": "classifier",
                    "intercept_route": "pass",  # Which route to intercept
                }
            },
            rationale="Add validation step",
            expected_impact="Better confidence handling",
        )

        modified = applier.apply(workflow, hypothesis)

        # The new node should be inserted
        node_ids = [n.id for n in modified.workflow.nodes]
        assert "confidence_gate" in node_ids

        # Find the classifier node
        classifier = next(n for n in modified.workflow.nodes if n.id == "classifier")

        # The classifier's "pass" route should now point to the new node
        assert classifier.routes["pass"] == "confidence_gate"

        # The new node should route to the original pass_handler
        confidence_gate = next(n for n in modified.workflow.nodes if n.id == "confidence_gate")
        assert confidence_gate.routes.get("complete") == "pass_handler"

    def test_add_node_with_default_route_complete(self):
        """Test that add_node defaults to 'complete' route when not specified."""
        from policyflow.benchmark.applier import BasicHypothesisApplier

        applier = BasicHypothesisApplier()
        workflow = create_test_workflow()

        hypothesis = Hypothesis(
            id="hyp_011",
            description="Add node after node_1",
            change_type="workflow_structure",
            target="workflow",
            suggested_change={
                "add_node": {
                    "id": "new_node",
                    "type": "ProcessingNode",
                    "params": {},
                    "after": "node_1",
                    # No intercept_route specified, should default to "complete"
                }
            },
            rationale="Add processing step",
            expected_impact="Better processing",
        )

        modified = applier.apply(workflow, hypothesis)

        # Find node_1
        node_1 = next(n for n in modified.workflow.nodes if n.id == "node_1")

        # node_1's "complete" route should now point to new_node
        assert node_1.routes["complete"] == "new_node"

        # new_node should route to node_2 (original target)
        new_node = next(n for n in modified.workflow.nodes if n.id == "new_node")
        assert new_node.routes.get("complete") == "node_2"
