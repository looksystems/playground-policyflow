"""PocketFlow nodes for policy evaluation."""

from .schema import NodeSchema, NodeParameter
from .registry import register_node, get_node_class, get_parser_schemas, get_all_nodes

from .llm_node import LLMNode
from .criterion import CriterionEvaluationNode
from .aggregate import ResultAggregatorNode
from .subcriterion import SubCriterionNode
from .confidence_gate import ConfidenceGateNode
from .transform import TransformNode
from .length_gate import LengthGateNode
from .keyword_scorer import KeywordScorerNode
from .pattern_match import PatternMatchNode
from .data_extractor import DataExtractorNode
from .sampler import SamplerNode
from .classifier import ClassifierNode
from .sentiment import SentimentNode

# Register all nodes
register_node(LLMNode)
register_node(CriterionEvaluationNode)
register_node(ResultAggregatorNode)
register_node(SubCriterionNode)
register_node(ConfidenceGateNode)
register_node(TransformNode)
register_node(LengthGateNode)
register_node(KeywordScorerNode)
register_node(PatternMatchNode)
register_node(DataExtractorNode)
register_node(SamplerNode)
register_node(ClassifierNode)
register_node(SentimentNode)

__all__ = [
    # Schema
    "NodeSchema",
    "NodeParameter",
    # Registry
    "register_node",
    "get_node_class",
    "get_parser_schemas",
    "get_all_nodes",
    # Nodes
    "LLMNode",
    "CriterionEvaluationNode",
    "ResultAggregatorNode",
    "SubCriterionNode",
    "ConfidenceGateNode",
    "TransformNode",
    "LengthGateNode",
    "KeywordScorerNode",
    "PatternMatchNode",
    "DataExtractorNode",
    "SamplerNode",
    "ClassifierNode",
    "SentimentNode",
]
