"""Multi-sample consensus evaluation node for PocketFlow."""

from typing import Literal

from pydantic import BaseModel, Field

from .llm_node import LLMNode
from .schema import NodeParameter, NodeSchema
from ..config import WorkflowConfig


class SampleResults(BaseModel):
    """Result from SamplerNode."""

    individual_results: list[bool] = Field(description="Results from each sample")
    aggregated_result: bool = Field(description="Final aggregated result")
    agreement_ratio: float = Field(
        ge=0.0, le=1.0, description="Ratio of samples that agree"
    )
    action: str = Field(description="Action: consensus, majority, or split")


class SamplerNode(LLMNode):
    """
    Runs evaluation N times and aggregates for consensus.
    This is an LLM-based node that uses temperature > 0 to get diverse samples.

    Shared Store:
        Reads: shared["input_text"] (or configurable key)
        Writes: shared["sample_results"] with individual results, aggregated result, agreement_ratio
    """

    parser_schema = NodeSchema(
        name="SamplerNode",
        description="Run evaluation N times and aggregate for consensus (LLM with sampling)",
        category="llm",
        parameters=[
            NodeParameter(
                "n_samples",
                "int",
                "Number of times to run the evaluation",
                required=True,
            ),
            NodeParameter(
                "aggregation",
                "str",
                "'majority' (>50% agree), 'unanimous' (100% agree), or 'any' (at least one True)",
                required=True,
            ),
            NodeParameter(
                "inner_prompt",
                "str",
                "Evaluation prompt to run multiple times (should return result: true/false)",
                required=True,
            ),
            NodeParameter(
                "system_prompt",
                "str",
                "Optional system prompt for the evaluation",
                required=False,
                default=None,
            ),
        ],
        actions=["consensus", "majority", "split"],
        yaml_example="""- type: SamplerNode
  id: consensus_check
  params:
    n_samples: 5
    aggregation: majority
    inner_prompt: "Is this content appropriate? Answer with result: true/false"
  routes:
    consensus: confident_decision
    majority: likely_decision
    split: needs_human_review""",
        parser_exposed=True,
    )

    def __init__(
        self,
        n_samples: int,
        aggregation: Literal["majority", "unanimous", "any"],
        inner_prompt: str,
        config: WorkflowConfig,
        model: str | None = None,
        input_key: str = "input_text",
        system_prompt: str | None = None,
        cache_ttl: int = 0,  # Default: disable caching for sampling
        rate_limit: int | None = None,
    ):
        """
        Initialize sampler node.

        Args:
            n_samples: Number of times to run the evaluation
            aggregation: How to combine results:
                - "majority": True if > 50% agree
                - "unanimous": True only if 100% agree
                - "any": True if any sample returns True
            inner_prompt: The evaluation prompt to run multiple times
            config: Workflow configuration
            model: LLM model identifier (uses class default_model if not provided)
            input_key: Key to read from shared store (default: "input_text")
            system_prompt: Optional system prompt for the LLM
            cache_ttl: Cache TTL in seconds (default: 0 = disabled for sampling)
            rate_limit: Requests per minute limit (default: None = unlimited)
        """
        super().__init__(config=config, model=model, cache_ttl=cache_ttl, rate_limit=rate_limit)
        self.n_samples = n_samples
        self.aggregation = aggregation
        self.inner_prompt = inner_prompt
        self.input_key = input_key
        self.system_prompt = system_prompt

    def prep(self, shared: dict) -> dict:
        """Retrieve input from shared store."""
        return {
            "input_text": shared.get(self.input_key, ""),
        }

    def exec(self, prep_res: dict) -> dict:
        """Run evaluation N times and aggregate results."""
        input_text = prep_res["input_text"]

        # Format the prompt with input text
        prompt = f"{self.inner_prompt}\n\nText to evaluate:\n\n{input_text}"

        # Collect individual sample results
        individual_results = []

        for i in range(self.n_samples):
            # Use temperature > 0 to get diverse samples
            # Note: We'll modify the config temporarily to use higher temperature
            original_temp = self.config.temperature
            self.config.temperature = max(0.7, original_temp)  # At least 0.7 for diversity

            try:
                # Call LLM and expect a YAML response with 'result' boolean field
                response = self.call_llm(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    yaml_response=True,
                    span_name=f"sampler_sample_{i+1}",
                )

                # Extract boolean result (expected format: {result: true/false, reasoning: "..."})
                sample_result = response.get("result", False)
                sample_reasoning = response.get("reasoning", "")

                individual_results.append({
                    "sample_id": i + 1,
                    "result": bool(sample_result),
                    "reasoning": sample_reasoning,
                })
            finally:
                # Restore original temperature
                self.config.temperature = original_temp

        # Aggregate results
        true_count = sum(1 for r in individual_results if r["result"])
        false_count = self.n_samples - true_count

        # Calculate agreement ratio (what % agree with the majority/winning side)
        max_agreement_count = max(true_count, false_count)
        agreement_ratio = max_agreement_count / self.n_samples

        # Determine aggregated result based on mode
        if self.aggregation == "majority":
            aggregated_result = true_count > (self.n_samples / 2)
        elif self.aggregation == "unanimous":
            aggregated_result = true_count == self.n_samples
        elif self.aggregation == "any":
            aggregated_result = true_count > 0
        else:
            # Fallback to majority
            aggregated_result = true_count > (self.n_samples / 2)

        # Determine action based on agreement
        if agreement_ratio == 1.0:
            action = "consensus"  # All samples agree
        elif agreement_ratio > 0.5:
            action = "majority"  # Clear majority but not unanimous
        else:
            action = "split"  # 50/50 or very close, no clear majority

        return {
            "individual_results": individual_results,
            "aggregated_result": aggregated_result,
            "agreement_ratio": agreement_ratio,
            "true_count": true_count,
            "false_count": false_count,
            "action": action,
        }

    def post(self, shared: dict, prep_res: dict, exec_res: dict) -> str:
        """Store sample results in shared store and return action."""
        shared["sample_results"] = {
            "individual_results": exec_res["individual_results"],
            "aggregated_result": exec_res["aggregated_result"],
            "agreement_ratio": exec_res["agreement_ratio"],
            "true_count": exec_res["true_count"],
            "false_count": exec_res["false_count"],
            "n_samples": self.n_samples,
            "aggregation_mode": self.aggregation,
        }

        return exec_res["action"]
