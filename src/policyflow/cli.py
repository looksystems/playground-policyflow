"""
CLI interface for policy evaluation.

Usage:
    policy-eval eval --policy policy.md --input "text to evaluate"
    policy-eval eval --policy policy.md --input-file input.txt
    policy-eval eval --workflow workflow.yaml --input "text to evaluate"
    policy-eval parse --policy policy.md --save-workflow workflow.yaml
"""

from pathlib import Path
from typing import Annotated

import yaml
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from . import WorkflowConfig, EvaluationResult
from .parser import parse_policy_to_workflow
from .workflow_builder import DynamicWorkflowBuilder

app = typer.Typer(
    name="policy-eval",
    help="Generic policy evaluator using LLM-powered workflows",
)
console = Console()


def _build_result_from_shared(shared: dict, parsed) -> EvaluationResult:
    """Convert workflow shared store to EvaluationResult.

    The dynamic workflow stores results in the shared dict.
    This function extracts them into a structured EvaluationResult.
    """
    # Extract result from shared store - the workflow should have stored it
    if "result" in shared and isinstance(shared["result"], EvaluationResult):
        return shared["result"]

    # Otherwise build a simple result from available data
    from .nodes.criterion import CriterionResult

    # Check for common result keys
    policy_satisfied = shared.get("policy_satisfied", shared.get("satisfied", False))
    confidence = shared.get("confidence", shared.get("overall_confidence", 0.5))

    # Build criterion results from any stored evaluations
    criterion_results = []
    for key, value in shared.items():
        if key.endswith("_result") and isinstance(value, dict):
            criterion_results.append(CriterionResult(
                criterion_id=key.replace("_result", ""),
                criterion_name=key.replace("_result", "").replace("_", " ").title(),
                met=value.get("met", False),
                reasoning=value.get("reasoning", ""),
                confidence=value.get("confidence", 0.5),
            ))

    return EvaluationResult(
        policy_title=parsed.title,
        policy_satisfied=policy_satisfied,
        overall_confidence=confidence,
        confidence_level="high" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "low",
        needs_review=confidence < 0.8,
        criterion_results=criterion_results,
        overall_reasoning=shared.get("reasoning", "Evaluated using dynamic workflow"),
    )


@app.command("eval")
def eval_cmd(
    policy: Annotated[
        Path | None, typer.Option("--policy", "-p", help="Path to policy markdown")
    ] = None,
    workflow: Annotated[
        Path | None, typer.Option("--workflow", "-w", help="Path to pre-parsed workflow YAML")
    ] = None,
    input_text: Annotated[
        str | None, typer.Option("--input", "-i", help="Text to evaluate")
    ] = None,
    input_file: Annotated[
        Path | None, typer.Option("--input-file", "-f", help="File containing text")
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LiteLLM model identifier"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: pretty, yaml, or minimal"),
    ] = "pretty",
    save_workflow: Annotated[
        Path | None,
        typer.Option("--save-workflow", help="Save parsed workflow to YAML file"),
    ] = None,
):
    """Evaluate text against a policy."""
    if input_file:
        text = input_file.read_text()
    elif input_text:
        text = input_text
    else:
        raise typer.BadParameter("Must provide --input or --input-file")

    if not policy and not workflow:
        raise typer.BadParameter("Must provide --policy or --workflow")

    config = WorkflowConfig()

    with console.status("Evaluating..."):
        if workflow:
            # Load pre-parsed workflow from YAML
            from .models import ParsedWorkflowPolicy
            workflow_data = yaml.safe_load(workflow.read_text())
            parsed = ParsedWorkflowPolicy.model_validate(workflow_data)
            builder = DynamicWorkflowBuilder(parsed, config)
            shared = builder.run(text)
            result = _build_result_from_shared(shared, parsed)
        else:
            # Parse policy using dynamic workflow parser (with nodes)
            parsed = parse_policy_to_workflow(policy.read_text(), config, model=model)
            builder = DynamicWorkflowBuilder(parsed, config)
            shared = builder.run(text)
            result = _build_result_from_shared(shared, parsed)

        # Save workflow if requested
        if save_workflow:
            with save_workflow.open("w") as f:
                yaml.dump(parsed.model_dump(mode="json"), f, default_flow_style=False)
            console.print(f"[dim]Workflow saved to {save_workflow}[/dim]")

    if output_format == "yaml":
        console.print(yaml.dump(result.model_dump(mode="json"), default_flow_style=False, sort_keys=False))
    elif output_format == "minimal":
        status = "SATISFIED" if result.policy_satisfied else "NOT SATISFIED"
        console.print(f"{status} ({result.overall_confidence:.0%})")
    else:
        _print_pretty_result(result)


@app.command("parse")
def parse_cmd(
    policy: Annotated[
        Path, typer.Option("--policy", "-p", help="Path to policy markdown")
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LiteLLM model identifier"),
    ] = None,
    save_workflow: Annotated[
        Path | None,
        typer.Option("--save-workflow", help="Save parsed workflow to YAML file"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: pretty or yaml"),
    ] = "pretty",
):
    """Parse and display policy structure (using dynamic workflow parser)."""
    config = WorkflowConfig()

    with console.status("Parsing policy..."):
        parsed = parse_policy_to_workflow(policy.read_text(), config, model=model)

    # Save workflow if requested
    if save_workflow:
        with save_workflow.open("w") as f:
            yaml.dump(parsed.model_dump(mode="json"), f, default_flow_style=False)
        console.print(f"[green]Workflow saved to {save_workflow}[/green]")

    if output_format == "yaml":
        console.print(yaml.dump(parsed.model_dump(mode="json"), default_flow_style=False, sort_keys=False))
    else:
        console.print(Panel(f"[bold]{parsed.title}[/bold]\n{parsed.description}"))

        # Show workflow nodes
        console.print("\n[bold]Workflow Nodes:[/bold]")
        for node in parsed.workflow.nodes:
            routes = ", ".join(f"{k}→{v}" for k, v in node.routes.items())
            console.print(f"  [{node.id}] [cyan]{node.type}[/cyan]")
            if node.params:
                params_str = str(node.params)
                if len(params_str) > 60:
                    params_str = params_str[:60] + "..."
                console.print(f"       params: {params_str}")
            if routes:
                console.print(f"       routes: {routes}")

        console.print(f"\n[dim]Start node: {parsed.workflow.start_node}[/dim]")


@app.command("batch")
def batch_cmd(
    policy: Annotated[
        Path | None, typer.Option("--policy", "-p", help="Path to policy markdown")
    ] = None,
    workflow: Annotated[
        Path | None, typer.Option("--workflow", "-w", help="Path to pre-parsed workflow YAML")
    ] = None,
    inputs: Annotated[
        Path | None, typer.Option("--inputs", help="YAML file with inputs list")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output YAML file")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="LiteLLM model identifier")
    ] = None,
):
    """Batch evaluate multiple inputs."""
    if not policy and not workflow:
        raise typer.BadParameter("Must provide --policy or --workflow")
    if not inputs or not output:
        raise typer.BadParameter("Must provide --inputs and --output")

    config = WorkflowConfig()

    # Load workflow
    if workflow:
        from .models import ParsedWorkflowPolicy
        workflow_data = yaml.safe_load(workflow.read_text())
        parsed = ParsedWorkflowPolicy.model_validate(workflow_data)
    else:
        parsed = parse_policy_to_workflow(policy.read_text(), config, model=model)

    builder = DynamicWorkflowBuilder(parsed, config)

    # Load inputs from YAML
    with inputs.open() as f:
        input_data = yaml.safe_load(f)

    # Support both list of strings and list of dicts
    items = []
    if isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, str):
                items.append(item)
            elif isinstance(item, dict):
                items.append(item.get("text") or item.get("input"))

    results = []
    with console.status(f"Evaluating {len(items)} inputs...") as status:
        for i, text in enumerate(items, 1):
            status.update(f"Evaluating {i}/{len(items)}...")
            shared = builder.run(text)
            result = _build_result_from_shared(shared, parsed)
            results.append({"input": text, "result": result.model_dump(mode="json")})

    with output.open("w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    console.print(f"[green]Saved {len(results)} results to {output}[/green]")


# ============================================================================
# Two-Step Parser Commands
# ============================================================================


@app.command("normalize")
def normalize_cmd(
    policy: Annotated[
        Path, typer.Option("--policy", "-p", help="Path to policy markdown")
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output YAML file path")
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LiteLLM model identifier"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: pretty or yaml"),
    ] = "yaml",
):
    """Step 1: Normalize policy document into structured YAML.

    Creates a normalized representation of the policy with
    hierarchical numbering (1, 1.1, 1.1.a style) that can be
    reviewed before workflow generation.
    """
    from .parser import normalize_policy

    config = WorkflowConfig()

    with console.status("Normalizing policy..."):
        normalized = normalize_policy(policy.read_text(), config, model=model)

    # Save to output file
    normalized.save_yaml(output)
    console.print(f"[green]Normalized policy saved to {output}[/green]")

    if output_format == "yaml":
        console.print(normalized.to_yaml())
    else:
        _print_normalized_structure(normalized)


@app.command("generate-workflow")
def generate_workflow_cmd(
    normalized: Annotated[
        Path, typer.Option("--normalized", "-n", help="Path to normalized policy YAML")
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output workflow YAML file")
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LiteLLM model identifier"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: pretty or yaml"),
    ] = "yaml",
):
    """Step 2: Generate workflow from normalized policy.

    Takes a normalized policy YAML and generates an executable workflow
    with node IDs matching clause numbers for explainability.
    """
    from .models import NormalizedPolicy
    from .parser import generate_workflow_from_normalized

    config = WorkflowConfig()

    with console.status("Loading normalized policy..."):
        normalized_policy = NormalizedPolicy.load_yaml(normalized)

    with console.status("Generating workflow..."):
        workflow = generate_workflow_from_normalized(
            normalized_policy,
            config,
            model=model,
            normalized_policy_path=str(normalized),
        )

    workflow.save_yaml(output)
    console.print(f"[green]Workflow saved to {output}[/green]")

    if output_format == "yaml":
        console.print(workflow.to_yaml())
    else:
        _print_workflow_hierarchy(workflow)


@app.command("parse-two-step")
def parse_two_step_cmd(
    policy: Annotated[
        Path, typer.Option("--policy", "-p", help="Path to policy markdown")
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-d", help="Output directory for artifacts")
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LiteLLM model identifier"),
    ] = None,
    prefix: Annotated[
        str,
        typer.Option("--prefix", help="Filename prefix for outputs"),
    ] = "policy",
):
    """Complete two-step parsing: normalize then generate workflow.

    Creates both normalized policy and workflow YAML files in the
    output directory.
    """
    from .parser import normalize_policy, generate_workflow_from_normalized

    config = WorkflowConfig()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = output_dir / f"{prefix}_normalized.yaml"
    workflow_path = output_dir / f"{prefix}_workflow.yaml"

    with console.status("Step 1: Normalizing policy..."):
        normalized = normalize_policy(policy.read_text(), config, model=model)
        normalized.save_yaml(normalized_path)
        console.print(f"  [dim]Saved: {normalized_path}[/dim]")

    with console.status("Step 2: Generating workflow..."):
        workflow = generate_workflow_from_normalized(
            normalized, config, model=model,
            normalized_policy_path=str(normalized_path),
        )
        workflow.save_yaml(workflow_path)
        console.print(f"  [dim]Saved: {workflow_path}[/dim]")

    console.print(f"\n[green]Two-step parsing complete![/green]")
    console.print(f"  Normalized: {normalized_path}")
    console.print(f"  Workflow:   {workflow_path}")


def _print_normalized_structure(normalized):
    """Pretty print normalized policy structure."""
    from .models import Clause

    console.print(Panel(f"[bold]{normalized.title}[/bold]\n{normalized.description}"))

    def print_clause(clause: Clause, indent: int = 0):
        prefix = "  " * indent
        logic_str = f" [{clause.logic}]" if clause.logic else ""
        console.print(
            f"{prefix}[cyan]{clause.number}[/cyan] {clause.title}{logic_str}"
        )
        text_display = clause.text[:60] + "..." if len(clause.text) > 60 else clause.text
        console.print(f"{prefix}  [dim]{text_display}[/dim]")
        for sub in clause.sub_clauses:
            print_clause(sub, indent + 1)

    for section in normalized.sections:
        console.print(f"\n[bold]Section {section.number}: {section.title}[/bold]")
        for clause in section.clauses:
            print_clause(clause, 1)


def _print_workflow_hierarchy(workflow):
    """Pretty print workflow with hierarchy."""
    console.print(Panel(f"[bold]{workflow.title}[/bold]\n{workflow.description}"))

    console.print("\n[bold]Node Hierarchy:[/bold]")

    def print_group(group, indent: int = 0):
        prefix = "  " * indent
        logic_str = f" [{group.logic}]" if group.logic else ""
        console.print(f"{prefix}[cyan]{group.clause_number}[/cyan]{logic_str}")
        console.print(f"{prefix}  Nodes: {', '.join(group.nodes)}")
        for sub in group.sub_groups:
            print_group(sub, indent + 1)

    for group in workflow.workflow.hierarchy:
        print_group(group, 1)

    console.print(f"\n[dim]Start node: {workflow.workflow.start_node}[/dim]")
    console.print(f"[dim]Total nodes: {len(workflow.workflow.nodes)}[/dim]")


def _print_pretty_result(result: EvaluationResult):
    """Pretty print evaluation result."""
    status_color = "green" if result.policy_satisfied else "red"
    status = "SATISFIED" if result.policy_satisfied else "NOT SATISFIED"

    console.print(
        Panel(
            f"[{status_color} bold]{status}[/] "
            f"(confidence: {result.overall_confidence:.0%})",
            title=f"Policy: {result.policy_title}",
        )
    )

    table = Table(title="Criterion Results")
    table.add_column("Criterion", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Reasoning")

    for cr in result.criterion_results:
        status_str = "[green]MET[/]" if cr.met else "[red]NOT MET[/]"
        reasoning = cr.reasoning
        if len(reasoning) > 50:
            reasoning = reasoning[:50] + "..."
        table.add_row(
            cr.criterion_name,
            status_str,
            f"{cr.confidence:.0%}",
            reasoning,
        )

    console.print(table)
    console.print(f"\n[bold]Overall:[/bold] {result.overall_reasoning}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
