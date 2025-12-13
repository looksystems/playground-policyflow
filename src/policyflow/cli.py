"""
CLI interface for policy evaluation.

Usage:
    policyflow parse --policy policy.md --save-workflow workflow.yaml
    policyflow parse --policy policy.md --save-normalized norm.yaml --save-workflow workflow.yaml
    policyflow eval --policy policy.md --input "text to evaluate"
    policyflow eval --workflow workflow.yaml --input "text to evaluate"
    policyflow batch --policy policy.md --inputs inputs.yaml --output results.yaml
"""

from pathlib import Path
from typing import Annotated

import yaml
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from . import WorkflowConfig, EvaluationResult
from .parser import parse_policy
from .workflow_builder import DynamicWorkflowBuilder

app = typer.Typer(
    name="policyflow",
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

    from .models import ClauseResult

    # Check for common result keys
    policy_satisfied = shared.get("policy_satisfied", shared.get("satisfied", False))
    confidence = shared.get("confidence", shared.get("overall_confidence", 0.5))

    # Build clause results from any stored evaluations
    clause_results = []
    for key, value in shared.items():
        if key.endswith("_result") and isinstance(value, dict):
            clause_results.append(ClauseResult(
                clause_id=key.replace("_result", ""),
                clause_name=key.replace("_result", "").replace("_", " ").title(),
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
        clause_results=clause_results,
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
            # Parse policy using two-step parser
            parsed = parse_policy(policy.read_text(), config, model=model)
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
    save_normalized: Annotated[
        Path | None,
        typer.Option("--save-normalized", help="Save intermediate normalized policy to YAML file"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: pretty or yaml"),
    ] = "pretty",
):
    """Parse policy into executable workflow (two-step parsing)."""
    config = WorkflowConfig()

    with console.status("Parsing policy..."):
        parsed = parse_policy(
            policy.read_text(),
            config,
            model=model,
            save_normalized=save_normalized,
        )

    if save_normalized:
        console.print(f"[dim]Normalized policy saved to {save_normalized}[/dim]")

    # Save workflow if requested
    if save_workflow:
        parsed.save_yaml(save_workflow)
        console.print(f"[green]Workflow saved to {save_workflow}[/green]")

    if output_format == "yaml":
        console.print(parsed.to_yaml())
    else:
        console.print(Panel(f"[bold]{parsed.title}[/bold]\n{parsed.description}"))

        # Show workflow hierarchy if available
        if parsed.workflow.hierarchy:
            console.print("\n[bold]Workflow Hierarchy:[/bold]")
            _print_workflow_hierarchy(parsed)
        else:
            # Fall back to flat node list
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
        console.print(f"[dim]Total nodes: {len(parsed.workflow.nodes)}[/dim]")


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
        parsed = parse_policy(policy.read_text(), config, model=model)

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

    if result.clause_results:
        table = Table(title="Clause Results")
        table.add_column("Clause", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Reasoning")

        for cr in result.clause_results:
            status_str = "[green]MET[/]" if cr.met else "[red]NOT MET[/]"
            reasoning = cr.reasoning
            if len(reasoning) > 50:
                reasoning = reasoning[:50] + "..."
            table.add_row(
                cr.clause_name,
                status_str,
                f"{cr.confidence:.0%}",
                reasoning,
            )

        console.print(table)

    console.print(f"\n[bold]Overall:[/bold] {result.overall_reasoning}")


def main():
    """Entry point for the CLI."""
    # Register benchmark commands
    from policyflow.benchmark.cli import register_benchmark_commands
    register_benchmark_commands(app)

    app()


if __name__ == "__main__":
    main()
