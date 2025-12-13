"""CLI commands for the benchmark system.

Usage:
    policyflow benchmark --workflow workflow.yaml --dataset golden_dataset.yaml
    policyflow analyze --report report.yaml --workflow workflow.yaml
    policyflow experiments list
    policyflow experiments best
    policyflow experiments compare exp_001 exp_002
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from policyflow.benchmark.analyzer import create_analyzer
from policyflow.benchmark.hypothesis import create_hypothesis_generator
from policyflow.benchmark.loader import load_golden_dataset
from policyflow.benchmark.models import (
    BenchmarkReport,
    Experiment,
    GoldenDataset,
)
from policyflow.benchmark.runner import BenchmarkConfig, SimpleBenchmarkRunner
from policyflow.benchmark.tracker import FileBasedExperimentTracker
from policyflow.config import WorkflowConfig
from policyflow.models import ParsedWorkflowPolicy

console = Console()

# ============================================================================
# Benchmark Command
# ============================================================================

benchmark_app = typer.Typer(
    name="benchmark",
    help="Run benchmark against golden dataset",
)


@benchmark_app.callback(invoke_without_command=True)
def benchmark_cmd(
    ctx: typer.Context,
    workflow: Annotated[
        Path,
        typer.Option("--workflow", "-w", help="Path to workflow YAML"),
    ],
    dataset: Annotated[
        Path,
        typer.Option("--dataset", "-d", help="Path to golden dataset YAML"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Path to save benchmark report"),
    ] = None,
    workflow_id: Annotated[
        str | None,
        typer.Option("--id", help="Workflow version identifier"),
    ] = None,
    category: Annotated[
        str | None,
        typer.Option("--category", "-c", help="Filter test cases by category"),
    ] = None,
):
    """Run benchmark against golden dataset."""
    if ctx.invoked_subcommand is not None:
        return

    # Load workflow
    console.print(f"[dim]Loading workflow from {workflow}...[/dim]")
    with workflow.open() as f:
        workflow_data = yaml.safe_load(f)
    parsed_workflow = ParsedWorkflowPolicy.model_validate(workflow_data)

    # Load dataset
    console.print(f"[dim]Loading dataset from {dataset}...[/dim]")
    golden_dataset = load_golden_dataset(dataset)
    test_cases = golden_dataset.test_cases

    # Apply category filter if specified
    if category:
        test_cases = golden_dataset.filter_by_category(category)
        console.print(f"[dim]Filtered to {len(test_cases)} test cases in '{category}' category[/dim]")

    # Configure and run benchmark
    config = BenchmarkConfig(
        workflow_id=workflow_id or f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    runner = SimpleBenchmarkRunner(config)

    with console.status(f"Running benchmark on {len(test_cases)} test cases..."):
        report = runner.run(parsed_workflow, test_cases)

    # Display results
    _print_benchmark_summary(report)

    # Save report if requested
    if output:
        report.save_yaml(output)
        console.print(f"\n[green]Report saved to {output}[/green]")


def _print_benchmark_summary(report: BenchmarkReport):
    """Print benchmark summary."""
    metrics = report.metrics

    # Overall panel
    accuracy_color = "green" if metrics.overall_accuracy >= 0.9 else "yellow" if metrics.overall_accuracy >= 0.7 else "red"
    console.print(
        Panel(
            f"[{accuracy_color} bold]{metrics.overall_accuracy:.1%}[/] overall accuracy\n"
            f"[dim]{len(report.results)} test cases • {len(report.failures)} failures • {len(report.errors)} errors[/dim]",
            title=f"Benchmark Report: {report.workflow_id}",
        )
    )

    # Category breakdown
    if metrics.category_accuracy:
        table = Table(title="Category Breakdown")
        table.add_column("Category", style="cyan")
        table.add_column("Accuracy", justify="right")

        for cat, acc in sorted(metrics.category_accuracy.items()):
            acc_color = "green" if acc >= 0.9 else "yellow" if acc >= 0.7 else "red"
            table.add_row(cat, f"[{acc_color}]{acc:.1%}[/]")

        console.print(table)

    # Criterion metrics
    if metrics.criterion_metrics:
        table = Table(title="Per-Criterion Metrics")
        table.add_column("Criterion", style="cyan")
        table.add_column("Accuracy", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1", justify="right")

        for crit_id, crit_metrics in sorted(metrics.criterion_metrics.items()):
            table.add_row(
                crit_id,
                f"{crit_metrics.accuracy:.1%}",
                f"{crit_metrics.precision:.1%}",
                f"{crit_metrics.recall:.1%}",
                f"{crit_metrics.f1:.2f}",
            )

        console.print(table)

    # Failures preview
    if report.failures:
        console.print(f"\n[yellow]Failed test cases ({len(report.failures)}):[/yellow]")
        for failure in report.failures[:5]:  # Show first 5
            console.print(f"  [dim]•[/dim] {failure.test_id}")
        if len(report.failures) > 5:
            console.print(f"  [dim]... and {len(report.failures) - 5} more[/dim]")


# ============================================================================
# Analyze Command
# ============================================================================

analyze_app = typer.Typer(
    name="analyze",
    help="Analyze benchmark failures",
)


@analyze_app.callback(invoke_without_command=True)
def analyze_cmd(
    ctx: typer.Context,
    report: Annotated[
        Path,
        typer.Option("--report", "-r", help="Path to benchmark report YAML"),
    ],
    workflow: Annotated[
        Path,
        typer.Option("--workflow", "-w", help="Path to workflow YAML"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Path to save analysis report"),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="Analysis mode: rule_based, llm, or hybrid"),
    ] = "hybrid",
):
    """Analyze benchmark failures to identify patterns."""
    if ctx.invoked_subcommand is not None:
        return

    # Load report and workflow
    console.print(f"[dim]Loading report from {report}...[/dim]")
    benchmark_report = BenchmarkReport.load_yaml(report)

    console.print(f"[dim]Loading workflow from {workflow}...[/dim]")
    with workflow.open() as f:
        workflow_data = yaml.safe_load(f)
    parsed_workflow = ParsedWorkflowPolicy.model_validate(workflow_data)

    # Analyze
    analyzer = create_analyzer(mode=mode)

    with console.status("Analyzing failures..."):
        analysis = analyzer.analyze(benchmark_report, parsed_workflow)

    # Display results
    _print_analysis_summary(analysis)

    # Save if requested
    if output:
        analysis.save_yaml(output)
        console.print(f"\n[green]Analysis saved to {output}[/green]")


def _print_analysis_summary(analysis):
    """Print analysis summary."""
    console.print(
        Panel(
            f"[bold]{len(analysis.patterns)}[/bold] failure patterns identified\n"
            f"[bold]{len(analysis.problematic_criteria)}[/bold] problematic criteria\n"
            f"[bold]{len(analysis.recommendations)}[/bold] recommendations",
            title="Analysis Report",
        )
    )

    # Patterns
    if analysis.patterns:
        console.print("\n[bold]Failure Patterns:[/bold]")
        for pattern in analysis.patterns:
            severity_color = {"high": "red", "medium": "yellow", "low": "dim"}.get(
                pattern.severity, "white"
            )
            console.print(
                f"  [{severity_color}][{pattern.severity.upper()}][/] {pattern.description}"
            )

    # Problematic criteria
    if analysis.problematic_criteria:
        table = Table(title="Problematic Criteria")
        table.add_column("Criterion", style="cyan")
        table.add_column("Failure Rate", justify="right")
        table.add_column("FP Rate", justify="right")
        table.add_column("FN Rate", justify="right")

        for pc in analysis.problematic_criteria:
            table.add_row(
                pc.criterion_id,
                f"{pc.failure_rate:.1%}",
                f"{pc.false_positive_rate:.1%}",
                f"{pc.false_negative_rate:.1%}",
            )

        console.print(table)

    # Recommendations
    if analysis.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for i, rec in enumerate(analysis.recommendations, 1):
            console.print(f"  {i}. {rec}")


# ============================================================================
# Hypothesize Command
# ============================================================================

hypothesize_app = typer.Typer(
    name="hypothesize",
    help="Generate improvement hypotheses",
)


@hypothesize_app.callback(invoke_without_command=True)
def hypothesize_cmd(
    ctx: typer.Context,
    analysis: Annotated[
        Path,
        typer.Option("--analysis", "-a", help="Path to analysis report YAML"),
    ],
    workflow: Annotated[
        Path,
        typer.Option("--workflow", "-w", help="Path to workflow YAML"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Path to save hypotheses"),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="Generation mode: template, llm, or hybrid"),
    ] = "hybrid",
):
    """Generate improvement hypotheses from analysis."""
    if ctx.invoked_subcommand is not None:
        return

    from policyflow.benchmark.models import AnalysisReport

    # Load analysis and workflow
    console.print(f"[dim]Loading analysis from {analysis}...[/dim]")
    analysis_report = AnalysisReport.load_yaml(analysis)

    console.print(f"[dim]Loading workflow from {workflow}...[/dim]")
    with workflow.open() as f:
        workflow_data = yaml.safe_load(f)
    parsed_workflow = ParsedWorkflowPolicy.model_validate(workflow_data)

    # Generate hypotheses
    generator = create_hypothesis_generator(mode=mode)

    with console.status("Generating hypotheses..."):
        hypotheses = generator.generate(analysis_report, parsed_workflow)

    # Display results
    console.print(
        Panel(
            f"[bold]{len(hypotheses)}[/bold] hypotheses generated",
            title="Improvement Hypotheses",
        )
    )

    for i, hyp in enumerate(hypotheses, 1):
        console.print(f"\n[cyan]Hypothesis {i}: {hyp.description}[/cyan]")
        console.print(f"  Type: {hyp.change_type}")
        console.print(f"  Target: {hyp.target}")
        console.print(f"  Rationale: {hyp.rationale}")
        console.print(f"  Expected impact: {hyp.expected_impact}")

    # Save if requested
    if output:
        hypotheses_yaml = yaml.dump(
            [h.model_dump(mode="json") for h in hypotheses],
            default_flow_style=False,
        )
        with output.open("w") as f:
            f.write(hypotheses_yaml)
        console.print(f"\n[green]Hypotheses saved to {output}[/green]")


# ============================================================================
# Experiments Commands
# ============================================================================

experiments_app = typer.Typer(
    name="experiments",
    help="Manage benchmark experiments",
)


@experiments_app.command("list")
def experiments_list(
    experiments_dir: Annotated[
        Path,
        typer.Option("--dir", help="Experiments directory"),
    ] = Path("experiments"),
):
    """List all recorded experiments."""
    tracker = FileBasedExperimentTracker(experiments_dir)
    history = tracker.get_history()

    if not history:
        console.print("[dim]No experiments recorded yet.[/dim]")
        return

    table = Table(title=f"Experiments ({len(history)})")
    table.add_column("ID", style="cyan")
    table.add_column("Timestamp")
    table.add_column("Accuracy", justify="right")
    table.add_column("Parent")

    for exp in history:
        table.add_row(
            exp.id,
            exp.timestamp.strftime("%Y-%m-%d %H:%M"),
            f"{exp.accuracy:.1%}",
            exp.parent_experiment_id or "-",
        )

    console.print(table)


@experiments_app.command("best")
def experiments_best(
    experiments_dir: Annotated[
        Path,
        typer.Option("--dir", help="Experiments directory"),
    ] = Path("experiments"),
):
    """Show the best experiment by accuracy."""
    tracker = FileBasedExperimentTracker(experiments_dir)
    best = tracker.get_best()

    if not best:
        console.print("[dim]No experiments recorded yet.[/dim]")
        return

    console.print(
        Panel(
            f"[bold green]{best.accuracy:.1%}[/] accuracy\n"
            f"Timestamp: {best.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
            f"Workflow: {best.workflow_snapshot[:50]}...",
            title=f"Best Experiment: {best.id}",
        )
    )


@experiments_app.command("compare")
def experiments_compare(
    exp1: Annotated[str, typer.Argument(help="First experiment ID")],
    exp2: Annotated[str, typer.Argument(help="Second experiment ID")],
    experiments_dir: Annotated[
        Path,
        typer.Option("--dir", help="Experiments directory"),
    ] = Path("experiments"),
):
    """Compare two experiments."""
    tracker = FileBasedExperimentTracker(experiments_dir)
    comparison = tracker.compare(exp1, exp2)

    if not comparison:
        console.print("[red]One or both experiments not found.[/red]")
        raise typer.Exit(1)

    improvement = comparison["accuracy_diff"]
    improvement_color = "green" if improvement > 0 else "red" if improvement < 0 else "dim"

    console.print(
        Panel(
            f"{exp1}: {comparison[f'{exp1}_accuracy']:.1%}\n"
            f"{exp2}: {comparison[f'{exp2}_accuracy']:.1%}\n\n"
            f"[{improvement_color}]Difference: {improvement:+.1%}[/]",
            title="Experiment Comparison",
        )
    )


# ============================================================================
# Generate Dataset Command
# ============================================================================

generate_dataset_app = typer.Typer(
    name="generate-dataset",
    help="Generate golden dataset from a policy",
)


@generate_dataset_app.callback(invoke_without_command=True)
def generate_dataset_cmd(
    ctx: typer.Context,
    policy: Annotated[
        Path,
        typer.Option("--policy", "-p", help="Path to normalized policy YAML"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path to save generated dataset"),
    ],
    cases_per_criterion: Annotated[
        int,
        typer.Option("--cases-per-criterion", "-n", help="Number of test cases per criterion"),
    ] = 3,
    edge_cases: Annotated[
        bool,
        typer.Option("--edge-cases/--no-edge-cases", help="Include edge cases"),
    ] = True,
    partial_matches: Annotated[
        bool,
        typer.Option("--partial-matches/--no-partial-matches", help="Include partial match cases"),
    ] = True,
    mode: Annotated[
        str,
        typer.Option("--mode", help="Generation mode: template, llm, or hybrid"),
    ] = "hybrid",
):
    """Generate golden dataset from a normalized policy."""
    if ctx.invoked_subcommand is not None:
        return

    from policyflow.benchmark.generator import create_generator
    from policyflow.benchmark.models import GeneratorConfig
    from policyflow.models import NormalizedPolicy

    # Load normalized policy
    console.print(f"[dim]Loading policy from {policy}...[/dim]")
    with policy.open() as f:
        policy_data = yaml.safe_load(f)
    normalized_policy = NormalizedPolicy.model_validate(policy_data)

    # Configure generator
    config = GeneratorConfig(
        cases_per_criterion=cases_per_criterion,
        include_edge_cases=edge_cases,
        include_partial_matches=partial_matches,
        mode=mode,
    )

    # Generate dataset
    generator = create_generator(mode=mode)

    with console.status(f"Generating dataset with {cases_per_criterion} cases per criterion..."):
        dataset = generator.generate(normalized_policy, config)

    # Display summary
    console.print(
        Panel(
            f"[bold]{len(dataset.test_cases)}[/bold] test cases generated\n"
            f"Policy: {normalized_policy.title}",
            title="Dataset Generated",
        )
    )

    # Category breakdown
    categories: dict[str, int] = {}
    for tc in dataset.test_cases:
        categories[tc.category] = categories.get(tc.category, 0) + 1

    if categories:
        table = Table(title="Category Breakdown")
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right")

        for cat, count in sorted(categories.items()):
            table.add_row(cat, str(count))

        console.print(table)

    # Save dataset
    dataset.save_yaml(output)
    console.print(f"\n[green]Dataset saved to {output}[/green]")


# ============================================================================
# Optimize Command
# ============================================================================

optimize_app = typer.Typer(
    name="optimize",
    help="Run optimization loop to improve workflow",
)


@optimize_app.callback(invoke_without_command=True)
def optimize_cmd(
    ctx: typer.Context,
    workflow: Annotated[
        Path,
        typer.Option("--workflow", "-w", help="Path to workflow YAML"),
    ],
    dataset: Annotated[
        Path,
        typer.Option("--dataset", "-d", help="Path to golden dataset YAML"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Path to save optimized workflow"),
    ] = None,
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", help="Maximum optimization iterations"),
    ] = 10,
    target_accuracy: Annotated[
        float | None,
        typer.Option("--target", help="Target accuracy to achieve"),
    ] = None,
    patience: Annotated[
        int,
        typer.Option("--patience", help="Stop after N iterations without improvement"),
    ] = 3,
):
    """Run optimization loop to improve workflow accuracy."""
    if ctx.invoked_subcommand is not None:
        return

    from policyflow.benchmark.models import OptimizationBudget
    from policyflow.benchmark.optimizer import create_optimizer

    # Load workflow and dataset
    console.print(f"[dim]Loading workflow from {workflow}...[/dim]")
    with workflow.open() as f:
        workflow_data = yaml.safe_load(f)
    parsed_workflow = ParsedWorkflowPolicy.model_validate(workflow_data)

    console.print(f"[dim]Loading dataset from {dataset}...[/dim]")
    golden_dataset = load_golden_dataset(dataset)

    # Configure optimizer
    budget = OptimizationBudget(
        max_iterations=max_iterations,
        target_metric=target_accuracy,
        patience=patience,
    )

    optimizer = create_optimizer()

    console.print(
        Panel(
            f"Max iterations: {max_iterations}\n"
            f"Target accuracy: {f'{target_accuracy:.0%}' if target_accuracy else 'None'}\n"
            f"Patience: {patience}",
            title="Starting Optimization",
        )
    )

    # Run optimization (with mocked benchmark to avoid actual LLM calls for now)
    from policyflow.benchmark.runner import SimpleBenchmarkRunner

    with console.status("Running optimization loop..."):
        result = optimizer.optimize(
            workflow=parsed_workflow,
            dataset=golden_dataset,
            budget=budget,
        )

    # Display results
    converged_status = "[green]converged[/green]" if result.converged else "[yellow]stopped[/yellow]"
    console.print(
        Panel(
            f"Best accuracy: [bold]{result.best_metric:.1%}[/bold]\n"
            f"Iterations: {len(result.history)}\n"
            f"Status: {converged_status} ({result.convergence_reason})\n"
            f"Total time: {result.total_time_seconds:.1f}s",
            title="Optimization Complete",
        )
    )

    # Show history
    if result.history:
        table = Table(title="Optimization History")
        table.add_column("Iteration", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Changes")

        for step in result.history:
            changes = ", ".join(step.changes_made[:2]) if step.changes_made else "-"
            if len(step.changes_made) > 2:
                changes += f" (+{len(step.changes_made) - 2} more)"
            table.add_row(
                str(step.iteration),
                f"{step.metric:.1%}",
                changes,
            )

        console.print(table)

    # Save optimized workflow if requested
    if output and result.best_workflow_yaml:
        with output.open("w") as f:
            f.write(result.best_workflow_yaml)
        console.print(f"\n[green]Optimized workflow saved to {output}[/green]")


# ============================================================================
# Improve Command (Convenience)
# ============================================================================

improve_app = typer.Typer(
    name="improve",
    help="Full improvement loop: benchmark → analyze → hypothesize → optimize",
)


@improve_app.callback(invoke_without_command=True)
def improve_cmd(
    ctx: typer.Context,
    workflow: Annotated[
        Path,
        typer.Option("--workflow", "-w", help="Path to workflow YAML"),
    ],
    dataset: Annotated[
        Path,
        typer.Option("--dataset", "-d", help="Path to golden dataset YAML"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Path to save improved workflow"),
    ] = None,
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", help="Maximum optimization iterations"),
    ] = 5,
    target_accuracy: Annotated[
        float | None,
        typer.Option("--target", help="Target accuracy to achieve"),
    ] = None,
):
    """Run full improvement loop: benchmark, analyze, generate hypotheses, and optimize."""
    if ctx.invoked_subcommand is not None:
        return

    from policyflow.benchmark.models import OptimizationBudget
    from policyflow.benchmark.optimizer import create_optimizer

    # Load workflow and dataset
    console.print(f"[dim]Loading workflow from {workflow}...[/dim]")
    with workflow.open() as f:
        workflow_data = yaml.safe_load(f)
    parsed_workflow = ParsedWorkflowPolicy.model_validate(workflow_data)

    console.print(f"[dim]Loading dataset from {dataset}...[/dim]")
    golden_dataset = load_golden_dataset(dataset)

    # Step 1: Initial benchmark
    console.print("\n[bold]Step 1: Initial Benchmark[/bold]")
    config = BenchmarkConfig(workflow_id=f"improve_{datetime.now().strftime('%H%M%S')}")
    runner = SimpleBenchmarkRunner(config)

    with console.status("Running initial benchmark..."):
        initial_report = runner.run(parsed_workflow, golden_dataset.test_cases)

    initial_accuracy = initial_report.metrics.overall_accuracy
    console.print(f"Initial accuracy: [{'green' if initial_accuracy >= 0.8 else 'yellow'}]{initial_accuracy:.1%}[/]")

    # Step 2: Analyze failures
    console.print("\n[bold]Step 2: Analyzing Failures[/bold]")
    analyzer = create_analyzer(mode="hybrid")

    with console.status("Analyzing failures..."):
        analysis = analyzer.analyze(initial_report, parsed_workflow)

    console.print(f"Found {len(analysis.patterns)} failure patterns, {len(analysis.problematic_criteria)} problematic criteria")

    # Step 3: Generate hypotheses
    console.print("\n[bold]Step 3: Generating Hypotheses[/bold]")
    hypothesis_gen = create_hypothesis_generator(mode="hybrid")

    with console.status("Generating hypotheses..."):
        hypotheses = hypothesis_gen.generate(analysis, parsed_workflow)

    console.print(f"Generated {len(hypotheses)} improvement hypotheses")

    # Step 4: Optimize
    console.print("\n[bold]Step 4: Running Optimization[/bold]")
    budget = OptimizationBudget(
        max_iterations=max_iterations,
        target_metric=target_accuracy,
        patience=2,
    )

    optimizer = create_optimizer()

    with console.status("Optimizing workflow..."):
        result = optimizer.optimize(
            workflow=parsed_workflow,
            dataset=golden_dataset,
            budget=budget,
        )

    # Final summary
    improvement = result.best_metric - initial_accuracy
    improvement_color = "green" if improvement > 0 else "red" if improvement < 0 else "dim"

    console.print(
        Panel(
            f"Initial accuracy: {initial_accuracy:.1%}\n"
            f"Final accuracy: [bold]{result.best_metric:.1%}[/bold]\n"
            f"[{improvement_color}]Improvement: {improvement:+.1%}[/]\n\n"
            f"Status: {'[green]converged[/green]' if result.converged else '[yellow]stopped[/yellow]'} ({result.convergence_reason})",
            title="Improvement Complete",
        )
    )

    # Save improved workflow if requested
    if output and result.best_workflow_yaml:
        with output.open("w") as f:
            f.write(result.best_workflow_yaml)
        console.print(f"\n[green]Improved workflow saved to {output}[/green]")


# ============================================================================
# Main App Integration
# ============================================================================


def register_benchmark_commands(app: typer.Typer):
    """Register benchmark commands with the main CLI app.

    Args:
        app: The main typer application
    """
    app.add_typer(benchmark_app, name="benchmark")
    app.add_typer(analyze_app, name="analyze")
    app.add_typer(hypothesize_app, name="hypothesize")
    app.add_typer(experiments_app, name="experiments")
    app.add_typer(generate_dataset_app, name="generate-dataset")
    app.add_typer(optimize_app, name="optimize")
    app.add_typer(improve_app, name="improve")
