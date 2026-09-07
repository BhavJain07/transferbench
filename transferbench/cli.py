"""The public CLI: planning is free; all real execution requires an explicit budget."""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from polars.exceptions import PolarsError

from transferbench import __version__

app = typer.Typer(
    no_args_is_help=True,
    invoke_without_command=True,
    help="TransferBench — a generalization benchmark for AI safety claims.",
)


@contextmanager
def errors():
    try:
        yield
    except (ValueError, OSError, ImportError, PolarsError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


def bundled_config(name: str) -> Path:
    for directory in (
        Path.cwd() / "configs",
        Path(__file__).resolve().parents[1] / "configs",
        Path(__file__).parent / "_configs",
    ):
        if (directory / name).is_file():
            return directory / name
    raise FileNotFoundError(f"Cannot find bundled config {name}")


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Print the package version.")] = False,
):
    load_dotenv(Path.cwd() / ".env", override=False)
    if version:
        typer.echo(__version__)
        raise typer.Exit()


def _preview(plan, yes: bool, dry_run: bool) -> bool:
    typer.echo(json.dumps(plan.projection, indent=2))
    if dry_run:
        return False
    if not plan.projection["simulated"]:
        typer.echo(
            "Real model execution sends synthetic task content to configured providers and may incur charges."
        )
        typer.echo(
            "Ceiling applies to conservative configured estimates; also set a provider-side invoice cap."
        )
        if not yes:
            typer.confirm("Proceed?", default=False, abort=True)
    return True


def _finish(run_dir: Path, report: bool = False):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    typer.echo(f"Results: {run_dir}")
    typer.echo(
        f"Status: {manifest['status']}; episodes: {manifest['n_completed']}/{manifest['n_episodes']}; estimated spend: ${manifest['estimated_cost_usd']:.6f}"
    )
    if report and (run_dir / "episodes.parquet").exists():
        from transferbench.analysis import generate_report

        typer.echo(f"Report: {generate_report(run_dir)}")
    if manifest["status"] != "completed":
        raise typer.Exit(2)


@app.command()
def smoke(output_root: Annotated[Path, typer.Option()] = Path("results")):
    """Exercise all four scaffolds without credentials, network, or paid calls."""
    from transferbench.runner.matrix import run_matrix

    with errors():
        path = run_matrix(bundled_config("smoke.yaml"), output_root=output_root.resolve())
        _finish(path, report=True)


@app.command()
def matrix(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    dry_run: Annotated[bool, typer.Option()] = False,
    output_root: Annotated[Path | None, typer.Option()] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
    report: Annotated[bool, typer.Option("--report")] = False,
):
    """Plan/run a controlled matrix. --dry-run never resolves provider clients."""
    from transferbench.runner.matrix import plan_matrix, run_matrix

    with errors():
        plan = plan_matrix(config)
        if _preview(plan, yes, dry_run):
            path = run_matrix(
                config, output_root=output_root.resolve() if output_root else None, run_id=run_id
            )
            _finish(path, report)


@app.command()
def run(
    model: Annotated[str, typer.Option()] = "fake_a",
    scaffold: Annotated[str, typer.Option()] = "tool_agent",
    attack: Annotated[str, typer.Option()] = "indirect",
    defense: Annotated[str, typer.Option()] = "none",
    task_limit: Annotated[int, typer.Option(min=1)] = 1,
    repeats: Annotated[int, typer.Option(min=1)] = 1,
    registry: Annotated[Path | None, typer.Option()] = None,
    monitor_model: Annotated[str | None, typer.Option()] = None,
    max_cost_usd: Annotated[float, typer.Option(min=0)] = 1,
    max_turns: Annotated[int, typer.Option(min=1)] = 20,
    max_tokens: Annotated[int, typer.Option(min=32)] = 2048,
    seed: Annotated[int, typer.Option(min=0)] = 42,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    dry_run: Annotated[bool, typer.Option()] = False,
    output_root: Annotated[Path, typer.Option()] = Path("results"),
):
    """Run exactly one condition (use matrix to add matched C0–C3 controls)."""
    from transferbench.runner.config import MatrixConfig
    from transferbench.runner.matrix import plan_matrix, run_matrix

    with errors():
        path = bundled_config("smoke.yaml")
        config = MatrixConfig.model_validate(
            {
                "experiment": {"name": "single_condition"},
                "model_registry": str((registry or bundled_config("models.yaml")).resolve()),
                "models": [model],
                "scaffolds": [scaffold],
                "attacks": [attack],
                "defenses": [defense],
                "tasks": {"suites": ["workspace"], "limit": task_limit},
                "repeats": repeats,
                "seed": seed,
                "monitor_model": monitor_model,
                "include_controls": False,
                "limits": {
                    "max_cost_usd": max_cost_usd,
                    "max_turns": max_turns,
                    "max_tokens": max_tokens,
                },
            }
        )
        plan = plan_matrix(path, config)
        if _preview(plan, yes, dry_run):
            _finish(run_matrix(path, output_root=output_root.resolve(), config_override=config))


@app.command()
def discover(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    source_model: Annotated[str, typer.Option()],
    source_scaffold: Annotated[str, typer.Option()],
    family: Annotated[str, typer.Option()] = "indirect_instruction",
    candidates: Annotated[int, typer.Option(min=1, max=3)] = 3,
    top_k: Annotated[int, typer.Option(min=1, max=3)] = 1,
    output_root: Annotated[Path | None, typer.Option()] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
):
    """Source-select finite templates, freeze top-k, and write held-out transfer.yaml."""
    from transferbench.runner.config import load_config
    from transferbench.runner.discovery import discover as run_discovery
    from transferbench.runner.matrix import plan_matrix

    with errors():
        cfg = load_config(config)
        probe = cfg.model_copy(
            update={
                "models": [source_model],
                "scaffolds": [source_scaffold],
                "defenses": ["none"],
                "monitor_model": None,
            }
        )
        plan = plan_matrix(config, probe)
        n = len(plan.tasks) * candidates * cfg.repeats
        spec = plan.models[source_model]
        bound = (
            n
            * cfg.limits.max_turns
            * spec.pricing.estimate(spec.max_input_tokens, cfg.limits.max_tokens)
        )
        typer.echo(
            f"Selection episodes: {n}; conservative bound: ${bound:.6f}; hard estimate ceiling: ${cfg.limits.max_cost_usd:.6f}"
        )
        typer.echo(
            "Finite template pool: 3 unique payloads per task/family. Selection is not held-out evidence."
        )
        if not spec.simulated and not yes:
            typer.confirm(
                "Send synthetic data to the source provider and incur charges?",
                default=False,
                abort=True,
            )
        path = run_discovery(
            config,
            source_model=source_model,
            source_scaffold=source_scaffold,
            family=family,
            candidates=candidates,
            top_k=top_k,
            output_root=output_root,
        )
        typer.echo(f"Frozen selection: {path}")
        typer.echo(f"Next: transferbench matrix {path / 'transfer.yaml'} --report")


@app.command("select-defense")
def select_defense(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    source_model: Annotated[str, typer.Option()],
    source_scaffold: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()],
    utility_weight: Annotated[float, typer.Option(min=0)] = 0.5,
):
    """Freeze the best static defense from complete source-only C0–C3 evidence."""
    from transferbench.runner.defense_selection import select_defense as select

    with errors():
        path = select(
            run_dir,
            source_model=source_model,
            source_scaffold=source_scaffold,
            output=output,
            utility_weight=utility_weight,
        )
        typer.echo(
            f"Frozen defense: {path}. Set defense_selection in a new, held-out matrix config."
        )


@app.command()
def analyze(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    publishable: Annotated[bool, typer.Option()] = False,
):
    """Write stratified transfer estimates and clustered uncertainty to summary.json."""
    from transferbench.analysis import analyze_run

    with errors():
        summary = analyze_run(run_dir, publishable=publishable)
        typer.echo(f"Summary: {run_dir / 'summary.json'}")
        typer.echo(json.dumps(summary.get("counts", {}), indent=2))


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path | None, typer.Option()] = None,
    publishable: Annotated[bool, typer.Option()] = False,
):
    """Generate a self-contained HTML report, with no external JavaScript services."""
    from transferbench.analysis import generate_report

    with errors():
        typer.echo(str(generate_report(run_dir, output=output, publishable=publishable)))


@app.command()
def verify(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)]):
    """Check recorded evidence hashes; integrity is not independent authenticity."""
    from transferbench.runner.manifest import verify_run

    with errors():
        result = verify_run(run_dir)
        typer.echo(json.dumps(result, indent=2))
        if not result["valid"]:
            raise typer.Exit(1)


@app.command("agentdojo-export")
def agentdojo_export(
    suite: Annotated[str, typer.Option()],
    facts: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    seed: Annotated[int, typer.Option(min=0)] = 42,
):
    """Export supported local AgentDojo seed retrieval proxies; never native scores."""
    from transferbench.environments.agentdojo.upstream import (
        export_seed_tasks,
        write_seed_tasks_jsonl,
    )

    with errors():
        expected = json.loads(facts.read_text())
        if not isinstance(expected, dict):
            raise ValueError(
                "facts must be a JSON mapping from native task IDs to exact fact lists"
            )
        tasks = export_seed_tasks(suite, list(expected), expected, seed=seed)
        typer.echo(str(write_seed_tasks_jsonl(output, tasks)))


if __name__ == "__main__":
    app()
