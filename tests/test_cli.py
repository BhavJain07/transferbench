"""Concise offline tests of the public Typer surface, not real-provider findings."""

import json
import shutil
import tomllib
from pathlib import Path

import pytest
import yaml
from inspect_ai.log import read_eval_log
from typer.testing import CliRunner

from transferbench import __version__, cli
from transferbench.environments.agentdojo import upstream
from transferbench.models import registry as registry_module
from transferbench.runner import matrix as matrix_module
from transferbench.runner.manifest import read_episodes
from transferbench.tasks.dataset import generate_synthetic_tasks

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
COMMANDS = [
    "smoke",
    "matrix",
    "run",
    "discover",
    "select-defense",
    "analyze",
    "report",
    "verify",
    "agentdojo-export",
]


@pytest.fixture
def runner(tmp_path, monkeypatch):
    # Do not load a developer's .env or write Inspect trace/cache state to their home.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def forbidden(*args, **kwargs):
        raise AssertionError("CLI tests must never resolve external provider clients")

    monkeypatch.setattr(registry_module, "get_model", forbidden)
    return CliRunner()


def config_file(tmp_path, *, paid=False, scaffolds=None):
    registry = tmp_path / "models.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "selected": {
                        "provider": "openai" if paid else "fake",
                        "model": "synthetic-version" if paid else "vulnerable",
                        "family": "offline-cli-fixture",
                        "pricing": {
                            "input_per_million": 1 if paid else 0,
                            "output_per_million": 2 if paid else 0,
                        },
                        "version_pinned": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "matrix.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": ["selected"],
                "model_registry": str(registry),
                "scaffolds": scaffolds or ["chat_single"],
                "tasks": {"limit": 1, "suites": ["workspace"]},
                "attacks": ["none"],
                "defenses": ["none"],
                "include_controls": False,
                "limits": {"max_tokens": 64, "max_turns": 5, "max_cost_usd": 1 if paid else 0},
            }
        ),
        encoding="utf-8",
    )
    return path


def completed_directory(tmp_path, *, status="completed"):
    directory = tmp_path / "stub-run"
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "status": status,
                "n_completed": 1,
                "n_episodes": 1,
                "estimated_cost_usd": 0.02,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_version_without_subcommand(runner):
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == __version__


@pytest.mark.parametrize("command", [None, *COMMANDS], ids=lambda command: command or "root")
def test_help_for_every_command(runner, command):
    result = runner.invoke(cli.app, ([command] if command else []) + ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output and "--help" in result.output
    if command is None:
        assert all(name in result.output for name in COMMANDS)


def test_matrix_dry_run_reports_count_without_resolving_or_executing(runner, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not resolve a model or execute a matrix")

    monkeypatch.setattr(registry_module.ModelSpec, "resolve", forbidden)
    monkeypatch.setattr(matrix_module, "run_matrix", forbidden)
    result = runner.invoke(cli.app, ["matrix", str(CONFIGS / "smoke.yaml"), "--dry-run"])
    assert result.exit_code == 0, result.output
    projection = json.loads(result.output)
    assert projection["estimated_episodes"] == 64
    assert projection["conditions"] == {"C0": 8, "C1": 8, "C2": 24, "C3": 24}
    assert projection["simulated"] and projection["conservative_maximum_usd"] == 0


def test_run_no_attack_selects_exactly_one_condition(runner, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning one condition must not execute models")

    monkeypatch.setattr(matrix_module, "run_matrix", forbidden)
    for defense, condition in (("none", "C0"), ("tool_policy", "C3")):
        result = runner.invoke(
            cli.app, ["run", "--attack", "no_attack", "--defense", defense, "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        projection = json.loads(result.output)
        assert projection["estimated_episodes"] == 1
        assert projection["conditions"] == {
            name: int(name == condition) for name in ("C0", "C1", "C2", "C3")
        }


def test_run_monitor_validation_and_explicit_alias(runner):
    args = ["run", "--attack", "no_attack", "--defense", "sanitizer + monitor", "--dry-run"]
    missing = runner.invoke(cli.app, args)
    assert missing.exit_code == 1
    assert "Error:" in missing.output and "requires monitor_model" in missing.output
    valid = runner.invoke(cli.app, [*args, "--monitor-model", "fake_b"])
    assert valid.exit_code == 0, valid.output
    assert json.loads(valid.output)["conditions"] == {"C0": 0, "C1": 0, "C2": 0, "C3": 1}


@pytest.mark.parametrize(
    "command,extra",
    [
        ("matrix", []),
        ("discover", ["--source-model", "fake_a", "--source-scaffold", "chat_single"]),
    ],
)
def test_nonexistent_config_is_a_clear_usage_error(runner, tmp_path, command, extra, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, [command, "missing.yaml", *extra])
    assert result.exit_code == 2
    assert "does not exist" in result.output and "missing.yaml" in result.output


def test_paid_matrix_plans_prompts_and_yes_skips_confirmation_without_real_calls(
    runner, tmp_path, monkeypatch
):
    config = config_file(tmp_path, paid=True)
    directory = completed_directory(tmp_path)
    calls = []

    def execute(path, **kwargs):
        calls.append((path, kwargs))
        return directory

    monkeypatch.setattr(matrix_module, "run_matrix", execute)
    dry = runner.invoke(cli.app, ["matrix", str(config), "--dry-run"])
    assert dry.exit_code == 0 and not json.loads(dry.output)["simulated"]
    assert not calls
    declined = runner.invoke(cli.app, ["matrix", str(config)], input="n\n")
    assert declined.exit_code == 1 and "Proceed?" in declined.output
    assert "may incur charges" in declined.output and not calls

    def no_prompt(*args, **kwargs):
        raise AssertionError("--yes must not prompt")

    monkeypatch.setattr(cli.typer, "confirm", no_prompt)
    result = runner.invoke(
        cli.app, ["matrix", str(config), "--yes", "--output-root", "output", "--run-id", "accepted"]
    )
    assert result.exit_code == 0, result.output
    assert "Results:" in result.output and "Status: completed" in result.output
    assert "Proceed?" not in result.output
    assert calls == [
        (config, {"output_root": (tmp_path / "output").resolve(), "run_id": "accepted"})
    ]


def test_smoke_dispatch_runs_small_genuine_inspect_matrix(runner, tmp_path, monkeypatch):
    config = config_file(tmp_path, scaffolds=["chat_single", "tool_agent"])
    selected = []

    def bundled(name):
        selected.append(name)
        assert name == "smoke.yaml"
        return config

    finished = []

    def finish(path, report=False):
        finished.append((path, report))

    monkeypatch.setattr(cli, "bundled_config", bundled)
    monkeypatch.setattr(cli, "_finish", finish)  # Avoid generating an additional HTML report.
    result = runner.invoke(cli.app, ["smoke", "--output-root", "smoke-results"])
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert selected == ["smoke.yaml"] and len(finished) == 1
    directory, report = finished[0]
    assert report is True and directory.parent == (tmp_path / "smoke-results").resolve()
    episodes = read_episodes(directory / "episodes.parquet")
    assert len(episodes) == 2 and all(
        episode.status == "ok" and episode.simulated for episode in episodes
    )
    logs = [read_eval_log(path) for path in (directory / "inspect").glob("*.eval")]
    assert logs and all(log.status == "success" for log in logs)
    samples = []
    for log in logs:
        assert log.samples is not None
        samples.extend(log.samples)
    assert any(event.event == "tool" for sample in samples for event in sample.events)


@pytest.mark.parametrize("command", ["analyze", "report"])
def test_invalid_analysis_files_have_clear_cli_errors(runner, tmp_path, command):
    directory = tmp_path / "invalid-run"
    directory.mkdir()
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    missing = runner.invoke(cli.app, [command, str(directory)])
    assert missing.exit_code == 1 and "Error:" in missing.output
    (directory / "episodes.parquet").write_bytes(b"not a parquet file")
    corrupt = runner.invoke(cli.app, [command, str(directory)])
    assert corrupt.exit_code == 1
    assert "Error:" in corrupt.output, f"Unformatted exception: {corrupt.exception!r}"
    assert "parquet" in corrupt.output.casefold()
    assert not (directory / "summary.json").exists() and not (directory / "report.html").exists()


def test_agentdojo_export_forwards_arguments_and_handles_optional_dependency(
    runner, tmp_path, monkeypatch
):
    expected = {"user_task_17": ["Exact synthetic fact"]}
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(expected), encoding="utf-8")
    output = tmp_path / "export.jsonl"
    calls = []
    task = generate_synthetic_tasks()[0]

    def export(suite, task_ids, expected_facts, *, seed):
        calls.append((suite, task_ids, expected_facts, seed))
        return [task]

    monkeypatch.setattr(upstream, "export_seed_tasks", export)
    args = [
        "agentdojo-export",
        "--suite",
        "workspace",
        "--facts",
        str(facts),
        "--output",
        str(output),
        "--seed",
        "17",
    ]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert calls == [("workspace", ["user_task_17"], expected, 17)]
    assert json.loads(output.read_text()) == task.model_dump(mode="json")

    def unavailable(*args, **kwargs):
        raise upstream.AgentDojoUnavailable("Install the optional agentdojo dependency")

    monkeypatch.setattr(upstream, "export_seed_tasks", unavailable)
    missing = runner.invoke(cli.app, args)
    assert missing.exit_code == 1
    assert "Error:" in missing.output and "optional agentdojo" in missing.output


def test_bundled_config_falls_back_to_packaged_configs_with_local_precedence(
    runner, tmp_path, monkeypatch
):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert (
        project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]["configs"]
        == "transferbench/_configs"
    )
    package = tmp_path / "installed" / "transferbench"
    packaged = package / "_configs"
    shutil.copytree(CONFIGS, packaged)
    monkeypatch.setattr(cli, "__file__", str(package / "cli.py"))
    assert cli.bundled_config("smoke.yaml") == packaged / "smoke.yaml"
    assert cli.bundled_config("models.yaml") == packaged / "models.yaml"
    result = runner.invoke(cli.app, ["matrix", str(cli.bundled_config("smoke.yaml")), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["estimated_episodes"] == 64
    local = tmp_path / "configs"
    local.mkdir()
    (local / "smoke.yaml").write_text("local config", encoding="utf-8")
    assert cli.bundled_config("smoke.yaml") == local / "smoke.yaml"
    with pytest.raises(FileNotFoundError, match="Cannot find bundled config"):
        cli.bundled_config("not-bundled.yaml")


def test_verify_invalid_integrity_returns_json_and_failure_status(runner, tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"publishable": False}), encoding="utf-8")
    result = runner.invoke(cli.app, ["verify", str(tmp_path)])
    assert result.exit_code == 1
    verification = json.loads(result.output)
    assert not verification["valid"] and not verification["publishable"]
    assert "No file integrity records" in verification["errors"]


def test_matrix_incomplete_finish_uses_exit_code_two(runner, tmp_path, monkeypatch):
    config = config_file(tmp_path)
    directory = completed_directory(tmp_path, status="budget_exceeded")
    monkeypatch.setattr(matrix_module, "run_matrix", lambda *args, **kwargs: directory)
    result = runner.invoke(cli.app, ["matrix", str(config), "--yes"])
    assert result.exit_code == 2
    assert "Status: budget_exceeded" in result.output and "Results:" in result.output
