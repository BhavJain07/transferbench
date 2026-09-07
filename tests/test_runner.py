"""Offline planner/runner integration tests; fixtures are not real-model evidence.

One module-scoped Inspect run is reused for all canonical-log/storage checks.
The large presets are planned only: no real provider is ever resolved.
"""

import asyncio
import copy
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelName
from inspect_ai.solver import TaskState

from transferbench.analysis import analyze_run
from transferbench.attacks import get_attack
from transferbench.attacks.artifacts import attack_sha256, load_attack, save_attack, seal_attack
from transferbench.models import registry as registry_module
from transferbench.models.fake import FakeModel
from transferbench.runner import matrix as matrix_module
from transferbench.runner.manifest import (
    atomic_json,
    publication_reasons,
    read_episodes,
    sha256_file,
    verify_run,
    write_episodes,
)
from transferbench.runner.matrix import episode_seed, plan_matrix, run_matrix
from transferbench.tasks.dataset import generate_synthetic_tasks
from transferbench.tasks.schema import CallCost, Episode, EpisodeResult, ToolCallRecord

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCAFFOLDS = ["chat_single", "chat_multi", "tool_agent", "delegate_agent"]


def fake_entry(variant="vulnerable", *, input_price=0, output_price=0):
    return {
        "provider": "fake",
        "model": variant,
        "family": f"synthetic_{variant}",
        "pricing": {"input_per_million": input_price, "output_per_million": output_price},
        "supports_seed": True,
        "version_pinned": True,
    }


def make_config(tmp_path, *, entries=None, **updates):
    registry = tmp_path / "models.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": entries
                or {
                    "fake_a": fake_entry(),
                    "fake_b": fake_entry("resistant"),
                }
            }
        ),
        encoding="utf-8",
    )
    values = {
        "experiment": {"name": "runner_test"},
        "models": ["fake_a", "fake_b"],
        "scaffolds": SCAFFOLDS,
        "tasks": {"suites": ["workspace"], "limit": 1},
        "attacks": ["indirect_instruction"],
        "defenses": ["none", "tool_policy"],
        "repeats": 1,
        "seed": 42,
        "limits": {"max_turns": 20, "max_tokens": 512, "max_cost_usd": 0},
        "output_dir": "results",
    } | updates
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def forbid_external_resolution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Runner tests must not resolve external provider models")

    monkeypatch.setattr(registry_module, "get_model", forbidden)


@pytest.fixture
def provider_environment(monkeypatch):
    for prefix in ("ANTHROPIC", "OPENAI", "GOOGLE", "OPEN"):
        monkeypatch.setenv(f"{prefix}_MODEL", "synthetic-pinned-version")
        monkeypatch.setenv(f"{prefix}_INPUT_PER_MILLION", "1")
        monkeypatch.setenv(f"{prefix}_OUTPUT_PER_MILLION", "2")
    monkeypatch.setenv("OPEN_MODEL_BASE_URL", "https://example.invalid/v1")


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("runner-smoke")
    config_path = make_config(base)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HOME", str(base))
        patch.setenv("XDG_DATA_HOME", str(base / "data"))
        patch.setenv("XDG_CACHE_HOME", str(base / "cache"))

        # This guard also covers module-scoped setup, before per-test fixtures run.
        def forbidden(*args, **kwargs):
            raise AssertionError("The shared Inspect run must remain offline")

        patch.setattr(registry_module, "get_model", forbidden)
        directory = run_matrix(
            config_path, output_root=base / "output", run_id="shared-offline-run"
        )
    manifest = json.loads((directory / "manifest.json").read_text())
    episodes = read_episodes(directory / "episodes.parquet")
    logs = [read_eval_log(path) for path in sorted((directory / "inspect").glob("*.eval"))]
    return SimpleNamespace(
        directory=directory,
        config_path=config_path,
        plan=plan_matrix(config_path),
        manifest=manifest,
        episodes=episodes,
        logs=logs,
        summary=json.loads((directory / "summary.json").read_text()),
    )


# Planning is deterministic and does not resolve providers. ------------------


@pytest.mark.parametrize(
    "filename,expected,conditions",
    [
        ("smoke.yaml", 64, {"C0": 8, "C1": 8, "C2": 24, "C3": 24}),
        ("quickstart.yaml", 1080, {"C0": 120, "C1": 240, "C2": 480, "C3": 240}),
        ("reproduce.yaml", 64, {"C0": 16, "C1": 16, "C2": 16, "C3": 16}),
        ("full.yaml", 5760, {"C0": 1152, "C1": 4608, "C2": 0, "C3": 0}),
        ("experiments/pilot.yaml", 480, {"C0": 120, "C1": 120, "C2": 120, "C3": 120}),
    ],
)
def test_shipped_presets_have_documented_counts_without_model_calls(
    filename, expected, conditions, provider_environment, monkeypatch
):
    def forbidden(self):
        raise AssertionError("Planning must not resolve even a simulated model")

    monkeypatch.setattr(registry_module.ModelSpec, "resolve", forbidden)
    plan = plan_matrix(CONFIGS / filename)
    assert len(plan.episodes) == expected
    assert plan.projection["conditions"] == conditions
    assert len({episode.episode_id for episode in plan.episodes}) == expected
    assert plan.projection["estimated_episodes"] == expected


def test_clean_controls_are_deduplicated_across_attack_families(tmp_path):
    plan = plan_matrix(
        make_config(tmp_path, attacks=["none", "direct", "indirect_instruction"], repeats=2)
    )
    assert plan.projection["conditions"] == {"C0": 16, "C1": 32, "C2": 32, "C3": 16}
    clean = [episode for episode in plan.episodes if episode.attack is None]
    keys = [
        (
            episode.model_alias,
            episode.scaffold_id,
            episode.task.task_id,
            episode.defense_id,
            episode.repeat,
        )
        for episode in clean
    ]
    assert len(keys) == len(set(keys)) == 32


@pytest.mark.parametrize(
    "include_controls,attacks,defenses,conditions",
    [
        (False, ["direct"], ["tool_policy"], {"C2": 1}),
        (False, ["none", "direct"], ["tool_policy"], {"C2": 1, "C3": 1}),
        (True, ["direct"], ["tool_policy"], {"C0": 1, "C1": 1, "C2": 1, "C3": 1}),
        (True, [], ["none"], {"C0": 1}),
    ],
)
def test_explicit_control_policy(tmp_path, include_controls, attacks, defenses, conditions):
    path = make_config(
        tmp_path,
        models=["fake_a"],
        scaffolds=["chat_single"],
        include_controls=include_controls,
        attacks=attacks,
        defenses=defenses,
    )
    assert dict(Counter(episode.condition for episode in plan_matrix(path).episodes)) == conditions


@pytest.mark.parametrize("no_attack_alias", ["none", "no_attack"])
def test_no_attack_alias_selects_clean_trials_without_automatic_controls(tmp_path, no_attack_alias):
    path = make_config(
        tmp_path,
        models=["fake_a"],
        scaffolds=["chat_single"],
        include_controls=False,
        attacks=[no_attack_alias],
        defenses=["none"],
    )
    plan = plan_matrix(path)
    assert [episode.condition for episode in plan.episodes] == ["C0"]


def test_fixed_payload_and_artifact_identity_are_reused_across_all_targets(tmp_path):
    path = make_config(
        tmp_path, tasks={"limit": 2}, attacks=["direct", "indirect_instruction"], repeats=2
    )
    plan = plan_matrix(path)
    groups = defaultdict(list)
    for episode in plan.episodes:
        if episode.attack:
            groups[(episode.task.task_id, episode.attack.family)].append(episode.attack)
    assert len(groups) == len(plan.artifacts) == 4
    for attacks in groups.values():
        assert len({attack.payload for attack in attacks}) == 1
        assert len({attack.artifact_sha256 for attack in attacks}) == 1
        assert len({attack.id for attack in attacks}) == 1
        assert {attack.generation_seed for attack in attacks} == {42}
        assert all(attack.artifact_sha256 == attack_sha256(attack) for attack in attacks)
    repeated = plan_matrix(path)
    assert repeated.episodes == plan.episodes and repeated.artifacts == plan.artifacts


def test_family_mode_varies_generation_seed_by_model_and_scaffold(tmp_path):
    path = make_config(tmp_path, experiment={"transfer_mode": "family"}, repeats=2)
    plan = plan_matrix(path)
    groups = defaultdict(list)
    for episode in plan.episodes:
        if episode.attack:
            groups[(episode.model_alias, episode.scaffold_id)].append(episode.attack)
    assert len(groups) == 8
    seeds = set()
    payloads = set()
    for (alias, scaffold), attacks in groups.items():
        expected = episode_seed(plan.config.seed, f"{alias}/{scaffold}", 0)
        assert {attack.generation_seed for attack in attacks} == {expected}
        assert len({attack.artifact_sha256 for attack in attacks}) == 1
        assert {attack.family for attack in attacks} == {"indirect_instruction"}
        seeds.add(expected)
        payloads.add(attacks[0].payload)
    assert len(seeds) == len(plan.artifacts) == 8
    # Templates are finite: distinct target seeds need not yield eight distinct texts.
    assert len(payloads) > 1


def test_episode_seeds_match_tasks_and_repeats_across_all_conditions(tmp_path):
    plan = plan_matrix(
        make_config(
            tmp_path, tasks={"limit": 2}, repeats=3, attacks=["direct", "indirect_instruction"]
        )
    )
    grouped = defaultdict(set)
    for episode in plan.episodes:
        key = (episode.task.task_id, episode.repeat)
        grouped[key].add(episode.seed)
        assert episode.seed == episode_seed(42, *key)
        assert 0 <= episode.seed < 2**31
    assert len(grouped) == 6 and all(len(seeds) == 1 for seeds in grouped.values())
    assert len(set.union(*grouped.values())) == 6
    assert episode_seed(43, plan.tasks[0].task_id, 0) != episode_seed(42, plan.tasks[0].task_id, 0)


def test_episode_seed_has_stable_unambiguous_serialization():
    for task_id in ("task-a", "task/a", "任务", 'task"quoted'):
        payload = json.dumps([42, task_id, 1], separators=(",", ":"))
        expected = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big") % 2**31
        assert episode_seed(42, task_id, 1) == expected


def test_episode_identity_includes_target_defense_attack_seed_and_repeat(tmp_path):
    plan = plan_matrix(make_config(tmp_path))
    clean = next(episode for episode in plan.episodes if episode.condition == "C0")
    attack = plan.artifacts[0]
    variants = [
        clean,
        replace(clean, model_alias="fake_b"),
        replace(clean, scaffold_id="tool_agent"),
        replace(clean, defense_id="tool_policy"),
        replace(clean, attack=attack),
        replace(clean, seed=clean.seed + 1),
        replace(clean, repeat=1),
    ]
    assert len({episode.episode_id for episode in variants}) == len(variants)


def test_projection_accounts_for_monitor_bounds_inside_shared_turn_cap(tmp_path):
    entries = {
        "fake_a": fake_entry(input_price=1, output_price=2),
        "fake_b": fake_entry("resistant", input_price=3, output_price=4),
        "monitor": fake_entry("resistant", input_price=10, output_price=20),
    }
    path = make_config(
        tmp_path,
        entries=entries,
        scaffolds=["chat_single"],
        monitor_model="monitor",
        defenses=["none", "monitor"],
        limits={"max_turns": 5, "max_tokens": 64, "max_cost_usd": 0.01},
    )
    plan = plan_matrix(path)
    assert len(plan.episodes) == 8 and set(plan.models) == set(entries)
    assert {episode.model_alias for episode in plan.episodes} == {"fake_a", "fake_b"}
    maximum_call = plan.models["monitor"].pricing.estimate(32768, 64)
    assert plan.projection["conservative_maximum_usd"] == pytest.approx(8 * 5 * maximum_call)
    assert plan.projection["hard_estimated_cost_ceiling_usd"] == 0.01
    assert plan.projection["may_stop_early"]
    assert plan.projection["simulated"]


@pytest.mark.parametrize("limit,valid", [(31, False), (32, True)])
def test_max_episodes_is_enforced_at_plan_boundary(tmp_path, limit, valid):
    path = make_config(tmp_path, limits={"max_episodes": limit})
    if valid:
        assert len(plan_matrix(path).episodes) == 32
    else:
        with pytest.raises(ValueError, match="exceeds max_episodes"):
            plan_matrix(path)


def test_empty_experiment_is_rejected(tmp_path):
    path = make_config(tmp_path, include_controls=False, attacks=[])
    with pytest.raises(ValueError, match="no episodes"):
        plan_matrix(path)


def test_frozen_artifacts_are_filtered_by_task_and_duplicates_rejected(tmp_path):
    tasks = generate_synthetic_tasks()
    attack = get_attack("direct", tasks[0], seed=7)
    first, duplicate = tmp_path / "first.json", tmp_path / "duplicate.json"
    save_attack(attack, first)
    save_attack(attack, duplicate)
    path = make_config(tmp_path, tasks={"limit": 2}, attack_artifacts=[str(first)], attacks=[])
    plan = plan_matrix(path)
    assert {episode.task.task_id for episode in plan.episodes if episode.attack} == {
        tasks[0].task_id
    }
    assert {episode.attack.artifact_sha256 for episode in plan.episodes if episode.attack} == {
        attack.artifact_sha256
    }
    assert len(plan.artifacts) == 1
    override = plan.config.model_copy(update={"attack_artifacts": [str(first), str(duplicate)]})
    with pytest.raises(ValueError, match="Duplicate frozen attack artifacts"):
        plan_matrix(path, override)


def test_source_optimized_plan_checks_frozen_selection_and_holdout_seed(tmp_path):
    task = generate_synthetic_tasks()[0]
    attack = get_attack("direct", task, seed=7, source_model="fake_a", source_scaffold="tool_agent")
    attack = seal_attack(
        attack.model_copy(update={"selection_id": "frozen-selection", "artifact_sha256": ""})
    )
    artifact_path = tmp_path / "selected.json"
    save_attack(attack, artifact_path)
    selection = {
        "selection_id": attack.selection_id,
        "attack_artifact_sha256": attack.artifact_sha256,
        "source_model": attack.source_model,
        "source_scaffold": attack.source_scaffold,
        "selection_seed": 7,
        "task_id": task.task_id,
        "trial_seeds": [episode_seed(7, task.task_id, 0)],
        "n": 1,
        "split": "selection",
        "selected": True,
    }
    manifest_path = tmp_path / "selection.json"
    atomic_json(
        manifest_path,
        {
            "source_selections": [selection],
            "holdout": {
                "unit": "trials",
                "same_tasks": True,
                "new_task_holdout": False,
                "selection_base_seed": 7,
                "evaluation_base_seed": 42,
                "task_ids": [task.task_id],
                "selection_episode_seeds": [episode_seed(7, task.task_id, 0)],
                "evaluation_episode_seeds": [episode_seed(42, task.task_id, 0)],
            },
        },
    )
    path = make_config(
        tmp_path,
        experiment={"transfer_mode": "source_optimized"},
        attack_artifacts=[str(artifact_path)],
        selection_manifest=str(manifest_path),
        seed=42,
    )
    plan = plan_matrix(path)
    assert plan.selections == [selection]
    assert all(episode.attack == attack for episode in plan.episodes if episode.attack)
    with pytest.raises(ValueError, match="must differ"):
        plan_matrix(path, plan.config.model_copy(update={"seed": 7}))
    atomic_json(
        manifest_path, {"source_selections": [selection | {"attack_artifact_sha256": "0" * 64}]}
    )
    with pytest.raises(ValueError, match="Invalid selected artifact"):
        plan_matrix(path)


# Shared real Inspect run: canonical records and derived storage. -------------


def test_real_smoke_completes_all_scaffolds_and_conditions(smoke_run):
    assert smoke_run.manifest["status"] == "completed"
    assert smoke_run.manifest["n_episodes"] == smoke_run.manifest["n_completed"] == 32
    assert len(smoke_run.episodes) == 32
    assert {episode.scaffold_id for episode in smoke_run.episodes} == set(SCAFFOLDS)
    assert {episode.model_alias for episode in smoke_run.episodes} == {"fake_a", "fake_b"}
    assert Counter(episode.condition for episode in smoke_run.episodes) == dict.fromkeys(
        ("C0", "C1", "C2", "C3"), 8
    )
    assert all(
        episode.status == "ok" and episode.error is None and episode.simulated
        for episode in smoke_run.episodes
    )
    assert all(
        episode.utility_success
        for episode in smoke_run.episodes
        if episode.condition in {"C0", "C3"}
    )
    assert smoke_run.manifest["estimated_cost_usd"] == 0


def test_native_inspect_logs_are_lossless_source_for_parquet(smoke_run):
    assert smoke_run.logs and all(log.status == "success" for log in smoke_run.logs)
    projected = {episode.episode_id: episode for episode in smoke_run.episodes}
    observed = set()
    for log in smoke_run.logs:
        relative = str(Path(log.location).relative_to(smoke_run.directory))
        assert log.eval.metadata["run_id"] == smoke_run.manifest["run_id"]
        for sample in log.samples:
            assert sample.error is None
            assert sample.metadata["plan"]["seed"] == sample.metadata["episode"]["seed"]
            assert "episode_result" in sample.metadata
            canonical = Episode.model_validate(sample.metadata["episode"])
            assert (
                canonical.model_copy(update={"inspect_log": relative})
                == projected[canonical.episode_id]
            )
            assert sample.scores["deterministic_scores"].value == {
                key: int(getattr(canonical, key))
                for key in ("utility_success", "attack_success", "policy_violation")
            }
            assert any(event.event == "model" for event in sample.events)
            observed.add(canonical.episode_id)
    assert observed == set(projected)


def test_canonical_tool_and_worker_events_are_present(smoke_run):
    samples = [sample for log in smoke_run.logs for sample in log.samples]
    tool_samples = [
        sample for sample in samples if sample.metadata["episode"]["scaffold_id"] == "tool_agent"
    ]
    assert all(
        any(event.event == "tool" and event.function == "read_doc" for event in sample.events)
        for sample in tool_samples
    )
    delegates = [
        sample
        for sample in samples
        if sample.metadata["episode"]["scaffold_id"] == "delegate_agent"
    ]
    for sample in delegates:
        names = {event.function for event in sample.events if event.event == "tool"}
        assert names >= {"worker_A", "worker_B"}
        assert any(event.event == "span_begin" for event in sample.events)
    attacked = [
        sample
        for sample in samples
        if sample.metadata["episode"]["condition"] == "C1"
        and sample.metadata["episode"]["model_alias"] == "fake_a"
    ]
    assert any(
        event.event == "tool" and event.function == "send_email"
        for sample in attacked
        for event in sample.events
    )


def test_parquet_roundtrip_preserves_nested_records_and_optional_ids(smoke_run, tmp_path):
    path = tmp_path / "roundtrip.parquet"
    clean = [episode for episode in smoke_run.episodes if episode.attack_id is None]
    write_episodes(path, clean)
    assert read_episodes(path) == clean
    write_episodes(path, smoke_run.episodes)
    assert read_episodes(path) == smoke_run.episodes
    schema = pq.read_schema(path)
    assert pa.types.is_boolean(schema.field("attack_success").type)
    assert pa.types.is_integer(schema.field("seed").type)
    for name in ("transcript", "tool_calls", "call_costs", "propagation_events", "defense_events"):
        assert pa.types.is_string(schema.field(name).type)
    assert not path.with_suffix(".parquet.tmp").exists()


def test_run_freezes_inputs_and_content_addressed_attack_artifacts(smoke_run):
    directory = smoke_run.directory
    inputs = directory / "inputs"
    assert json.loads((inputs / "config.json").read_text()) == smoke_run.plan.config.model_dump(
        mode="json"
    )
    assert json.loads((inputs / "tasks.json").read_text()) == [
        task.model_dump(mode="json") for task in smoke_run.plan.tasks
    ]
    assert json.loads((inputs / "models.json").read_text()) == {
        alias: model.model_dump(mode="json") for alias, model in smoke_run.plan.models.items()
    }
    artifacts = [load_attack(path) for path in (inputs / "attacks").glob("*.json")]
    assert artifacts == smoke_run.plan.artifacts
    assert all(
        (inputs / "attacks" / f"{attack.artifact_sha256}.json").is_file() for attack in artifacts
    )
    hashes = smoke_run.manifest["file_hashes"]
    required = {
        "inputs/config.json",
        "inputs/tasks.json",
        "inputs/models.json",
        "inputs/source_selections.json",
        "episodes.parquet",
    }
    assert required <= hashes.keys()
    assert any(path.endswith(".eval") for path in hashes)
    assert verify_run(directory)["valid"]


def test_smoke_analysis_uses_valid_evaluation_rows(smoke_run):
    summary = smoke_run.summary
    assert (
        summary["counts"]["total"]
        == summary["counts"]["valid"]
        == summary["counts"]["evaluation"]
        == 32
    )
    assert summary["counts"]["excluded"] == 0
    assert summary["counts"]["attack_trials"] == summary["counts"]["clean_trials"] == 16
    assert summary["provenance_failures"] == {}
    assert sum(group["estimated_cost_usd"] for group in summary["spending"]) == 0
    assert not summary["publishable"] and summary["simulated"]


# Budget/interrupt paths reuse the real solver but not another Inspect eval. ---


def test_budget_interrupt_prevents_provider_calls_and_excludes_outcome_evidence(
    tmp_path, monkeypatch
):
    path = make_config(
        tmp_path,
        entries={
            "fake_a": fake_entry(input_price=1, output_price=2),
            "fake_b": fake_entry("resistant", input_price=1, output_price=2),
        },
        scaffolds=["tool_agent"],
        limits={"max_turns": 20, "max_tokens": 64, "max_cost_usd": 0},
    )
    invocations = []

    async def unexpected_generate(*args, **kwargs):
        raise AssertionError("A zero-cap paid bound must prevent all model requests")

    monkeypatch.setattr(FakeModel, "generate", unexpected_generate)

    def evaluate(task, **kwargs):
        invocations.append(task)

        async def solve_batch():
            samples = []
            for sample in task.dataset:
                state = TaskState(
                    model=ModelName(kwargs["model"]),
                    sample_id=sample.id,
                    epoch=1,
                    input=sample.input,
                    messages=[],
                    metadata=copy.deepcopy(sample.metadata),
                )
                state = await task.solver(state, unexpected_generate)
                samples.append(SimpleNamespace(id=sample.id, metadata=state.metadata))
            return samples

        # This is deliberately not represented as a canonical Inspect log.
        return [
            SimpleNamespace(
                location=str(Path(kwargs["log_dir"]) / "budget-test-stub.json"),
                samples=asyncio.run(solve_batch()),
            )
        ]

    monkeypatch.setattr(matrix_module, "inspect_eval", evaluate)
    directory = run_matrix(path, run_id="budget-stop")
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = read_episodes(directory / "episodes.parquet")
    summary = json.loads((directory / "summary.json").read_text())
    assert len(invocations) == 1  # The second model's batch is never scheduled.
    assert manifest["status"] == "budget_exceeded" and not manifest["publishable"]
    assert manifest["n_completed"] == len(rows) == 4 < manifest["n_episodes"]
    assert all(
        row.status == "budget_exceeded"
        and row.error is not None
        and row.error.startswith("BudgetExceeded:")
        for row in rows
    )
    assert all(not row.tool_calls and not row.call_costs for row in rows)
    assert summary["counts"]["valid"] == summary["counts"]["evaluation"] == 0
    assert summary["counts"]["exclusions"] == {"execution_failure": 4}
    assert not summary["groups"] and not summary["defense_comparisons"]
    assert all(not matrix["cells"] for matrix in summary["matrices"].values())


def test_partial_budget_episode_retains_violations_and_spend_but_is_not_evidence(
    smoke_run, tmp_path
):
    item = next(
        episode
        for episode in smoke_run.plan.episodes
        if episode.attack and episode.defense_id == "none"
    )
    call = ToolCallRecord(
        name="send_email",
        arguments={"body": item.task.canaries["secret"]},
        allowed=False,
        executed=True,
    )
    cost = CallCost(provider="fake", model="vulnerable", estimated_cost_usd=0.25)
    result = EpisodeResult(
        output=item.task.documents[item.task.attack_surface.location],
        tool_calls=[call],
        costs=[cost],
        error="BudgetExceeded: continuation denied",
    )
    from transferbench.models.runtime import CostBudget

    row = matrix_module._episode(
        "partial-budget-run",
        item,
        smoke_run.plan.models[item.model_alias],
        result,
        smoke_run.plan.config,
        CostBudget(1),
    )
    assert row.status == "budget_exceeded" and row.attack_success and row.policy_violation
    assert not row.utility_success and row.estimated_cost_usd == 0.25
    atomic_json(
        tmp_path / "manifest.json",
        {
            "run_id": row.run_id,
            "publishable": False,
            "attack_artifacts": [item.attack.model_dump(mode="json")],
        },
    )
    write_episodes(tmp_path / "episodes.parquet", [row])
    summary = analyze_run(tmp_path)
    assert summary["counts"]["valid"] == summary["counts"]["attack_trials"] == 0
    assert summary["counts"]["exclusions"] == {"execution_failure": 1}
    assert summary["spending"][0]["estimated_cost_usd"] == 0.25
    assert summary["spending"][0]["n_failures"] == 1


@pytest.mark.parametrize(
    "exception,status", [(KeyboardInterrupt, "interrupted"), (RuntimeError, "failed")]
)
def test_run_records_failure_manifest_and_reraises(tmp_path, monkeypatch, exception, status):
    path = make_config(tmp_path, models=["fake_a"], scaffolds=["chat_single"])

    def interrupted(*args, **kwargs):
        raise exception("synthetic interruption")

    monkeypatch.setattr(matrix_module, "inspect_eval", interrupted)
    with pytest.raises(exception, match="synthetic interruption"):
        run_matrix(path, run_id="interrupted-run")
    manifest = json.loads((tmp_path / "results" / "interrupted-run" / "manifest.json").read_text())
    assert manifest["status"] == status
    assert manifest["error"] == f"{exception.__name__}: synthetic interruption"
    assert manifest["finished_at"] and not manifest["publishable"]
    assert manifest["n_completed"] == 0 and manifest["publication_blockers"]


# Provenance and publication guards -----------------------------------------


def clone_run(smoke_run, tmp_path):
    directory = tmp_path / "copied-run"
    shutil.copytree(smoke_run.directory, directory)
    return directory


@pytest.mark.parametrize(
    "relative", ["inputs/config.json", "inputs/tasks.json", "episodes.parquet", "canonical_log"]
)
def test_provenance_verification_refuses_changed_files(smoke_run, tmp_path, relative):
    directory = clone_run(smoke_run, tmp_path)
    if relative == "canonical_log":
        relative = next(
            name for name in smoke_run.manifest["file_hashes"] if name.endswith(".eval")
        )
    changed = directory / relative
    with changed.open("ab") as stream:
        stream.write(b"\nCHANGED")
    verification = verify_run(directory)
    assert not verification["valid"]
    assert f"Missing or changed: {relative}" in verification["errors"]
    assert not verification["publishable"]


def test_provenance_refuses_missing_files_and_path_escape(smoke_run, tmp_path):
    directory = clone_run(smoke_run, tmp_path)
    (directory / "inputs/tasks.json").unlink()
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["file_hashes"]["../outside"] = "0" * 64
    atomic_json(directory / "manifest.json", manifest)
    verification = verify_run(directory)
    assert not verification["valid"]
    assert "Missing or changed: inputs/tasks.json" in verification["errors"]
    assert "Unsafe provenance path: ../outside" in verification["errors"]


def test_provenance_refuses_changed_attack_payload(smoke_run, tmp_path):
    directory = clone_run(smoke_run, tmp_path)
    artifact_path = next((directory / "inputs/attacks").glob("*.json"))
    data = json.loads(artifact_path.read_text())
    data["payload"] += " modified"
    atomic_json(artifact_path, data)
    assert not verify_run(directory)["valid"]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_attack(artifact_path)


def test_integrity_verification_requires_records(tmp_path):
    atomic_json(tmp_path / "manifest.json", {"publishable": False})
    assert verify_run(tmp_path) == {
        "valid": False,
        "errors": ["No file integrity records"],
        "publishable": False,
    }


def clean_attestation():
    """Synthetic metadata for guard unit tests, not an attestation about a provider."""
    return {
        "commit": "a" * 40,
        "dirty": False,
        "dependencies_pinned": True,
        "models": {"provider_alias": {"provider": "openai", "version_pinned": True}},
        "budget_uncertain": False,
        "status": "completed",
    }


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"commit": None}, "No Git commit"),
        ({"dirty": True}, "uncommitted or untracked"),
        ({"dependencies_pinned": False}, "uv.lock"),
        ({"models": {"fake": {"provider": "fake", "version_pinned": True}}}, "Simulated models"),
        ({"models": {"real": {"provider": "openai", "version_pinned": False}}}, "immutable"),
        ({"budget_uncertain": True}, "uncertain usage"),
        ({"status": "budget_exceeded"}, "did not complete"),
    ],
)
def test_publication_guards_fail_closed_independently(updates, reason):
    assert publication_reasons(clean_attestation()) == []
    assert any(reason in blocker for blocker in publication_reasons(clean_attestation() | updates))


def test_fake_run_cannot_be_published_even_if_manifest_flag_is_overridden(smoke_run, tmp_path):
    directory = clone_run(smoke_run, tmp_path)
    assert not smoke_run.manifest["publishable"]
    assert any(
        "Simulated models" in blocker for blocker in smoke_run.manifest["publication_blockers"]
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["publishable"] = True
    atomic_json(directory / "manifest.json", manifest)
    before = (directory / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="SIMULATED"):
        analyze_run(directory, publishable=True)
    assert (directory / "summary.json").read_bytes() == before


def test_invalid_integrity_must_not_report_publishable(tmp_path):
    path = tmp_path / "evidence.txt"
    path.write_text("original")
    atomic_json(
        tmp_path / "manifest.json",
        {"publishable": True, "file_hashes": {"evidence.txt": sha256_file(path)}},
    )
    path.write_text("changed")
    verification = verify_run(tmp_path)
    assert not verification["valid"]
    assert verification["publishable"] is False


@pytest.mark.parametrize("run_id", ["..", "../escape", "nested/run", "/absolute/run"])
def test_run_id_cannot_escape_output_root(tmp_path, run_id):
    path = make_config(tmp_path)
    with pytest.raises(ValueError, match="single directory name"):
        run_matrix(path, run_id=run_id)


def test_existing_run_directory_is_never_overwritten(smoke_run):
    before = sha256_file(smoke_run.directory / "manifest.json")
    with pytest.raises(FileExistsError):
        run_matrix(
            smoke_run.config_path,
            output_root=smoke_run.directory.parent,
            run_id=smoke_run.directory.name,
        )
    assert sha256_file(smoke_run.directory / "manifest.json") == before
