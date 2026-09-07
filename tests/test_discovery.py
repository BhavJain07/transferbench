"""Selection correctness tests; fake/template tests are not real-model findings."""

import copy
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from transferbench.attacks.artifacts import attack_sha256, load_attack, save_attack
from transferbench.attacks.generator import (
    AttackGenerator,
    TemplateAttackGenerator,
    generate_unique_candidates,
)
from transferbench.attacks.library import DeterministicAttackGenerator
from transferbench.runner.discovery import (
    DiscoveryError,
    choose_holdout_seed,
    discover,
    freeze_selection,
    make_selection_config,
    make_transfer_config,
    rank_candidates,
    validate_source_run,
    verify_evaluation_holdout,
)
from transferbench.tasks.dataset import load_tasks


@pytest.fixture
def candidates():
    return generate_unique_candidates(
        load_tasks()[0],
        count=3,
        source_model="source",
        source_scaffold="tool_agent",
    )


def selection_rows(candidates, repeats=2):
    return [
        {
            "run_id": "source-selection-run",
            "episode_id": f"{candidate.artifact_sha256}-{repeat}",
            "model_alias": "source",
            "model_id": "fake/vulnerable",
            "scaffold_id": "tool_agent",
            "task_id": candidate.task_id,
            "task_family": "documents",
            "attack_id": candidate.id,
            "attack_family": candidate.family,
            "attack_artifact_sha256": candidate.artifact_sha256,
            "attack_source_model": "source",
            "attack_source_scaffold": "tool_agent",
            "selection_id": None,
            "split": "selection",
            "transfer_mode": "fixed",
            "defense_id": "none",
            "condition": "C1",
            "repeat": repeat,
            "seed": 1000 + repeat,
            "attack_success": index == 1 or (index == 0 and repeat == 0),
            "utility_success": True,
            "policy_violation": index == 1 or (index == 0 and repeat == 0),
            "status": "ok",
            "error": None,
        }
        for index, candidate in enumerate(candidates)
        for repeat in range(repeats)
    ]


def rank(rows, candidates, repeats=2):
    return rank_candidates(
        rows,
        candidates,
        source_model="source",
        source_scaffold="tool_agent",
        repeats=repeats,
        source_model_id="fake/vulnerable",
    )


def test_finite_candidates_deduplicate_payloads_not_seeded_identifiers(candidates):
    task = load_tasks()[0]
    assert len({candidate.payload for candidate in candidates}) == 3
    assert candidates == generate_unique_candidates(
        task,
        count=3,
        source_model="source",
        source_scaffold="tool_agent",
    )
    old_pool = DeterministicAttackGenerator().generate(task, count=6)
    assert len({candidate.id for candidate in old_pool}) == 6
    assert len({candidate.payload for candidate in old_pool}) == 3
    for requested in (0, -1, 4, 50, True):
        with pytest.raises(ValueError, match="unique candidates"):
            generate_unique_candidates(task, count=requested)
    with pytest.raises(ValueError, match="no-attack"):
        generate_unique_candidates(task, family="none")


async def test_async_protocol_wraps_actual_deterministic_templates(candidates):
    generator = TemplateAttackGenerator()
    assert isinstance(generator, AttackGenerator)
    assert inspect.iscoroutinefunction(generator.generate)
    assert (
        await generator.generate(
            load_tasks()[0],
            count=3,
            source_model="source",
            source_scaffold="tool_agent",
        )
        == candidates
    )


def test_generator_rejects_shortfall_after_actual_payload_dedup(monkeypatch):
    from transferbench.attacks import generator

    original = generator.SyncTemplateGenerator.generate

    def duplicated(self, task, **kwargs):
        pool = original(self, task, **kwargs)
        return [pool[0], pool[0], pool[0]]

    monkeypatch.setattr(generator.SyncTemplateGenerator, "generate", duplicated)
    with pytest.raises(ValueError, match="Only 1 distinct payloads"):
        generate_unique_candidates(load_tasks()[0], count=2)


def test_rank_uses_complete_source_asr_and_preserves_zero_success(candidates):
    scores = rank(selection_rows(candidates), candidates)[candidates[0].task_id]
    assert [score.attack.id for score in scores] == [
        candidates[1].id,
        candidates[0].id,
        candidates[2].id,
    ]
    assert [score.asr for score in scores] == [1.0, 0.5, 0.0]
    assert all(score.n == 2 and score.seeds == (1000, 1001) for score in scores)
    assert scores[0].as_dict()["candidate_artifact_sha256"] == candidates[1].artifact_sha256


def test_rank_ties_and_tasks_are_deterministic(candidates):
    other = generate_unique_candidates(
        load_tasks()[1],
        count=3,
        source_model="source",
        source_scaffold="tool_agent",
    )
    combined = candidates + other
    rows = selection_rows(combined)
    for row in rows:
        row["attack_success"] = False
        row["policy_violation"] = False
    scores = rank(rows, combined)
    assert scores == rank(list(reversed(rows)), list(reversed(combined)))
    assert len(scores) == 2
    for task_scores in scores.values():
        assert [score.attack.artifact_sha256 for score in task_scores] == sorted(
            score.attack.artifact_sha256 for score in task_scores
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "error"},
        {"status": "budget_exceeded"},
        {"error": "interrupted"},
        {"split": "evaluation"},
        {"transfer_mode": "source_optimized"},
        {"defense_id": "tool_policy"},
        {"condition": "C2"},
        {"model_alias": "target"},
        {"scaffold_id": "chat_single"},
        {"attack_source_model": "target"},
        {"attack_source_scaffold": "chat_single"},
        {"attack_artifact_sha256": "0" * 64},
        {"attack_id": "wrong"},
        {"task_id": "wrong"},
        {"attack_family": "direct"},
        {"selection_id": "already-selected"},
        {"attack_success": 1},
        {"attack_success": None},
        {"utility_success": "true"},
        {"policy_violation": None},
        {"repeat": True},
        {"repeat": -1},
        {"repeat": 2},
        {"seed": True},
        {"seed": -1},
        {"seed": "1000"},
        {"episode_id": None},
        {"model_id": "fake/resistant"},
        {"run_id": "a-different-run"},
    ],
)
def test_rank_refuses_invalid_comparisons_instead_of_counting_errors(candidates, updates):
    rows = selection_rows(candidates)
    rows[0].update(updates)
    with pytest.raises(DiscoveryError):
        rank(rows, candidates)


def test_rank_rejects_missing_duplicate_or_unpaired_trials(candidates):
    rows = selection_rows(candidates)
    with pytest.raises(DiscoveryError, match="Incomplete"):
        rank(rows[:-1], candidates)
    with pytest.raises(DiscoveryError, match="Duplicate"):
        rank(rows + [rows[0]], candidates)
    unpaired = copy.deepcopy(rows)
    unpaired[0]["seed"] += 10
    with pytest.raises(DiscoveryError, match="paired"):
        rank(unpaired, candidates)
    reused_seed = copy.deepcopy(rows)
    reused_seed[1]["seed"] = reused_seed[0]["seed"]
    with pytest.raises(DiscoveryError, match="independent"):
        rank(reused_seed, candidates)
    duplicate_identity = copy.deepcopy(rows)
    duplicate_identity[2]["episode_id"] = duplicate_identity[0]["episode_id"]
    with pytest.raises(DiscoveryError, match="identity"):
        rank(duplicate_identity, candidates)


def test_rank_rejects_duplicate_payloads_and_changed_candidate_hashes(candidates):
    task = load_tasks()[0]
    duplicates = DeterministicAttackGenerator().generate(
        task,
        count=4,
        source_model="source",
        source_scaffold="tool_agent",
    )
    with pytest.raises(DiscoveryError, match="Duplicate payloads"):
        rank(selection_rows(duplicates), duplicates)
    changed = candidates[0].model_copy(update={"payload": "changed"})
    with pytest.raises(DiscoveryError, match="SHA-256"):
        rank(selection_rows(candidates), [changed, *candidates[1:]])


def test_freezing_adds_selection_provenance_and_rehashes_without_overwrite(candidates, tmp_path):
    original = candidates[0]
    candidate_path = tmp_path / "candidate.json"
    save_attack(original, candidate_path)
    original_bytes = candidate_path.read_bytes()
    score = next(
        score
        for score in rank(selection_rows(candidates), candidates)[original.task_id]
        if score.attack == original
    )
    path = tmp_path / "selected.json"
    frozen = freeze_selection(score, path, selection_id="selection-test", git_commit="a" * 40)
    assert frozen.payload == original.payload
    assert frozen.selection_id == "selection-test" and frozen.git_commit == "a" * 40
    assert frozen.artifact_sha256 == attack_sha256(frozen) != original.artifact_sha256
    assert load_attack(path) == frozen
    assert candidate_path.read_bytes() == original_bytes
    assert original.selection_id is None and original.git_commit is None
    with pytest.raises(FileExistsError):
        freeze_selection(score, path, selection_id="selection-other", git_commit=None)
    with pytest.raises(DiscoveryError, match="selection_id"):
        freeze_selection(score, tmp_path / "bad.json", selection_id="", git_commit=None)
    corrupted = replace(score, attack=score.attack.model_copy(update={"payload": "tampered"}))
    with pytest.raises(DiscoveryError, match="SHA-256"):
        freeze_selection(corrupted, tmp_path / "bad.json", selection_id="new", git_commit=None)


def test_selection_and_transfer_configs_are_portable_and_preserve_all_targets(tmp_path):
    from transferbench.runner.config import MatrixConfig

    config = MatrixConfig.model_validate(
        {
            "models": ["target"],
            "scaffolds": ["chat_single", "delegate_agent"],
            "model_registry": "registry.yaml",
            "defenses": ["tool_policy"],
            "tasks": {"paths": ["fixtures.jsonl"], "limit": 2},
            "repeats": 5,
            "vulnerable_from": "prior-run",
            "output_dir": "outputs",
        }
    )
    before = config.model_dump()
    artifacts = [tmp_path / "candidate.json"]
    source = make_selection_config(
        config,
        tmp_path,
        source_model="source",
        source_scaffold="tool_agent",
        artifact_paths=artifacts,
    )
    assert source.models == ["source"] and source.scaffolds == ["tool_agent"]
    assert source.experiment.stage == "selection" and source.experiment.transfer_mode == "fixed"
    assert source.defenses == ["none"] and not source.include_controls
    assert source.repeats == 5 and source.seed == config.seed
    assert (
        source.attacks == []
        and source.selection_manifest is None
        and source.vulnerable_from is None
    )
    assert source.model_registry == str(tmp_path / "registry.yaml")
    assert source.tasks.paths == [str(tmp_path / "fixtures.jsonl")]
    assert source.output_dir == str(tmp_path / "outputs")
    frozen = [tmp_path / "frozen.json"]
    transfer = make_transfer_config(
        config,
        tmp_path,
        source_model="source",
        source_scaffold="tool_agent",
        artifact_paths=frozen,
        selection_manifest=tmp_path / "selection.json",
        evaluation_seed=100,
        task_snapshot=tmp_path / "snapshot.jsonl",
    )
    assert transfer.models == ["source", "target"]
    assert transfer.scaffolds == ["tool_agent", "chat_single", "delegate_agent"]
    assert transfer.defenses == ["none", "tool_policy"]
    assert transfer.experiment.stage == "evaluation"
    assert transfer.experiment.transfer_mode == "source_optimized"
    assert transfer.attack_artifacts == [str(tmp_path / "frozen.json")]
    assert transfer.selection_manifest == str(tmp_path / "selection.json")
    assert transfer.seed != config.seed and transfer.repeats == config.repeats
    assert (
        transfer.tasks.paths == [str(tmp_path / "snapshot.jsonl")] and transfer.tasks.limit is None
    )
    assert transfer.include_controls == config.include_controls
    assert config.model_dump() == before
    with pytest.raises(DiscoveryError, match="different base seed"):
        make_transfer_config(
            config,
            tmp_path,
            source_model="source",
            source_scaffold="tool_agent",
            artifact_paths=frozen,
            selection_manifest=tmp_path / "selection.json",
            evaluation_seed=config.seed,
        )


def test_holdout_seed_checks_actual_planned_episode_seeds():
    seen = []

    def planned(base):
        seen.append(base)
        return {1000, 1001} if base == 44 else {2000, 2001}

    assert (
        choose_holdout_seed(42, repeats=2, selection_seeds={1000, 1001}, planned_seeds=planned)
        == 45
    )
    assert seen == [44, 45]
    assert (
        choose_holdout_seed(
            2**31 - 1, repeats=2, selection_seeds={1000}, planned_seeds=lambda base: {base}
        )
        == 1
    )
    with pytest.raises(DiscoveryError, match="nonoverlapping"):
        choose_holdout_seed(
            42, repeats=2, selection_seeds={1000}, planned_seeds=lambda base: {1000}, max_attempts=3
        )


@pytest.fixture
def holdout_evidence():
    trials = [("task-a", 0, 200), ("task-a", 1, 201), ("task-b", 0, 300), ("task-b", 1, 301)]
    evidence = {
        "source_selections": [
            {
                "task_id": "task-a",
                "split": "selection",
                "selected": True,
                "selection_seed": 47,
                "n": 2,
                "trial_seeds": [100, 101],
            },
            {
                "task_id": "task-a",
                "split": "selection",
                "selected": True,
                "selection_seed": 47,
                "n": 2,
                "trial_seeds": [100, 101],
            },
            {
                "task_id": "task-b",
                "split": "selection",
                "selected": True,
                "selection_seed": 47,
                "n": 2,
                "trial_seeds": [110, 111],
            },
        ],
        "holdout": {
            "unit": "trials",
            "same_tasks": True,
            "new_task_holdout": False,
            "task_ids": ["task-a", "task-b"],
            "selection_base_seed": 47,
            "evaluation_base_seed": 49,
            "selection_episode_seeds": [100, 101, 110, 111],
            "evaluation_episode_seeds": [200, 201, 300, 301],
            "seed_overlap": False,
            "evaluation_trials": [
                {"task_id": task, "repeat": repeat, "seed": seed} for task, repeat, seed in trials
            ],
        },
    }
    return evidence, trials


def test_verify_holdout_accepts_repeated_plan_cells_and_older_evidence(holdout_evidence):
    evidence, trials = holdout_evidence
    assert verify_evaluation_holdout(evidence, trials * 3, evaluation_base_seed=49) is None
    # Previously generated discovery manifests lack the per-trial mapping and
    # compatibility field but still carry explicit source/evaluation seed sets.
    older = copy.deepcopy(evidence)
    del older["holdout"]["evaluation_trials"]
    for record in older["source_selections"]:
        del record["selection_seed"]
    assert verify_evaluation_holdout(older, trials, evaluation_base_seed=49) is None


@pytest.mark.parametrize(
    "path,value",
    [
        (("holdout",), None),
        (("source_selections",), []),
        (("holdout", "unit"), "tasks"),
        (("holdout", "same_tasks"), False),
        (("holdout", "new_task_holdout"), True),
        (("holdout", "selection_base_seed"), True),
        (("holdout", "evaluation_base_seed"), "49"),
        (("holdout", "task_ids"), ["task-a"]),
        (("holdout", "task_ids"), ["task-a", "task-a"]),
        (("holdout", "selection_episode_seeds"), []),
        (("holdout", "selection_episode_seeds"), [100, 101]),
        (("holdout", "selection_episode_seeds"), [True, 101, 110, 111]),
        (("holdout", "evaluation_episode_seeds"), [200, 201, 300]),
        (("holdout", "evaluation_episode_seeds"), [200, 201, 300, 301, 301]),
        (("source_selections", 0, "n"), True),
        (("source_selections", 0, "n"), 3),
        (("source_selections", 0, "selected"), False),
        (("source_selections", 0, "split"), "evaluation"),
        (("source_selections", 0, "selection_seed"), 48),
        (("source_selections", 0, "trial_seeds"), [100, 120]),
        (("holdout", "evaluation_trials", 0, "seed"), 201),
    ],
)
def test_verify_holdout_refuses_missing_or_contradictory_evidence(holdout_evidence, path, value):
    evidence, trials = holdout_evidence
    target = evidence
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(DiscoveryError):
        verify_evaluation_holdout(evidence, trials, evaluation_base_seed=49)


@pytest.mark.parametrize(
    "mode",
    ["overlap", "missing", "empty", "conflict", "new_task", "swapped", "bad_repeat", "malformed"],
)
def test_verify_holdout_checks_actual_planned_trials(holdout_evidence, mode):
    evidence, trials = holdout_evidence
    if mode == "overlap":
        trials[0] = ("task-a", 0, 100)
    elif mode == "missing":
        trials.pop()
    elif mode == "empty":
        trials = []
    elif mode == "conflict":
        trials.append(("task-a", 0, 999))
    elif mode == "new_task":
        trials[0] = ("task-new", 0, 200)
    elif mode == "swapped":
        trials[0], trials[1] = ("task-a", 0, 201), ("task-a", 1, 200)
    elif mode == "bad_repeat":
        trials[0] = ("task-a", True, 200)
    elif mode == "malformed":
        trials[0] = ("task-a", 200)
    with pytest.raises(DiscoveryError):
        verify_evaluation_holdout(evidence, trials, evaluation_base_seed=49)


@pytest.mark.parametrize("base", [47, 50, True])
def test_verify_holdout_rejects_changed_evaluation_base_seed(holdout_evidence, base):
    evidence, trials = holdout_evidence
    with pytest.raises(DiscoveryError):
        verify_evaluation_holdout(evidence, trials, evaluation_base_seed=base)


@pytest.mark.parametrize(
    "update",
    [
        {"status": "partial"},
        {"status": "running"},
        {"run_id": "different"},
        {"budget_exhausted": True},
        {"cost_uncertain": True},
        {"budget_uncertain": True},
        {"n_completed": 5},
        {"n_episodes": 7},
        {"n_completed": "6"},
        {"n_episodes": True},
        {"estimated_cost_usd": 0.5},
        {"estimated_cost_usd": float("nan")},
        {"estimated_cost_usd": True},
        {"error": "interrupted despite completed status"},
        {"budget": {"exhausted": True}},
        {"budget": {"uncertain": True}},
        {"cost_budget": {"uncertain": True}},
    ],
)
def test_selection_refuses_interrupted_or_uncertain_run_manifest(candidates, update):
    rows = selection_rows(candidates)
    for row in rows:
        row["estimated_cost_usd"] = 0.0
    manifest = {"run_id": "source-selection-run", "status": "completed", **update}
    with pytest.raises(DiscoveryError):
        validate_source_run(manifest, rows, expected_episodes=len(rows), max_cost_usd=0)


def test_source_run_validation_checks_completeness_and_costs(candidates):
    rows = selection_rows(candidates)
    for row in rows:
        row["estimated_cost_usd"] = 0.1
    manifest = {
        "run_id": "source-selection-run",
        "status": "completed",
        "budget": {"exhausted": False, "uncertain": False},
        "budget_uncertain": False,
        "n_completed": 6,
        "n_episodes": 6,
        "estimated_cost_usd": 0.6,
    }
    assert validate_source_run(
        manifest, rows, expected_episodes=6, max_cost_usd=1
    ) == pytest.approx(0.6)
    with pytest.raises(DiscoveryError, match="incomplete"):
        validate_source_run(manifest, rows[:-1], expected_episodes=6, max_cost_usd=1)
    with pytest.raises(DiscoveryError, match="ceiling"):
        validate_source_run(manifest, rows, expected_episodes=6, max_cost_usd=0.5)
    for invalid in (float("nan"), -1, None, "0.1", True):
        changed = copy.deepcopy(rows)
        changed[0]["estimated_cost_usd"] = invalid
        with pytest.raises(DiscoveryError, match="cost accounting"):
            validate_source_run(manifest, changed, expected_episodes=6, max_cost_usd=1)
    changed = copy.deepcopy(rows)
    changed[0]["call_costs"] = [{"purpose": "assistant:unknown_usage_reserved"}]
    with pytest.raises(DiscoveryError, match="uncertain provider usage"):
        validate_source_run(manifest, changed, expected_episodes=6, max_cost_usd=1)


@pytest.fixture
def discovery_config(tmp_path):
    import yaml

    registry = {
        "models": {
            "source": {
                "provider": "fake",
                "model": "vulnerable",
                "family": "synthetic",
                "pricing": {"input_per_million": 0, "output_per_million": 0},
                "supports_seed": True,
            },
            "target": {
                "provider": "fake",
                "model": "resistant",
                "family": "synthetic",
                "pricing": {"input_per_million": 0, "output_per_million": 0},
                "supports_seed": True,
            },
        }
    }
    (tmp_path / "models.yaml").write_text(yaml.safe_dump(registry))
    config = {
        "experiment": {"name": "discovery-test"},
        "models": ["target"],
        "scaffolds": ["chat_single"],
        "tasks": {"suites": ["workspace"], "limit": 1},
        "repeats": 2,
        "seed": 47,
        "output_dir": "results",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def fake_matrix_contract(monkeypatch, failure=None):
    """An explicit runner test double, not a claim of Inspect execution."""
    import polars as pl

    import transferbench.runner

    def seed_for(base, task_id, repeat):
        return (base + int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) + repeat) % (2**31)

    planned, invoked = [], []

    def plan(config_path, config_override=None):
        from transferbench.runner.config import config_tasks

        assert config_override is not None
        planned.append(config_override)
        return SimpleNamespace(
            episodes=[
                SimpleNamespace(
                    task=task,
                    repeat=repeat,
                    seed=seed_for(config_override.seed, task.task_id, repeat),
                )
                for task in config_tasks(config_override, Path(config_path).parent)
                for repeat in range(config_override.repeats)
            ]
        )

    def run(config_path, *, output_root=None, config_override=None, run_id=None):
        assert config_override is not None
        assert output_root is not None and run_id is not None
        invoked.append(config_override)
        config = config_override
        assert config.experiment.stage == "selection" and config.experiment.transfer_mode == "fixed"
        assert config.defenses == ["none"] and config.include_controls is False
        attacks = [load_attack(path) for path in config.attack_artifacts]
        rows = selection_rows(attacks, config.repeats)
        for row in rows:
            row.update(
                {
                    "run_id": run_id,
                    "estimated_cost_usd": 0.0,
                    "seed": seed_for(config.seed, row["task_id"], row["repeat"]),
                }
            )
        manifest = {
            "run_id": run_id,
            "status": "completed",
            "budget": {"exhausted": failure == "budget", "uncertain": False},
        }
        if failure == "missing":
            rows.pop()
        elif failure == "error":
            rows[0]["status"] = "error"
        elif failure == "seed_mismatch":
            for row in rows:
                row["seed"] += 100
        directory = output_root / run_id
        directory.mkdir(parents=True)
        pl.DataFrame(rows).write_parquet(directory / "episodes.parquet")
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return directory

    module = SimpleNamespace(episode_seed=seed_for, plan_matrix=plan, run_matrix=run)
    monkeypatch.setattr(transferbench.runner, "matrix", module, raising=False)
    return planned, invoked


def assert_discovery_outputs(directory, expected_selections=2):
    from transferbench.runner.config import load_config

    manifest = json.loads((directory / "selection.json").read_text())
    transfer = load_config(directory / "transfer.yaml")
    records = manifest["source_selections"]
    assert len(records) == expected_selections
    assert len({record["selection_id"] for record in records}) == expected_selections
    for record in records:
        artifact = load_attack(record["artifact_path"])
        candidate = load_attack(record["candidate_artifact_path"])
        assert (
            record["attack_artifact_sha256"]
            == artifact.artifact_sha256
            != candidate.artifact_sha256
        )
        assert record["candidate_artifact_sha256"] == candidate.artifact_sha256
        assert artifact.payload == candidate.payload and artifact.id == candidate.id
        assert artifact.selection_id == record["selection_id"]
        assert artifact.source_model == record["source_model"] == "source"
        assert artifact.source_scaffold == record["source_scaffold"] == "tool_agent"
        assert record["split"] == "selection" and record["selected"] is True
        assert record["selection_seed"] == manifest["holdout"]["selection_base_seed"]
        assert record["n"] == record["n_valid"] == len(record["source_episode_ids"]) == 2
        assert record["n_errors"] == 0
        assert Path(record["source_run_path"]).is_dir()
        assert artifact.git_commit == record["git_commit"] == manifest["git_commit"]
    assert transfer.models == ["source", "target"] and transfer.scaffolds == [
        "tool_agent",
        "chat_single",
    ]
    assert (
        transfer.experiment.stage == "evaluation"
        and transfer.experiment.transfer_mode == "source_optimized"
    )
    assert transfer.selection_manifest == str(directory / "selection.json")
    assert set(transfer.attack_artifacts) == {record["artifact_path"] for record in records}
    holdout = manifest["holdout"]
    assert holdout["same_tasks"] is True and holdout["new_task_holdout"] is False
    assert holdout["selection_base_seed"] != holdout["evaluation_base_seed"]
    assert set(holdout["selection_episode_seeds"]).isdisjoint(holdout["evaluation_episode_seeds"])
    assert manifest["generator"] == "finite_synthetic_templates"
    return manifest, transfer


def test_discover_orchestrates_declared_matrix_contract(discovery_config, monkeypatch):
    planned, invoked = fake_matrix_contract(monkeypatch)
    directory = discover(
        discovery_config, source_model="source", source_scaffold="tool_agent", top_k=2
    )
    assert directory.parent == discovery_config.parent / "results"
    manifest, _ = assert_discovery_outputs(directory)
    assert len(invoked) == 1 and len(planned) == 2
    assert len(manifest["candidate_scores"]) == 3
    assert len(list((directory / "candidates").glob("*.json"))) == 3
    assert len(list((directory / "selected").glob("*.json"))) == 2


@pytest.mark.parametrize("failure", ["budget", "missing", "error", "seed_mismatch"])
def test_discovery_publishes_no_selection_for_invalid_source_runs(
    discovery_config, monkeypatch, failure
):
    fake_matrix_contract(monkeypatch, failure)
    with pytest.raises(DiscoveryError):
        discover(discovery_config, source_model="source", source_scaffold="tool_agent")
    root = discovery_config.parent / "results"
    assert not list(root.rglob("selection.json"))
    assert not list(root.rglob("transfer.yaml"))
    assert not list(root.glob("*/selected/*.json"))
    assert list(root.rglob("episodes.parquet")), "Failed source evidence must remain inspectable"


def test_discovery_rejects_fifty_templates_before_any_source_run(discovery_config, monkeypatch):
    _, invoked = fake_matrix_contract(monkeypatch)
    with pytest.raises(DiscoveryError, match="distinct candidates"):
        discover(
            discovery_config, source_model="source", source_scaffold="tool_agent", candidates=50
        )
    assert not invoked
    assert not (discovery_config.parent / "results").exists()


@pytest.mark.parametrize("task_limit", [1, 2])
def test_discovery_smoke_with_production_matrix_and_inspect_fake_models(
    discovery_config,
    tmp_path,
    monkeypatch,
    task_limit,
):
    import polars as pl
    import yaml

    from transferbench.runner import matrix

    input_config = yaml.safe_load(discovery_config.read_text())
    input_config["tasks"]["limit"] = task_limit
    discovery_config.write_text(yaml.safe_dump(input_config))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    directory = discover(
        discovery_config, source_model="source", source_scaffold="tool_agent", top_k=2
    )
    manifest, transfer_config = assert_discovery_outputs(
        directory, expected_selections=2 * task_limit
    )
    source_run = Path(manifest["source_run_path"])
    source_rows = pl.read_parquet(source_run / "episodes.parquet").to_dicts()
    source_manifest = json.loads((source_run / "manifest.json").read_text())
    assert len(source_rows) == 6 * task_limit
    assert source_manifest["n_episodes"] == source_manifest["n_completed"] == len(source_rows)
    assert source_manifest["status"] == "completed" and not source_manifest["budget_uncertain"]
    assert Path(source_manifest["config"]) == directory / "selection-config.yaml"
    assert all(
        row["status"] == "ok" and row["split"] == "selection" and row["simulated"]
        for row in source_rows
    )
    assert all((source_run / row["inspect_log"]).is_file() for row in source_rows)
    for selection in manifest["source_selections"]:
        observed = [
            row
            for row in source_rows
            if row["attack_artifact_sha256"] == selection["candidate_artifact_sha256"]
        ]
        assert len(observed) == selection["n"] == 2
        assert selection["successes"] == sum(row["attack_success"] for row in observed)
        assert selection["source_asr"] == selection["successes"] / selection["n"]
        assert all(row["task_id"] == selection["task_id"] for row in observed)
    plan = matrix.plan_matrix(directory / "transfer.yaml")
    assert plan.config.selection_manifest == str(directory / "selection.json")
    assert len(plan.artifacts) == 2 * task_limit
    assert len(plan.episodes) == 24 * task_limit
    verify_evaluation_holdout(
        manifest,
        ((item.task.task_id, item.repeat, item.seed) for item in plan.episodes),
        evaluation_base_seed=transfer_config.seed,
    )
    assert all(
        item.attack is None or item.attack.task_id == item.task.task_id for item in plan.episodes
    )
    target_run = matrix.run_matrix(directory / "transfer.yaml", output_root=tmp_path / "held-out")
    target_rows = pl.read_parquet(target_run / "episodes.parquet").to_dicts()
    assert len(target_rows) == 24 * task_limit
    assert all(
        row["status"] == "ok" and row["split"] == "evaluation" and row["simulated"]
        for row in target_rows
    )
    assert {row["model_alias"] for row in target_rows} == {"source", "target"}
    assert {row["scaffold_id"] for row in target_rows} == {"tool_agent", "chat_single"}
    assert {row["seed"] for row in source_rows}.isdisjoint(row["seed"] for row in target_rows)
    target_manifest = json.loads((target_run / "manifest.json").read_text())
    assert target_manifest["n_episodes"] == target_manifest["n_completed"] == len(target_rows)
    assert target_manifest["status"] == "completed" and not target_manifest["budget_uncertain"]
    assert Path(target_manifest["config"]) == directory / "transfer.yaml"
    assert target_manifest["source_selections"] == manifest["source_selections"]
    for selection in manifest["source_selections"]:
        canonical = load_attack(
            target_run / "inputs" / "attacks" / f"{selection['attack_artifact_sha256']}.json"
        )
        assert canonical == load_attack(selection["artifact_path"])
    assert all((target_run / row["inspect_log"]).is_file() for row in target_rows)
