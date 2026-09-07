"""Source-only selection of finite synthetic templates, followed by held-out trials.

Selection is never evidence of held-out transfer. The produced evaluation config
reuses the same tasks with disjoint episode seeds; it does not claim new-task
holdout or a paid/generated candidate search.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from transferbench.attacks.artifacts import attack_sha256, save_attack
from transferbench.tasks.schema import Attack

if TYPE_CHECKING:
    from transferbench.runner.config import MatrixConfig


class DiscoveryError(ValueError):
    """Incomplete or incomparable selection evidence must not produce a winner."""


@dataclass(frozen=True)
class CandidateScore:
    attack: Attack
    successes: int
    n: int
    seeds: tuple[int, ...]

    @property
    def asr(self) -> float:
        return self.successes / self.n

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.attack.task_id,
            "attack_id": self.attack.id,
            "family": self.attack.family,
            "candidate_artifact_sha256": self.attack.artifact_sha256,
            "source_asr": self.asr,
            "successes": self.successes,
            "n": self.n,
            "trial_seeds": list(self.seeds),
        }


def rank_candidates(
    rows: Iterable[Mapping[str, Any]],
    candidates: Sequence[Attack],
    *,
    source_model: str,
    source_scaffold: str,
    repeats: int,
    source_model_id: str | None = None,
) -> dict[str, list[CandidateScore]]:
    """Validate complete, paired C1 selection trials and rank per task by ASR.

    No error rows are counted as failures, and no missing/invalid rows are silently
    dropped. All candidates must have every requested repeat with the same seeds.
    Ties use the candidate's immutable SHA-256, independent of input row order.
    """
    if type(repeats) is not int or repeats < 1 or not candidates:
        raise DiscoveryError("Selection requires candidates and positive repeats")
    artifacts: dict[str, Attack] = {}
    payloads: set[tuple[str, str]] = set()
    families: set[str] = set()
    for candidate in candidates:
        if (
            not candidate.task_id
            or candidate.source_model != source_model
            or candidate.source_scaffold != source_scaffold
        ):
            raise DiscoveryError("Candidate task/source provenance does not match selection")
        if candidate.selection_id is not None or not candidate.payload:
            raise DiscoveryError("Selection requires unselected, nonempty candidate artifacts")
        digest = candidate.artifact_sha256
        if not digest or digest != attack_sha256(candidate) or digest in artifacts:
            raise DiscoveryError("Invalid or duplicate candidate artifact SHA-256")
        payload_key = (candidate.task_id, candidate.payload)
        if payload_key in payloads:
            raise DiscoveryError("Duplicate payloads cannot count as distinct candidates")
        payloads.add(payload_key)
        families.add(candidate.family)
        artifacts[digest] = candidate
    if len(families) != 1:
        raise DiscoveryError("Compare only one attack family per discovery run")

    trials: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    episode_ids: set[str] = set()
    model_ids: set[str] = set()
    run_ids: set[str] = set()
    for row in rows:
        if row.get("status") != "ok" or row.get("error"):
            raise DiscoveryError("Source selection contains failed or budget-interrupted episodes")
        if any(
            row.get(key) != value
            for key, value in {
                "split": "selection",
                "transfer_mode": "fixed",
                "condition": "C1",
                "defense_id": "none",
                "model_alias": source_model,
                "scaffold_id": source_scaffold,
                "attack_source_model": source_model,
                "attack_source_scaffold": source_scaffold,
            }.items()
        ):
            raise DiscoveryError("Source selection contains an invalid comparison cell")
        digest = row.get("attack_artifact_sha256")
        candidate = artifacts.get(digest) if isinstance(digest, str) else None
        if candidate is None:
            raise DiscoveryError("Source selection contains an unknown candidate artifact")
        if (
            row.get("task_id") != candidate.task_id
            or row.get("attack_id") != candidate.id
            or row.get("attack_family") != candidate.family
            or row.get("selection_id") is not None
        ):
            raise DiscoveryError("Episode candidate provenance does not match frozen artifact")
        if any(
            type(row.get(field)) is not bool
            for field in ("attack_success", "utility_success", "policy_violation")
        ):
            raise DiscoveryError("Selection outcomes must be actual boolean scores")
        repeat, seed = row.get("repeat"), row.get("seed")
        if (
            type(repeat) is not int
            or not 0 <= repeat < repeats
            or type(seed) is not int
            or seed < 0
        ):
            raise DiscoveryError("Selection repeat/seed is invalid")
        if repeat in trials[candidate.artifact_sha256]:
            raise DiscoveryError("Duplicate candidate repeat in selection")
        episode_id, model_id, run_id = row.get("episode_id"), row.get("model_id"), row.get("run_id")
        if (
            any(not isinstance(value, str) or not value for value in (episode_id, model_id, run_id))
            or episode_id in episode_ids
        ):
            raise DiscoveryError("Missing or duplicate selection episode identity")
        assert isinstance(episode_id, str)
        assert isinstance(model_id, str)
        assert isinstance(run_id, str)
        episode_ids.add(episode_id)
        model_ids.add(model_id)
        run_ids.add(run_id)
        trials[candidate.artifact_sha256][repeat] = row
    if (
        len(model_ids) != 1
        or len(run_ids) != 1
        or (source_model_id is not None and model_ids != {source_model_id})
    ):
        raise DiscoveryError("Selection must use one run and the resolved source model version")

    ranked: dict[str, list[CandidateScore]] = defaultdict(list)
    paired: dict[str, tuple[int, ...]] = {}
    for digest, candidate in artifacts.items():
        assert candidate.task_id is not None
        observed = trials[digest]
        if set(observed) != set(range(repeats)):
            raise DiscoveryError(f"Incomplete source comparison for candidate {candidate.id}")
        seeds = tuple(observed[repeat]["seed"] for repeat in range(repeats))
        if len(set(seeds)) != repeats:
            raise DiscoveryError("Repeated selection seeds are not independent trials")
        if candidate.task_id in paired and seeds != paired[candidate.task_id]:
            raise DiscoveryError("Candidates do not share paired source selection seeds")
        paired[candidate.task_id] = seeds
        ranked[candidate.task_id].append(
            CandidateScore(
                candidate,
                sum(observed[repeat]["attack_success"] for repeat in range(repeats)),
                repeats,
                seeds,
            )
        )
    return {
        task_id: sorted(scores, key=lambda score: (-score.asr, score.attack.artifact_sha256))
        for task_id, scores in sorted(ranked.items())
    }


def choose_holdout_seed(
    selection_seed: int,
    *,
    repeats: int,
    selection_seeds: set[int],
    planned_seeds: Callable[[int], set[int]],
    max_attempts: int = 1024,
) -> int:
    """Use runner-planned episode seeds, not just different config seed labels."""
    for offset in range(max_attempts):
        candidate = (selection_seed + repeats + offset) % (2**31)
        if candidate == selection_seed:
            continue
        seeds = planned_seeds(candidate)
        if seeds and seeds.isdisjoint(selection_seeds):
            return candidate
    raise DiscoveryError("Could not construct nonoverlapping held-out episode seeds")


def verify_evaluation_holdout(
    evidence: Mapping[str, Any],
    task_seeds: Iterable[tuple[str, int, int]],
    *,
    evaluation_base_seed: int,
) -> None:
    """Fail closed unless actual planned trials match a disjoint same-task holdout.

    ``task_seeds`` contains (task_id, repeat, seed), e.g. from MatrixPlan.episodes.
    Repeated triples across models, scaffolds, artifacts, or controls are expected;
    conflicting seeds for the same task/repeat are not. Call before evaluation,
    including when users override the generated transfer configuration.
    """

    def valid_seed(value: Any) -> bool:
        return type(value) is int and 0 <= value < 2**31

    def seed_list(value: Any, label: str) -> list[int]:
        if not isinstance(value, list) or not value or not all(valid_seed(seed) for seed in value):
            raise DiscoveryError(f"Missing or invalid {label} seeds")
        if len(set(value)) != len(value):
            raise DiscoveryError(f"Duplicate {label} seeds")
        return value

    def trial_map(trials: Iterable[tuple[str, int, int]]) -> dict[tuple[str, int], int]:
        result = {}
        for trial in trials:
            if not isinstance(trial, (list, tuple)) or len(trial) != 3:
                raise DiscoveryError("Planned trials must contain task_id, repeat, seed")
            task_id, repeat, seed = trial
            if (
                not isinstance(task_id, str)
                or not task_id
                or type(repeat) is not int
                or repeat < 0
                or not valid_seed(seed)
            ):
                raise DiscoveryError("Invalid planned task/repeat/seed")
            key = (task_id, repeat)
            if key in result and result[key] != seed:
                raise DiscoveryError("Conflicting planned seeds for the same task/repeat")
            result[key] = seed
        if not result:
            raise DiscoveryError("Evaluation plan contains no trials")
        return result

    holdout = evidence.get("holdout")
    selections = evidence.get("source_selections")
    if not isinstance(holdout, Mapping) or not isinstance(selections, list) or not selections:
        raise DiscoveryError("Selection evidence requires holdout metadata and source selections")
    if (
        holdout.get("unit") != "trials"
        or holdout.get("same_tasks") is not True
        or holdout.get("new_task_holdout") is not False
    ):
        raise DiscoveryError("Only explicitly declared same-task trial holdout is supported")
    source_base = holdout.get("selection_base_seed")
    declared_base = holdout.get("evaluation_base_seed")
    if not all(valid_seed(seed) for seed in (source_base, declared_base, evaluation_base_seed)):
        raise DiscoveryError("Missing or invalid holdout base seed")
    if source_base == evaluation_base_seed:
        raise DiscoveryError("Evaluation base seed overlaps source selection")
    if declared_base != evaluation_base_seed:
        raise DiscoveryError("Evaluation base seed differs from declared holdout")
    task_ids = holdout.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task, str) or not task for task in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise DiscoveryError("Holdout task IDs must be nonempty and unique")
    source_seeds = set(seed_list(holdout.get("selection_episode_seeds"), "source selection"))
    declared_seeds = set(seed_list(holdout.get("evaluation_episode_seeds"), "declared evaluation"))
    selected_trials: dict[str, list[int]] = {}
    for selection in selections:
        if (
            not isinstance(selection, Mapping)
            or selection.get("split") != "selection"
            or selection.get("selected") is not True
        ):
            raise DiscoveryError("Invalid source selection record")
        task_id = selection.get("task_id")
        if not isinstance(task_id, str) or task_id not in task_ids:
            raise DiscoveryError("Source selection task differs from holdout tasks")
        seeds = seed_list(selection.get("trial_seeds"), "source artifact trial")
        if type(selection.get("n")) is not int or selection["n"] != len(seeds):
            raise DiscoveryError("Source selection count differs from its trial seeds")
        if selection.get("selection_seed", source_base) != source_base:
            raise DiscoveryError("Source selection base seed contradicts holdout metadata")
        if task_id in selected_trials and selected_trials[task_id] != seeds:
            raise DiscoveryError("Selected artifacts have conflicting source trials")
        selected_trials[task_id] = seeds
    if (
        set(selected_trials) != set(task_ids)
        or {seed for seeds in selected_trials.values() for seed in seeds} != source_seeds
    ):
        raise DiscoveryError("Source trial evidence does not match declared selection seeds/tasks")

    planned = trial_map(task_seeds)
    if {task_id for task_id, repeat in planned} != set(task_ids):
        raise DiscoveryError("Evaluation task set differs from same-task holdout")
    actual_seeds = set(planned.values())
    if not actual_seeds.isdisjoint(source_seeds):
        raise DiscoveryError("Evaluation episode seeds overlap source selection")
    if actual_seeds != declared_seeds:
        raise DiscoveryError("Actual planned seeds differ from declared evaluation seeds")
    for task_id in task_ids:
        repeats = {repeat for task, repeat in planned if task == task_id}
        if repeats != set(range(len(repeats))):
            raise DiscoveryError("Evaluation repeats must be contiguous from zero")
        if len({planned[task_id, repeat] for repeat in repeats}) != len(repeats):
            raise DiscoveryError("Evaluation repeats reuse the same episode seed")
    declared_trials = holdout.get("evaluation_trials")
    if declared_trials is not None:
        if not isinstance(declared_trials, list) or any(
            not isinstance(trial, Mapping) for trial in declared_trials
        ):
            raise DiscoveryError("Declared evaluation_trials must be a list of trial records")
        expected = trial_map(
            (trial.get("task_id"), trial.get("repeat"), trial.get("seed"))
            for trial in declared_trials
        )
        if expected != planned:
            raise DiscoveryError(
                "Actual planned task/repeat/seed mapping differs from declared holdout"
            )


def capture_git_commit() -> str | None:
    """Capture code provenance; an unborn or non-Git checkout is explicitly null."""
    try:
        result = subprocess.run(
            ["git", "--no-pager", "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return (
        commit
        if result.returncode == 0
        and len(commit) in (40, 64)
        and all(char in "0123456789abcdef" for char in commit)
        else None
    )


def freeze_selection(
    score: CandidateScore,
    path: Path,
    *,
    selection_id: str,
    git_commit: str | None,
) -> Attack:
    """Derive a new immutable artifact; never modify the candidate being ranked."""
    if not selection_id:
        raise DiscoveryError("A selected artifact requires a unique selection_id")
    if not score.attack.artifact_sha256 or score.attack.artifact_sha256 != attack_sha256(
        score.attack
    ):
        raise DiscoveryError("Candidate SHA-256 changed before freezing selection")
    attack = score.attack.model_copy(
        update={
            "selection_id": selection_id,
            "git_commit": git_commit,
            "artifact_sha256": "",
        }
    )
    return save_attack(attack, path)


def _portable_config(config: MatrixConfig, base: Path) -> dict[str, Any]:
    from transferbench.runner.config import resolve_path

    data = config.model_dump(mode="json")
    for field in ("model_registry", "output_dir", "selection_manifest", "vulnerable_from"):
        if data[field] is not None:
            data[field] = str(resolve_path(base, data[field]))
    data["attack_artifacts"] = [str(resolve_path(base, path)) for path in data["attack_artifacts"]]
    if data["tasks"]["paths"] is not None:
        data["tasks"]["paths"] = [str(resolve_path(base, path)) for path in data["tasks"]["paths"]]
    return data


def make_selection_config(
    config: MatrixConfig,
    base: Path,
    *,
    source_model: str,
    source_scaffold: str,
    artifact_paths: Sequence[Path],
    task_snapshot: Path | None = None,
) -> MatrixConfig:
    from transferbench.runner.config import MatrixConfig

    data = _portable_config(config, base)
    data.update(
        {
            "experiment": {
                "name": config.experiment.name + "-selection",
                "stage": "selection",
                "transfer_mode": "fixed",
            },
            "models": [source_model],
            "scaffolds": [source_scaffold],
            "attacks": [],
            "defenses": ["none"],
            "monitor_model": None,
            "include_controls": False,
            "attack_artifacts": [str(path.resolve()) for path in artifact_paths],
            "selection_manifest": None,
            "vulnerable_from": None,
        }
    )
    if task_snapshot is not None:
        data["tasks"].update({"paths": [str(task_snapshot.resolve())], "limit": None})
    return MatrixConfig.model_validate(data)


def make_transfer_config(
    config: MatrixConfig,
    base: Path,
    *,
    source_model: str,
    source_scaffold: str,
    artifact_paths: Sequence[Path],
    selection_manifest: Path,
    evaluation_seed: int,
    task_snapshot: Path | None = None,
) -> MatrixConfig:
    from transferbench.runner.config import MatrixConfig

    if evaluation_seed == config.seed:
        raise DiscoveryError("Evaluation requires a different base seed from selection")
    data = _portable_config(config, base)
    data.update(
        {
            "experiment": {
                "name": config.experiment.name + "-transfer",
                "stage": "evaluation",
                "transfer_mode": "source_optimized",
            },
            "models": list(dict.fromkeys([source_model, *config.models])),
            "scaffolds": list(dict.fromkeys([source_scaffold, *config.scaffolds])),
            "defenses": list(dict.fromkeys(["none", *config.defenses])),
            "attacks": [],
            "seed": evaluation_seed,
            "attack_artifacts": [str(path.resolve()) for path in artifact_paths],
            "selection_manifest": str(selection_manifest.resolve()),
            "vulnerable_from": None,
        }
    )
    if task_snapshot is not None:
        data["tasks"].update({"paths": [str(task_snapshot.resolve())], "limit": None})
    return MatrixConfig.model_validate(data)


def _write_exclusive(path: Path, text: str) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_run(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_episodes: int,
    max_cost_usd: float,
) -> float:
    """Refuse partial/error runs and explicit exhausted/uncertain cost accounting.

    Episode completeness is checked independently of producer summary counts.
    Ranking subsequently validates each cell and its exact paired repeat support.
    """
    if len(rows) != expected_episodes:
        raise DiscoveryError("Source run is incomplete; refusing unequal candidate support")
    if manifest.get("status", "completed") not in {"completed", "complete", "ok"} or manifest.get(
        "error"
    ):
        raise DiscoveryError("Source run did not complete successfully")
    for field in ("n_episodes", "n_completed"):
        if field in manifest and (
            type(manifest[field]) is not int or manifest[field] != expected_episodes
        ):
            raise DiscoveryError(
                f"Source manifest {field} does not match the complete episode count"
            )
    run_id = manifest.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or any(row.get("run_id") != run_id for row in rows)
    ):
        raise DiscoveryError("Source run manifest/episode identities disagree")
    flags = {
        "budget_exhausted",
        "budget_interrupted",
        "budget_uncertain",
        "cost_uncertain",
        "uncertain_cost",
        "exhausted",
        "uncertain",
    }
    accounting = [manifest]
    accounting.extend(
        manifest[key]
        for key in ("budget", "cost_budget", "accounting")
        if isinstance(manifest.get(key), Mapping)
    )
    for record in accounting:
        if any(
            key in record and record[key] is not False and record[key] is not None for key in flags
        ):
            raise DiscoveryError("Source run has exhausted or uncertain cost accounting")
    costs = []
    for row in rows:
        if row.get("status") != "ok" or row.get("error"):
            raise DiscoveryError("Source run contains failed or budget-interrupted episodes")
        cost = row.get("estimated_cost_usd")
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise DiscoveryError("Source run contains invalid cost accounting")
        if "unknown_usage_reserved" in json.dumps(row.get("call_costs", [])):
            raise DiscoveryError("Source run contains uncertain provider usage")
        costs.append(cost)
    total = math.fsum(costs)
    if "estimated_cost_usd" in manifest:
        charged = manifest["estimated_cost_usd"]
        if (
            isinstance(charged, bool)
            or not isinstance(charged, (int, float))
            or not math.isfinite(charged)
            or charged < 0
            or not math.isclose(charged, total, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise DiscoveryError("Source manifest cost does not match episode accounting")
    if total > max_cost_usd + 1e-12:
        raise DiscoveryError("Source selection exceeded its configured cost ceiling")
    return total


def discover(
    config_path: Path,
    *,
    source_model: str,
    source_scaffold: str,
    family: str = "indirect_instruction",
    candidates: int = 3,
    top_k: int = 1,
    output_root: Path | None = None,
) -> Path:
    """Run source-only Inspect selection and return a new discovery directory.

    Outputs are candidate artifacts, a task snapshot, the canonical source run,
    selected artifacts, ``selection.json``, and a runnable ``transfer.yaml``.
    Target evaluation is not run automatically. Paths in generated configs are
    absolute so moving the config's working directory cannot change its inputs.
    """
    import polars as pl
    import yaml

    from transferbench.attacks.base import canonical_family
    from transferbench.attacks.generator import MAX_TEMPLATE_CANDIDATES, generate_unique_candidates
    from transferbench.models.registry import ModelRegistry
    from transferbench.runner.config import config_tasks, load_config, resolve_path

    if type(candidates) is not int or not 1 <= candidates <= MAX_TEMPLATE_CANDIDATES:
        raise DiscoveryError(
            f"Finite templates support 1..{MAX_TEMPLATE_CANDIDATES} distinct candidates, not {candidates!r}"
        )
    if type(top_k) is not int or not 1 <= top_k <= candidates:
        raise DiscoveryError("top_k must be between 1 and candidates")
    family = canonical_family(family)
    if family == "none":
        raise DiscoveryError("Cannot discover a no-attack family")
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    base = config_path.parent
    registry = ModelRegistry(resolve_path(base, config.model_registry))
    source_spec = registry.get(source_model)
    for alias in config.models:
        registry.get(alias)
    tasks = config_tasks(config, base)

    # The import is deliberately deferred: runner/CLI configuration may import
    # discovery before the matrix module itself has finished being initialized.
    from transferbench.runner import matrix

    seed_for: Callable[[int, str, int], int] | None = getattr(matrix, "episode_seed", None)
    if not callable(seed_for):
        raise DiscoveryError(
            "Matrix runner must expose episode_seed(base_seed, task_id, repeat) for verified trial holdout"
        )

    root = (
        Path(output_root).resolve()
        if output_root is not None
        else resolve_path(base, config.output_dir)
    )
    discovery_id = f"discovery-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
    directory = root / discovery_id
    directory.mkdir(parents=True, exist_ok=False)
    candidate_dir, selected_dir = directory / "candidates", directory / "selected"
    candidate_dir.mkdir()
    selected_dir.mkdir()
    commit = capture_git_commit()
    task_snapshot = directory / "tasks.jsonl"
    _write_exclusive(task_snapshot, "".join(task.model_dump_json() + "\n" for task in tasks))
    generated = [
        attack
        for task in tasks
        for attack in generate_unique_candidates(
            task,
            family=family,
            seed=config.seed,
            count=candidates,
            source_model=source_model,
            source_scaffold=source_scaffold,
        )
    ]
    candidate_paths: dict[str, Path] = {}
    for attack in generated:
        path = candidate_dir / f"{attack.artifact_sha256}.json"
        save_attack(attack, path)
        candidate_paths[attack.artifact_sha256] = path
    selection_config = make_selection_config(
        config,
        base,
        source_model=source_model,
        source_scaffold=source_scaffold,
        artifact_paths=list(candidate_paths.values()),
        task_snapshot=task_snapshot,
    )
    selection_config_path = directory / "selection-config.yaml"
    _write_exclusive(
        selection_config_path,
        yaml.safe_dump(selection_config.model_dump(mode="json"), sort_keys=False),
    )
    matrix.plan_matrix(selection_config_path, config_override=selection_config)
    source_run = Path(
        matrix.run_matrix(
            selection_config_path,
            output_root=directory / "source-runs",
            config_override=selection_config,
            run_id=f"{discovery_id}-selection",
        )
    ).resolve()
    manifest_path, episodes_path = source_run / "manifest.json", source_run / "episodes.parquet"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source_manifest, dict):
        raise DiscoveryError("Source manifest must be an object")
    rows = pl.read_parquet(episodes_path).to_dicts()
    source_cost = validate_source_run(
        source_manifest,
        rows,
        expected_episodes=len(generated) * config.repeats,
        max_cost_usd=config.limits.max_cost_usd,
    )
    ranked = rank_candidates(
        rows,
        generated,
        source_model=source_model,
        source_scaffold=source_scaffold,
        repeats=config.repeats,
        source_model_id=source_spec.inspect_id,
    )
    if any(row["seed"] != seed_for(config.seed, row["task_id"], row["repeat"]) for row in rows):
        raise DiscoveryError("Runner seed helper does not match actual source episode seeds")
    source_seeds = {row["seed"] for row in rows}

    def planned_seeds(base_seed: int) -> set[int]:
        return {
            seed_for(base_seed, task.task_id, repeat)
            for task in tasks
            for repeat in range(config.repeats)
        }

    evaluation_seed = choose_holdout_seed(
        config.seed,
        repeats=config.repeats,
        selection_seeds=source_seeds,
        planned_seeds=planned_seeds,
    )
    selected_paths: list[Path] = []
    selections: list[dict[str, Any]] = []
    candidate_scores: list[dict[str, Any]] = []
    for scores in ranked.values():
        for rank, score in enumerate(scores, start=1):
            candidate_scores.append({**score.as_dict(), "rank": rank, "selected": rank <= top_k})
            if rank > top_k:
                continue
            selection_id = "selection-" + uuid.uuid4().hex
            path = selected_dir / f"{selection_id}.json"
            frozen = freeze_selection(score, path, selection_id=selection_id, git_commit=commit)
            selected_paths.append(path)
            selections.append(
                {
                    **score.as_dict(),
                    "selection_id": selection_id,
                    "attack_artifact_sha256": frozen.artifact_sha256,
                    "artifact_path": str(path),
                    "candidate_artifact_path": str(candidate_paths[score.attack.artifact_sha256]),
                    "source_model": source_model,
                    "source_model_id": source_spec.inspect_id,
                    "source_scaffold": source_scaffold,
                    "source_simulated": source_spec.simulated,
                    "source_run_path": str(source_run),
                    "source_run_id": source_manifest["run_id"],
                    "source_episode_ids": [
                        row["episode_id"]
                        for row in rows
                        if row["attack_artifact_sha256"] == score.attack.artifact_sha256
                    ],
                    "split": "selection",
                    "selection_seed": config.seed,
                    "selected": True,
                    "rank": rank,
                    "n_valid": score.n,
                    "n_errors": 0,
                    "git_commit": commit,
                }
            )
    selection_manifest = directory / "selection.json"
    transfer_config = make_transfer_config(
        config,
        base,
        source_model=source_model,
        source_scaffold=source_scaffold,
        artifact_paths=selected_paths,
        selection_manifest=selection_manifest,
        evaluation_seed=evaluation_seed,
        task_snapshot=task_snapshot,
    )
    evidence = {
        "schema_version": "1.0",
        "discovery_id": discovery_id,
        "created_at": datetime.now(UTC).isoformat(),
        "generator": "finite_synthetic_templates",
        "unique_template_pool": MAX_TEMPLATE_CANDIDATES,
        "candidates_per_task": candidates,
        "top_k_per_task": top_k,
        "source_model": source_model,
        "source_model_id": source_spec.inspect_id,
        "source_scaffold": source_scaffold,
        "family": family,
        "source_simulated": source_spec.simulated,
        "git_commit": commit,
        "git_commit_available": commit is not None,
        "source_run_path": str(source_run),
        "source_run_id": source_manifest["run_id"],
        "source_manifest_sha256": _file_sha256(manifest_path),
        "source_episodes_sha256": _file_sha256(episodes_path),
        "source_estimated_cost_usd": source_cost,
        "task_snapshot_path": str(task_snapshot),
        "task_snapshot_sha256": _file_sha256(task_snapshot),
        "source_selections": selections,
        "candidate_scores": candidate_scores,
        "holdout": {
            "unit": "trials",
            "same_tasks": True,
            "new_task_holdout": False,
            "task_ids": [task.task_id for task in tasks],
            "selection_base_seed": config.seed,
            "evaluation_base_seed": evaluation_seed,
            "selection_episode_seeds": sorted(source_seeds),
            "evaluation_episode_seeds": sorted(planned_seeds(evaluation_seed)),
            "evaluation_trials": [
                {
                    "task_id": task.task_id,
                    "repeat": repeat,
                    "seed": seed_for(evaluation_seed, task.task_id, repeat),
                }
                for task in tasks
                for repeat in range(config.repeats)
            ],
            "seed_overlap": False,
            "limitation": "Disjoint scheduled seeds do not guarantee provider determinism or independent model samples; no new-task holdout is claimed.",
        },
        "selection_policy": "Complete paired C1 source trials; descending ASR, SHA-256 tie-break; reject any invalid comparison or cost interruption.",
    }
    _write_exclusive(
        selection_manifest, json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    transfer_path = directory / "transfer.yaml"
    transfer_plan = matrix.plan_matrix(config_path, config_override=transfer_config)
    verify_evaluation_holdout(
        evidence,
        (
            (episode.task.task_id, episode.repeat, episode.seed)
            for episode in transfer_plan.episodes
        ),
        evaluation_base_seed=transfer_config.seed,
    )
    _write_exclusive(
        transfer_path, yaml.safe_dump(transfer_config.model_dump(mode="json"), sort_keys=False)
    )
    return directory
