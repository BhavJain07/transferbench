"""Controlled matrix planning and bounded Inspect evaluation batches.

Inspect .eval logs are canonical. The Parquet table is a lossless projection of
sample metadata, not a parallel tracing system. No providers are called by planning.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from inspect_ai import Task as InspectTask
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from transferbench.attacks import get_attack
from transferbench.attacks.artifacts import load_attack, save_attack, seal_attack
from transferbench.attacks.base import canonical_family
from transferbench.defenses import get_defense
from transferbench.models.registry import ModelRegistry, ModelSpec
from transferbench.models.runtime import CostBudget, GenerationRuntime
from transferbench.runner.config import MatrixConfig, config_tasks, load_config, resolve_path
from transferbench.runner.manifest import (
    atomic_json,
    dependency_state,
    git_state,
    now,
    publication_reasons,
    read_episodes,
    sha256_file,
    write_episodes,
)
from transferbench.scaffolds import get_scaffold
from transferbench.scorers import score_attack, score_policy, score_utility
from transferbench.tasks.schema import Attack, Episode, EpisodeResult, TaskSpec


def episode_seed(base_seed: int, task_id: str, repeat: int) -> int:
    data = json.dumps([base_seed, task_id, repeat], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(data.encode()).digest()[:4], "big") % (2**31)


@dataclass(frozen=True)
class EpisodePlan:
    model_alias: str
    scaffold_id: str
    task: TaskSpec
    attack: Attack | None
    defense_id: str
    seed: int
    repeat: int

    @property
    def condition(self) -> Literal["C0", "C1", "C2", "C3"]:
        if self.attack is None:
            return "C0" if self.defense_id == "none" else "C3"
        return "C1" if self.defense_id == "none" else "C2"

    @property
    def episode_id(self) -> str:
        identity = [
            self.model_alias,
            self.scaffold_id,
            self.task.task_id,
            self.attack.artifact_sha256 if self.attack else None,
            self.defense_id,
            self.seed,
            self.repeat,
        ]
        return "ep-" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]


@dataclass
class MatrixPlan:
    config: MatrixConfig
    config_path: Path
    models: dict[str, ModelSpec]
    tasks: list[TaskSpec]
    episodes: list[EpisodePlan]
    artifacts: list[Attack]
    selections: list[dict]
    defense_selection: dict | None = None

    @property
    def projection(self) -> dict:
        bounds = {
            alias: model.pricing.estimate(model.max_input_tokens, self.config.limits.max_tokens)
            for alias, model in self.models.items()
        }
        monitor_bound = (
            bounds[self.config.monitor_model] if self.config.monitor_model is not None else 0
        )
        maximum = sum(
            max(bounds[episode.model_alias], monitor_bound) * self.config.limits.max_turns
            for episode in self.episodes
        )
        return {
            "estimated_episodes": len(self.episodes),
            "conservative_maximum_usd": maximum,
            "hard_estimated_cost_ceiling_usd": self.config.limits.max_cost_usd,
            "may_stop_early": maximum > self.config.limits.max_cost_usd,
            "simulated": all(spec.simulated for spec in self.models.values()),
            "conditions": {
                condition: sum(ep.condition == condition for ep in self.episodes)
                for condition in ("C0", "C1", "C2", "C3")
            },
            "note": "Worst-case bound, not a prediction. Includes worker/monitor calls within the shared turn cap.",
        }


def plan_matrix(config_path: str | Path, config_override: MatrixConfig | None = None) -> MatrixPlan:
    path = Path(config_path).resolve()
    config = config_override or load_config(path)
    base = path.parent
    registry = ModelRegistry(resolve_path(base, config.model_registry))
    aliases = list(config.models)
    if config.monitor_model is not None and config.monitor_model not in aliases:
        aliases.append(config.monitor_model)
    models = {alias: registry.get(alias) for alias in aliases}
    tasks = config_tasks(config, base)
    artifacts = [load_attack(resolve_path(base, artifact)) for artifact in config.attack_artifacts]
    if len({a.artifact_sha256 for a in artifacts}) != len(artifacts):
        raise ValueError("Duplicate frozen attack artifacts")
    selections = []
    selection_evidence = None
    if config.selection_manifest:
        selection_evidence = json.loads(resolve_path(base, config.selection_manifest).read_text())
        if not isinstance(selection_evidence, dict):
            raise ValueError("selection_manifest must be a JSON object")
        selections = selection_evidence.get("source_selections", [])
        if not isinstance(selections, list):
            raise ValueError("selection_manifest requires source_selections list")
    if config.experiment.transfer_mode == "source_optimized":
        from transferbench.analysis.provenance import Provenance

        provenance = Provenance(
            {
                "attack_artifacts": [a.model_dump() for a in artifacts],
                "source_selections": selections,
            }
        )
        for artifact in artifacts:
            problems = provenance.check(
                {
                    "attack_id": artifact.id,
                    "attack_artifact_sha256": artifact.artifact_sha256,
                    "attack_source_model": artifact.source_model,
                    "attack_source_scaffold": artifact.source_scaffold,
                    "selection_id": artifact.selection_id,
                    "task_id": artifact.task_id,
                    "transfer_mode": "source_optimized",
                }
            )
            if problems:
                raise ValueError(f"Invalid selected artifact {artifact.id}: {'; '.join(problems)}")
        for selection in selections:
            if selection.get("selection_seed") == config.seed:
                raise ValueError("Evaluation seed must differ from the source selection seed")
    vulnerable = None
    if config.vulnerable_from:
        source = resolve_path(base, config.vulnerable_from)
        rows = read_episodes(source / "episodes.parquet" if source.is_dir() else source)
        vulnerable = {
            (r.model_alias, r.scaffold_id, r.task_id, r.attack_id)
            for r in rows
            if r.status == "ok" and r.attack_success and r.defense_id == "none"
        }
        if not vulnerable:
            raise ValueError("No vulnerable undefended pairs found in vulnerable_from")
    episodes = []
    commit = git_state(path.parent)["commit"]
    all_artifacts = {attack.artifact_sha256: attack for attack in artifacts}
    for alias in config.models:
        for scaffold in config.scaffolds:
            for task in tasks:
                if artifacts:
                    attacks = [a for a in artifacts if a.task_id in (None, task.task_id)]
                else:
                    variant_seed = config.seed
                    if config.experiment.transfer_mode == "family":
                        variant_seed = episode_seed(config.seed, alias + "/" + scaffold, 0)
                    attacks = []
                    for name in config.attacks:
                        if canonical_family(name) != "none":
                            artifact = get_attack(name, task, seed=variant_seed)
                            artifact = seal_attack(
                                artifact.model_copy(
                                    update={"git_commit": commit, "artifact_sha256": ""}
                                )
                            )
                            attacks.append(artifact)
                            all_artifacts[artifact.artifact_sha256] = artifact
                for repeat in range(config.repeats):
                    seed = episode_seed(config.seed, task.task_id, repeat)
                    if config.include_controls or any(
                        canonical_family(name) == "none" for name in config.attacks
                    ):
                        controls = (
                            list(dict.fromkeys(["none", *config.defenses]))
                            if config.include_controls
                            else config.defenses
                        )
                        episodes.extend(
                            EpisodePlan(alias, scaffold, task, None, defense, seed, repeat)
                            for defense in controls
                        )
                    defenses = (
                        list(dict.fromkeys(["none", *config.defenses]))
                        if config.include_controls
                        else config.defenses
                    )
                    for attack in attacks:
                        if (
                            vulnerable is not None
                            and (alias, scaffold, task.task_id, attack.id) not in vulnerable
                        ):
                            continue
                        episodes.extend(
                            EpisodePlan(alias, scaffold, task, attack, defense, seed, repeat)
                            for defense in defenses
                        )
    if not episodes:
        raise ValueError("Experiment contains no episodes")
    if len(episodes) > config.limits.max_episodes:
        raise ValueError(
            f"Planned {len(episodes)} episodes exceeds max_episodes={config.limits.max_episodes}"
        )
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("Experimental plan contains duplicate episode identities")
    if config.experiment.transfer_mode == "source_optimized":
        from transferbench.runner.discovery import verify_evaluation_holdout

        if not isinstance(selection_evidence, dict):
            raise ValueError("source_optimized requires selection evidence")
        verify_evaluation_holdout(
            selection_evidence,
            [(ep.task.task_id, ep.repeat, ep.seed) for ep in episodes],
            evaluation_base_seed=config.seed,
        )
    defense_selection = None
    if config.defense_selection:
        from transferbench.runner.defense_selection import (
            load_defense_selection,
            validate_defense_evaluation,
        )

        defense_selection = load_defense_selection(resolve_path(base, config.defense_selection))
        validate_defense_evaluation(
            defense_selection,
            [(ep.task.task_id, ep.repeat, ep.seed) for ep in episodes],
            config.defenses,
        )
        source_alias = defense_selection["source_model"]
        if (
            source_alias not in config.models
            or models[source_alias].inspect_id != defense_selection["source_model_id"]
        ):
            raise ValueError(
                "Defense transfer requires the same source alias/version in evaluation models"
            )
        if defense_selection["source_scaffold"] not in config.scaffolds:
            raise ValueError("Defense transfer requires the source scaffold in evaluation")
    used = {ep.attack.artifact_sha256 for ep in episodes if ep.attack}
    return MatrixPlan(
        config,
        path,
        models,
        tasks,
        episodes,
        [a for digest, a in all_artifacts.items() if digest in used],
        selections,
        defense_selection,
    )


@scorer(
    metrics={"utility_success": [mean()], "attack_success": [mean()], "policy_violation": [mean()]}
)
def deterministic_scores():
    async def score(state: TaskState, target: Target) -> Score:
        episode = state.metadata["episode"]
        return Score(
            value={
                key: int(episode[key])
                for key in ("utility_success", "attack_success", "policy_violation")
            },
            explanation=f"Deterministic synthetic assertions; condition={episode['condition']}; status={episode['status']}. Filter invalid episodes in TransferBench analysis.",
        )

    return score


def _episode(
    run_id: str,
    item: EpisodePlan,
    spec: ModelSpec,
    result: EpisodeResult,
    config: MatrixConfig,
    budget: CostBudget,
    defense_selection: dict | None = None,
) -> Episode:
    attack = item.attack
    status = "ok"
    if result.error:
        status = "budget_exceeded" if result.error.startswith("BudgetExceeded:") else "error"
    costs = result.costs
    origin = (
        defense_selection
        if defense_selection and item.defense_id == defense_selection["defense_id"]
        else None
    )
    return Episode(
        run_id=run_id,
        episode_id=item.episode_id,
        model_id=spec.inspect_id,
        model_family=spec.family,
        model_alias=item.model_alias,
        scaffold_id=item.scaffold_id,
        task_id=item.task.task_id,
        task_family=item.task.family,
        attack_id=attack.id if attack else None,
        attack_family=attack.family if attack else None,
        attack_source_model=attack.source_model if attack else None,
        attack_source_scaffold=attack.source_scaffold if attack else None,
        attack_artifact_sha256=attack.artifact_sha256 if attack else None,
        selection_id=attack.selection_id if attack else None,
        defense_id=item.defense_id,
        defense_source_model=origin["source_model"] if origin else None,
        defense_source_scaffold=origin["source_scaffold"] if origin else None,
        condition=item.condition,
        transfer_mode=config.experiment.transfer_mode,
        split=config.experiment.stage,
        seed=item.seed,
        repeat=item.repeat,
        utility_success=score_utility(item.task, result),
        attack_success=score_attack(item.task, result, attack),
        policy_violation=score_policy(item.task, result),
        tool_calls=[call.model_dump(mode="json") for call in result.tool_calls],
        transcript=result.transcript,
        input_tokens=sum(cost.input_tokens for cost in costs),
        cached_tokens=sum(cost.cached_tokens for cost in costs),
        output_tokens=sum(cost.output_tokens for cost in costs),
        latency_ms=sum(cost.latency_ms for cost in costs),
        estimated_cost_usd=sum(cost.estimated_cost_usd for cost in costs),
        call_costs=[cost.model_dump(mode="json") for cost in costs],
        propagation_depth=max(
            (int(event["depth"]) for event in result.propagation_events if "depth" in event),
            default=0,
        ),
        propagation_events=result.propagation_events,
        defense_events=result.defense_events,
        status=status,
        error=result.error,
        simulated=spec.simulated or any(cost.provider == "fake" for cost in costs),
    )


def run_matrix(
    config_path: str | Path,
    *,
    output_root: Path | None = None,
    config_override: MatrixConfig | None = None,
    run_id: str | None = None,
) -> Path:
    plan = plan_matrix(config_path, config_override)
    config = plan.config
    root = output_root or resolve_path(plan.config_path.parent, config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = (
        run_id
        or f"{now()[:19].replace(':', '').replace('T', '_')}_{config.experiment.name}_{uuid.uuid4().hex[:8]}"
    )
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    run_dir = root / run_id
    run_dir.mkdir(exist_ok=False)
    inputs = run_dir / "inputs"
    inputs.mkdir()
    (run_dir / "inspect").mkdir()
    artifact_dir = inputs / "attacks"
    artifact_dir.mkdir()
    atomic_json(inputs / "config.json", config.model_dump(mode="json"))
    atomic_json(inputs / "tasks.json", [task.model_dump(mode="json") for task in plan.tasks])
    atomic_json(
        inputs / "models.json",
        {alias: model.model_dump(mode="json") for alias, model in plan.models.items()},
    )
    atomic_json(inputs / "source_selections.json", plan.selections)
    if plan.defense_selection:
        atomic_json(inputs / "defense_selection.json", plan.defense_selection)
    for attack in plan.artifacts:
        save_attack(attack, artifact_dir / f"{attack.artifact_sha256}.json")
    repo = Path(__file__).resolve().parents[2]
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "config": str(plan.config_path),
        "started_at": now(),
        "finished_at": None,
        "status": "running",
        "publishable": False,
        "models": {alias: model.model_dump(mode="json") for alias, model in plan.models.items()},
        "resolved_config": config.model_dump(mode="json"),
        "n_episodes": len(plan.episodes),
        "n_completed": 0,
        "projection": plan.projection,
        "attack_artifacts": [attack.model_dump(mode="json") for attack in plan.artifacts],
        "source_selections": plan.selections,
        "defense_selection": plan.defense_selection,
        **git_state(repo),
        **dependency_state(repo),
    }
    atomic_json(run_dir / "manifest.json", manifest)
    budget = CostBudget(config.limits.max_cost_usd)
    episodes: list[Episode] = []
    log_files: list[str] = []
    try:
        models = {alias: spec.resolve() for alias, spec in plan.models.items()}
        for alias in config.models:
            items = [episode for episode in plan.episodes if episode.model_alias == alias]
            for offset in range(0, len(items), 64):
                if budget.exhausted:
                    break
                batch = items[offset : offset + 64]
                index = {item.episode_id: item for item in batch}

                @solver
                def matrix_solver(index=index, alias=alias) -> Solver:
                    async def solve(state: TaskState, generate: Generate) -> TaskState:
                        item = index[str(state.sample_id)]
                        runtime = GenerationRuntime(
                            models[alias],
                            plan.models[alias],
                            budget,
                            seed=item.seed,
                            temperature=config.generation.temperature,
                            max_tokens=config.limits.max_tokens,
                            timeout_seconds=config.limits.timeout_seconds,
                            monitor_model=models[config.monitor_model]
                            if config.monitor_model is not None
                            else None,
                            monitor_spec=plan.models[config.monitor_model]
                            if config.monitor_model is not None
                            else None,
                        )
                        if budget.exhausted:
                            result = EpisodeResult(
                                error="BudgetExceeded: run stopped before this episode"
                            )
                            state.output = ModelOutput()
                        else:
                            state = await get_scaffold(item.scaffold_id).as_solver(
                                item.task,
                                models[alias],
                                get_defense(item.defense_id),
                                item.attack,
                                runtime=runtime,
                                max_turns=config.limits.max_turns,
                            )(state, generate)
                            result = EpisodeResult.model_validate(state.metadata["episode_result"])
                        episode = _episode(
                            run_id,
                            item,
                            plan.models[alias],
                            result,
                            config,
                            budget,
                            plan.defense_selection,
                        )
                        state.metadata["episode"] = episode.model_dump(mode="json")
                        state.metadata["episode_result"] = result.model_dump(mode="json")
                        state.completed = True
                        return state

                    return solve

                task = InspectTask(
                    name=f"transferbench-{alias}-{offset // 64}",
                    dataset=[
                        Sample(
                            id=item.episode_id,
                            input=item.task.user_goal,
                            metadata={
                                "plan": {
                                    "task_id": item.task.task_id,
                                    "scaffold": item.scaffold_id,
                                    "condition": item.condition,
                                    "seed": item.seed,
                                }
                            },
                        )
                        for item in batch
                    ],
                    solver=matrix_solver(),
                    scorer=deterministic_scores(),
                    model=models[alias],
                    metadata={
                        "run_id": run_id,
                        "model_alias": alias,
                        "synthetic_environment": True,
                    },
                )
                logs = inspect_eval(
                    task,
                    model=models[alias],
                    log_dir=str(run_dir / "inspect"),
                    log_format="eval",
                    display="none",
                    max_samples=1,
                    max_tasks=1,
                    fail_on_error=False,
                    retry_on_error=0,
                    log_model_api=True,
                )
                for log in logs:
                    location = str(log.location)
                    log_files.append(location)
                    for sample in log.samples or []:
                        if sample.metadata and "episode" in sample.metadata:
                            episode = Episode.model_validate(sample.metadata["episode"])
                            episode.inspect_log = str(Path(location).relative_to(run_dir))
                            episodes.append(episode)
                        else:
                            item = index[str(sample.id)]
                            result = EpisodeResult(
                                error=f"InspectError: {sample.error or log.error}"
                            )
                            episodes.append(
                                _episode(
                                    run_id,
                                    item,
                                    plan.models[alias],
                                    result,
                                    config,
                                    budget,
                                    plan.defense_selection,
                                )
                            )
                write_episodes(run_dir / "episodes.parquet", episodes)
                manifest.update(n_completed=len(episodes), estimated_cost_usd=budget.charged)
                atomic_json(run_dir / "manifest.json", manifest)
        if budget.exhausted:
            manifest["status"] = "budget_exceeded"
        elif len(episodes) != len(plan.episodes) or any(
            episode.status != "ok" for episode in episodes
        ):
            manifest["status"] = "completed_with_errors"
        else:
            manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        write_episodes(run_dir / "episodes.parquet", episodes)
        manifest.update(
            finished_at=now(),
            n_completed=len(episodes),
            inspect_logs=log_files,
            estimated_cost_usd=budget.charged,
            budget_uncertain=budget.uncertain,
        )
        manifest["publication_blockers"] = publication_reasons(manifest)
        manifest["publishable"] = not manifest["publication_blockers"]
        manifest["file_hashes"] = {
            str(path.relative_to(run_dir)): sha256_file(path)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        atomic_json(run_dir / "manifest.json", manifest)
    if episodes:
        from transferbench.analysis import analyze_run

        analyze_run(run_dir)
    return run_dir
