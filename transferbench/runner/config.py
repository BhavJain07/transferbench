"""Strict configuration with explicit experimental controls and bounded resources."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from transferbench.attacks.base import canonical_family
from transferbench.defenses import get_defense
from transferbench.models.registry import read_yaml
from transferbench.scaffolds import get_scaffold
from transferbench.tasks.dataset import load_suite
from transferbench.tasks.schema import Record, TaskSpec


class Experiment(Record):
    name: str = Field(default="experiment", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    stage: Literal["screening", "selection", "evaluation", "confirmatory"] = "evaluation"
    transfer_mode: Literal["fixed", "family", "source_optimized"] = "fixed"


class TaskConfig(Record):
    suites: list[str] = Field(default_factory=lambda: ["workspace", "delegation"], min_length=1)
    paths: list[str] | None = None
    limit: int | None = Field(default=None, ge=1)


class Generation(Record):
    temperature: float = Field(default=0.2, ge=0, le=2, allow_inf_nan=False)


class Limits(Record):
    max_turns: int = Field(default=20, ge=1, le=200)
    max_tokens: int = Field(default=2048, ge=32, le=128000)
    max_cost_usd: float = Field(default=0, ge=0, allow_inf_nan=False)
    timeout_seconds: float = Field(default=120, gt=0, le=3600, allow_inf_nan=False)
    max_episodes: int = Field(default=20000, ge=1)


class MatrixConfig(Record):
    experiment: Experiment = Field(default_factory=Experiment)
    model_registry: str = "models.yaml"
    models: list[str] = Field(min_length=1)
    scaffolds: list[str] = Field(min_length=1)
    tasks: TaskConfig = Field(default_factory=TaskConfig)
    attacks: list[str] = Field(default_factory=lambda: ["indirect_instruction"])
    defenses: list[str] = Field(default_factory=lambda: ["none"])
    repeats: int = Field(default=1, ge=1, le=1000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    generation: Generation = Field(default_factory=Generation)
    limits: Limits = Field(default_factory=Limits)
    include_controls: bool = True
    monitor_model: str | None = None
    attack_artifacts: list[str] = Field(default_factory=list)
    selection_manifest: str | None = None
    defense_selection: str | None = None
    vulnerable_from: str | None = None
    output_dir: str = "../results"

    @model_validator(mode="after")
    def validate_dimensions(self) -> "MatrixConfig":
        self.defenses = [
            "+".join(part.strip() for part in defense.split("+")) for defense in self.defenses
        ]
        for field in ("models", "scaffolds", "attacks", "defenses", "attack_artifacts"):
            values = getattr(self, field)
            if len(values) != len(set(values)):
                raise ValueError(f"{field} cannot contain duplicates")
        for scaffold in self.scaffolds:
            get_scaffold(scaffold)
        canonical = [canonical_family(attack) for attack in self.attacks]
        if len(canonical) != len(set(canonical)):
            raise ValueError("attack aliases cannot duplicate the same family")
        for defense in self.defenses:
            get_defense(defense)
        if (
            any(
                set(defense.split("+")) & {"monitor", "second_model_monitor"}
                for defense in self.defenses
            )
            and not self.monitor_model
        ):
            raise ValueError("monitor defense requires monitor_model")
        if self.experiment.transfer_mode == "source_optimized" and not self.attack_artifacts:
            raise ValueError("source_optimized requires frozen attack_artifacts")
        if self.experiment.transfer_mode == "source_optimized" and not self.selection_manifest:
            raise ValueError("source_optimized requires selection_manifest")
        if not self.defenses:
            raise ValueError("at least one defense is required")
        return self


def load_config(path: str | Path) -> MatrixConfig:
    return MatrixConfig.model_validate(read_yaml(Path(path)))


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def config_tasks(config: MatrixConfig, base: Path) -> list[TaskSpec]:
    paths = [resolve_path(base, p) for p in config.tasks.paths] if config.tasks.paths else None
    tasks = load_suite(config.tasks.suites, limit=config.tasks.limit, paths=paths)
    if not tasks:
        raise ValueError("Task selection is empty")
    return tasks
