"""Structural contract shared by all ready-to-run scaffold adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from inspect_ai.model import Model
from inspect_ai.solver import Solver

from transferbench.tasks.schema import Attack, EpisodeResult, TaskSpec

if TYPE_CHECKING:
    from transferbench.defenses.base import Defense
    from transferbench.models.runtime import GenerationRuntime


@runtime_checkable
class Scaffold(Protocol):
    @property
    def name(self) -> str: ...

    async def run(
        self,
        task: TaskSpec,
        model: Model,
        defense: Defense,
        attack: Attack | None,
        *,
        runtime: GenerationRuntime,
        max_turns: int = 20,
    ) -> EpisodeResult: ...

    def as_solver(
        self,
        task: TaskSpec,
        model: Model,
        defense: Defense,
        attack: Attack | None,
        *,
        runtime: GenerationRuntime,
        max_turns: int = 20,
    ) -> Solver: ...


__all__ = ["Scaffold"]
