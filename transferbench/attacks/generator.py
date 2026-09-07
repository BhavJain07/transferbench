"""Finite, unique synthetic templates with an async generation interface.

No model is called and no optimization is claimed here. Paid candidate generation
would require a separately implemented generator using the accounted runtime.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from transferbench.attacks.base import canonical_family
from transferbench.attacks.library import DeterministicAttackGenerator as SyncTemplateGenerator
from transferbench.tasks.schema import Attack, TaskSpec

MAX_TEMPLATE_CANDIDATES = 3


@runtime_checkable
class AttackGenerator(Protocol):
    async def generate(
        self,
        task: TaskSpec,
        *,
        seed: int = 0,
        count: int = 1,
        source_model: str | None = None,
        source_scaffold: str | None = None,
    ) -> list[Attack]: ...


def generate_unique_candidates(
    task: TaskSpec,
    *,
    family: str = "indirect_instruction",
    seed: int = 0,
    count: int = 3,
    source_model: str | None = None,
    source_scaffold: str | None = None,
) -> list[Attack]:
    """Deduplicate actual payloads, not IDs that merely differ by generation seed."""
    if type(count) is not int or not 1 <= count <= MAX_TEMPLATE_CANDIDATES:
        raise ValueError(
            f"Template generation supports 1..{MAX_TEMPLATE_CANDIDATES} unique candidates, not {count!r}"
        )
    family = canonical_family(family)
    if family == "none":
        raise ValueError("Cannot discover a no-attack template")
    pool = SyncTemplateGenerator(family).generate(
        task,
        seed=seed,
        count=MAX_TEMPLATE_CANDIDATES,
        source_model=source_model,
        source_scaffold=source_scaffold,
    )
    unique: dict[str, Attack] = {}
    for candidate in pool:
        if candidate.payload:
            unique.setdefault(candidate.payload, candidate)
    if len(unique) < count:
        raise ValueError(
            f"Only {len(unique)} distinct payloads exist; cannot generate {count} candidates"
        )
    return list(unique.values())[:count]


@dataclass(frozen=True)
class TemplateAttackGenerator:
    family: str = "indirect_instruction"

    async def generate(
        self,
        task: TaskSpec,
        *,
        seed: int = 0,
        count: int = 1,
        source_model: str | None = None,
        source_scaffold: str | None = None,
    ) -> list[Attack]:
        return generate_unique_candidates(
            task,
            family=self.family,
            seed=seed,
            count=count,
            source_model=source_model,
            source_scaffold=source_scaffold,
        )
