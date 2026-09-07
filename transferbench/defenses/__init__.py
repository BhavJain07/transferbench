from transferbench.defenses.base import (
    Defense,
    DefenseChain,
    DefenseContext,
    is_untrusted,
    mark_untrusted,
)
from transferbench.defenses.monitor import SecondModelMonitor
from transferbench.defenses.policies import ContextSeparation, NoDefense, Sanitizer, ToolPolicy

DEFENSE_NAMES = ("none", "context_separation", "tool_policy", "sanitizer", "second_model_monitor")


def get_defense(name: str) -> Defense:
    """Resolve one defense or an ordered '+' composition, with no scaffold branching."""
    if "+" in name:
        return DefenseChain(get_defense(part.strip()) for part in name.split("+"))
    registry = {
        "none": NoDefense,
        "context_separation": ContextSeparation,
        "tool_policy": ToolPolicy,
        "sanitizer": Sanitizer,
        "second_model_monitor": SecondModelMonitor,
        "monitor": SecondModelMonitor,
    }
    try:
        return registry[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown defense {name!r}; choose from {list(DEFENSE_NAMES)}") from exc


__all__ = [
    "DEFENSE_NAMES",
    "ContextSeparation",
    "Defense",
    "DefenseChain",
    "DefenseContext",
    "NoDefense",
    "Sanitizer",
    "SecondModelMonitor",
    "ToolPolicy",
    "get_defense",
    "is_untrusted",
    "mark_untrusted",
]
