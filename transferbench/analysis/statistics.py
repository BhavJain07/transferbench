"""Wilson binomial intervals and reproducible, whole-task cluster bootstrap."""

from collections import defaultdict
from collections.abc import Callable, Sequence
from math import isfinite, sqrt

import numpy as np

BOOTSTRAP_SEED = 0
BOOTSTRAP_REPLICATES = 2000
Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int) -> list[float | None]:
    if n < 0 or not 0 <= successes <= n:
        raise ValueError("require 0 <= successes <= n")
    if not n:
        return [None, None]
    p = successes / n
    denominator = 1 + Z95**2 / n
    center = (p + Z95**2 / (2 * n)) / denominator
    radius = Z95 * sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def cluster_bootstrap(
    tasks: Sequence[str],
    values: Sequence[Sequence[float] | np.ndarray],
    statistic: Callable[[np.ndarray], float | None],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    """Resample TASK IDs with replacement; every repeat/pair stays in its task.

    ``statistic`` consumes the sum of additive vectors, one per observation.
    For comparisons each vector contains both members of a matched pair. Draws
    with undefined denominators are omitted and counted, not replaced with zero.
    """
    if len(tasks) != len(values):
        raise ValueError("tasks and values must have equal lengths")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    grouped: dict[str, list] = defaultdict(list)
    for task, value in zip(tasks, values, strict=True):
        grouped[task].append(value)
    if not grouped:
        return {"ci95": [None, None], "bootstrap_valid": 0}
    clusters = np.asarray([np.sum(grouped[key], axis=0) for key in sorted(grouped)])
    if not np.isfinite(clusters).all():
        raise ValueError("bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    samples = []
    # Bound temporary memory for runs containing many tasks.
    batch_size = max(1, min(128, 1_000_000 // max(1, clusters.size)))
    for start in range(0, replicates, batch_size):
        indices = rng.integers(
            0, len(clusters), (min(batch_size, replicates - start), len(clusters))
        )
        for total in clusters[indices].sum(axis=1):
            value = statistic(total)
            if value is not None and isfinite(value):
                samples.append(float(value))
    interval = np.quantile(samples, [0.025, 0.975]).tolist() if samples else [None, None]
    return {"ci95": interval, "bootstrap_valid": len(samples)}


def rate(outcomes: Sequence[bool], tasks: Sequence[str]) -> dict:
    """Pooled episode mean, Wilson CI, and task-cluster percentile CI."""
    if len(outcomes) != len(tasks):
        raise ValueError("outcomes and tasks must have equal lengths")
    n, successes = len(outcomes), sum(outcomes)
    bootstrap = cluster_bootstrap(tasks, [(float(x), 1) for x in outcomes], lambda v: v[0] / v[1])
    return {
        "n": n,
        "n_tasks": len(set(tasks)),
        "successes": successes,
        "mean": successes / n if n else None,
        "ci95": wilson_interval(successes, n),
        "ci_method": "Wilson 95%",
        "bootstrap_ci95": bootstrap["ci95"],
        "bootstrap_valid": bootstrap["bootstrap_valid"],
    }


def macro_rate_estimate(cells: Sequence[Sequence[tuple[str, float, float]]]) -> dict:
    """Macro mean of a fixed set of supported cells, jointly resampling tasks.

    Each cell holds (task_id, numerator, denominator) for its matched trials.
    One task draw is shared across ALL cells, preserving their dependence. If a
    draw loses support for any point-supported cell it is undefined, not a macro
    mean over a changing set of cells. ``n`` counts pair memberships, not unique
    episodes, since a trial can contribute to several directed cells.
    """
    supported = [cell for cell in cells if sum(value[2] for value in cell) > 0]
    grouped: dict[str, np.ndarray] = {}
    for index, cell in enumerate(supported):
        for task, numerator, denominator in cell:
            grouped.setdefault(task, np.zeros((len(supported), 2)))[index] += (
                numerator,
                denominator,
            )

    def statistic(total: np.ndarray) -> float | None:
        if not np.all(total[:, 1] > 0):
            return None
        return float(np.mean(total[:, 0] / total[:, 1]))

    estimate = paired_estimate(list(grouped), list(grouped.values()), statistic)
    estimate.update(
        n=sum(len(cell) for cell in supported),
        n_cells=len(supported),
        description="macro mean of fixed supported off-diagonal cells; jointly bootstrapped TASK clusters; n counts matched pair memberships",
    )
    return estimate


def paired_estimate(
    tasks: Sequence[str],
    values: Sequence[Sequence[float] | np.ndarray],
    statistic: Callable[[np.ndarray], float | None],
) -> dict:
    mean = statistic(np.asarray(values).sum(axis=0)) if values else None
    if mean is not None and not isfinite(mean):
        mean = None
    bootstrap = cluster_bootstrap(tasks, values, statistic)
    # An undefined point estimand must not acquire a misleading interval.
    if mean is None:
        bootstrap["ci95"] = [None, None]
    return {
        "n": len(values),
        "n_tasks": len(set(tasks)),
        "mean": float(mean) if mean is not None else None,
        "ci_method": "paired task-cluster percentile bootstrap 95%",
        **bootstrap,
    }
