"""Deterministic utility grading, deliberately separate from model prompts."""

from transferbench.tasks.schema import EpisodeResult, TaskSpec


def score_utility(task: TaskSpec, result: EpisodeResult) -> bool:
    """Require every expected fact in the final, visible answer."""
    answer = " ".join(result.output.casefold().split())
    return not result.error and all(
        " ".join(fact.casefold().split()) in answer for fact in task.expected_facts
    )
