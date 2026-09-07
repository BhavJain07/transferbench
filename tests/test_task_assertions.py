"""Prevent vacuous or redundant deterministic utility assertions."""

import pytest
from pydantic import ValidationError

from transferbench.scorers.utility import score_utility
from transferbench.tasks.dataset import generate_synthetic_tasks
from transferbench.tasks.schema import EpisodeResult, TaskSpec


@pytest.mark.parametrize("fact", ["", " ", "\t\n", "\u2003\u00a0"])
def test_blank_expected_facts_are_rejected(fact):
    data = generate_synthetic_tasks()[0].model_dump()
    data["expected_facts"] = [fact]
    with pytest.raises(ValidationError, match="blank assertions"):
        TaskSpec.model_validate(data)


@pytest.mark.parametrize(
    "facts",
    [
        ["approved", "approved"],
        ["Revenue Report", "revenue report"],
        ["45 days", " 45\tDAYS\n"],
        ["STRASSE", "Straße"],
    ],
)
def test_grading_equivalent_facts_are_rejected(facts):
    data = generate_synthetic_tasks()[0].model_dump()
    data["expected_facts"] = facts
    with pytest.raises(ValidationError, match="unique after grading normalization"):
        TaskSpec.model_validate(data)


def test_exact_source_fact_text_is_preserved():
    data = generate_synthetic_tasks()[0].model_dump()
    data["expected_facts"] = ["Revenue\tReport", "45 DAYS"]
    task = TaskSpec.model_validate(data)
    assert task.expected_facts == ["Revenue\tReport", "45 DAYS"]
    assert score_utility(task, EpisodeResult(output="Revenue report: 45 days."))


def test_all_authored_tasks_still_validate_and_grade():
    tasks = generate_synthetic_tasks()
    assert len(tasks) == 24
    for original in tasks:
        task = TaskSpec.model_validate_json(original.model_dump_json())
        assert score_utility(task, EpisodeResult(output="\n".join(task.expected_facts)))
        assert not score_utility(task, EpisodeResult(output=""))
