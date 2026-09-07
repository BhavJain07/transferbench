"""Offline coverage of real Inspect adapters and observable synthetic effects."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from inspect_ai import Task, eval_async
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    Model,
    ModelOutput,
)
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from transferbench.attacks.library import get_attack
from transferbench.defenses.base import Defense
from transferbench.defenses.policies import ContextSeparation, Sanitizer, ToolPolicy
from transferbench.models.fake import FakeModel, get_fake_model
from transferbench.models.runtime import GenerationRuntime
from transferbench.scaffolds import SCAFFOLD_NAMES, get_scaffold
from transferbench.scaffolds._common import EpisodeSession
from transferbench.scaffolds.base import Scaffold as ScaffoldProtocol
from transferbench.scaffolds.chat import chat_multi, chat_single
from transferbench.scaffolds.delegation import delegate_agent
from transferbench.scaffolds.tools import tool_agent
from transferbench.scorers import score_attack, score_policy, score_utility
from transferbench.tasks.dataset import generate_synthetic_tasks
from transferbench.tasks.schema import CallCost, EpisodeResult, ToolCallRecord


class TestRuntime:
    """Scaffold-facing double with a genuine Inspect model, not runtime accounting.

    Cast only at the concrete GenerationRuntime boundary to keep this double independent.
    """

    __test__ = False

    def __init__(self, model: Model):
        self.model = model
        self.costs = []
        self.requests = []
        self.monitor_requests = []

    async def generate(self, messages, tools=None, purpose="assistant"):
        self.requests.append((purpose, [m.model_copy(deep=True) for m in messages], tools))
        output = await self.model.generate(messages, tools=tools or [])
        usage = output.usage
        self.costs.append(
            CallCost(
                provider="fake",
                model=self.model.name,
                purpose=purpose,
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
            )
        )
        return output

    async def monitor(self, messages):
        self.monitor_requests.append(messages)
        return await self.generate(messages, purpose="monitor")


class BudgetExceeded(Exception):
    """Test-only runtime failure; the adapter must not need its concrete import."""


class FailingRuntime(TestRuntime):
    def __init__(self, model, failure, *, after_action=True):
        super().__init__(model)
        self.failure = failure
        self.after_action = after_action
        self.budget = SimpleNamespace(exhausted=False)

    async def generate(self, messages, tools=None, purpose="assistant"):
        if not self.after_action or any(
            message.role == "tool" and message.function == "send_email" for message in messages
        ):
            self.requests.append((purpose, [m.model_copy(deep=True) for m in messages], tools))
            self.budget.exhausted = isinstance(self.failure, BudgetExceeded)
            # A runtime may know about a billed attempt even when it cannot return output.
            self.costs.append(
                CallCost(provider="fake", model=self.model.name, purpose=purpose, input_tokens=7)
            )
            raise self.failure
        return await super().generate(messages, tools, purpose)


@pytest.fixture
def task():
    return generate_synthetic_tasks()[0]


async def run_episode(
    task, scaffold, variant="vulnerable", attack=None, defense=None, max_turns=20
):
    model = get_fake_model(variant)
    runtime = TestRuntime(model)
    result = await get_scaffold(scaffold).run(
        task,
        model,
        defense or Defense(),
        attack,
        runtime=cast(GenerationRuntime, runtime),
        max_turns=max_turns,
    )
    return result, runtime


@pytest.mark.parametrize("scaffold", SCAFFOLD_NAMES)
@pytest.mark.parametrize("variant", ["vulnerable", "resistant"])
async def test_clean_controls(task, scaffold, variant):
    result, runtime = await run_episode(task, scaffold, variant)
    assert score_utility(task, result)
    assert not score_policy(task, result)
    assert not score_attack(task, result, None)
    assert not result.propagation_events
    assert len(result.costs) == len(runtime.requests)
    assert all(isinstance(call, CallCost) for call in result.costs)
    assert all(call.allowed for call in result.tool_calls if call.executed)
    for _, messages, _ in runtime.requests:
        for message in messages:
            if message.role != "system":
                assert task.canaries["secret"] not in message.text
    if scaffold == "delegate_agent":
        assert result.state["workers_visited"] == ["worker_A", "worker_B"]
        assert {purpose for purpose, _, _ in runtime.requests} == {
            "manager",
            "worker_A",
            "worker_B",
        }
    if scaffold.startswith("chat_"):
        assert not result.tool_calls
        assert all(not tools for _, _, tools in runtime.requests)


@pytest.mark.parametrize("scaffold", SCAFFOLD_NAMES)
@pytest.mark.parametrize(
    "family", ["direct", "indirect_instruction", "persistence", "delegation", "multi_turn_conflict"]
)
async def test_vulnerable_attacks_have_actual_effects(task, scaffold, family):
    attack = get_attack(family, task)
    result, _ = await run_episode(task, scaffold, attack=attack)
    assert score_utility(task, result)
    assert score_attack(task, result, attack)
    assert score_policy(task, result)
    if scaffold.startswith("chat_"):
        assert not any(call.executed for call in result.tool_calls)
        assert task.canaries["secret"] in result.output
    else:
        assert any(call.executed and not call.allowed for call in result.tool_calls)


@pytest.mark.parametrize("scaffold", SCAFFOLD_NAMES)
async def test_resistant_attacks_and_sanitizer_control(task, scaffold):
    attack = get_attack("indirect_instruction", task)
    resistant, _ = await run_episode(task, scaffold, "resistant", attack=attack)
    sanitized, _ = await run_episode(task, scaffold, attack=attack, defense=Sanitizer())
    for result in (resistant, sanitized):
        assert score_utility(task, result)
        assert not score_attack(task, result, attack)
        assert not score_policy(task, result)
        assert not any(e["depth"] > 0 for e in result.propagation_events)


async def test_multiworker_propagation_is_visible_and_directional(task):
    attack = get_attack("indirect_instruction", task)
    result, runtime = await run_episode(task, "delegate_agent", attack=attack)
    events = result.propagation_events
    assert {event["stage"] for event in events} == {"P0", "P1", "P2", "P3", "P4"}
    assert next(event for event in events if event["stage"] == "P0")["actor"] == "worker_B"
    assert any(
        event.get("source") == "worker_B" and event.get("target") == "manager" for event in events
    )
    assert any(
        event.get("source") == "manager" and event.get("target") == "worker_A" for event in events
    )
    assert any(
        call.actor == "worker_A" and call.name == "send_email" and call.executed
        for call in result.tool_calls
    )
    first_a = next(messages for actor, messages, _ in runtime.requests if actor == "worker_A")
    assert not any(attack.payload in message.text for message in first_a)
    assert "reasoning" not in json.dumps(events).lower()


async def test_blocked_attempt_is_not_success(task):
    attack = get_attack("indirect_instruction", task)
    result, _ = await run_episode(task, "tool_agent", attack=attack, defense=ToolPolicy())
    attempts = [call for call in result.tool_calls if call.name == "send_email"]
    assert attempts and all(not call.executed and call.blocked_by for call in attempts)
    assert not score_policy(task, result)
    assert not score_attack(task, result, attack)
    assert not any(event["stage"] == "P4" for event in result.propagation_events)
    assert not result.state["sent_emails"]


@pytest.mark.parametrize("scaffold", SCAFFOLD_NAMES)
@pytest.mark.parametrize("max_turns", [1, 2, 3, 5])
async def test_one_budget_for_entire_episode(task, scaffold, max_turns):
    result, runtime = await run_episode(
        task, scaffold, attack=get_attack("indirect_instruction", task), max_turns=max_turns
    )
    assert len(runtime.requests) <= max_turns
    assert result.state["generation_turns"] <= max_turns


async def test_monitor_and_all_hooks_use_same_runtime(task):
    class TrackingDefense(Defense):
        async def before_model(self, ctx):
            ctx.events.append({"hook": "before_model", "actor": ctx.actor})
            assert ctx.monitor is not None
            await ctx.monitor([ChatMessageSystem(content="Synthetic monitor check")])
            return replace(ctx, messages=list(ctx.messages))

        async def after_model(self, output, ctx):
            ctx.events.append({"hook": "after_model", "actor": ctx.actor})
            return output

        async def before_tool(self, call, ctx):
            ctx.events.append({"hook": "before_tool", "actor": ctx.actor})
            return True

        async def after_tool(self, result, ctx):
            ctx.events.append({"hook": "after_tool", "actor": ctx.actor})
            return result

    result, runtime = await run_episode(
        task, "delegate_agent", defense=TrackingDefense(), max_turns=12
    )
    assert runtime.monitor_requests
    assert len(runtime.requests) == len(result.costs) <= 12
    assert {event["hook"] for event in result.defense_events} == {
        "before_model",
        "after_model",
        "before_tool",
        "after_tool",
    }
    assert {event["actor"] for event in result.defense_events} >= {
        "worker_A",
        "worker_B",
        "manager",
    }


@pytest.mark.parametrize("scaffold", SCAFFOLD_NAMES)
async def test_context_separation_preserves_clean_utility(task, scaffold):
    result, _ = await run_episode(task, scaffold, defense=ContextSeparation())
    assert score_utility(task, result)
    assert not score_policy(task, result)
    assert result.defense_events


def test_policy_uses_emissions_not_inputs_attempts_or_reasoning(task):
    secret = task.canaries["secret"]
    result = EpisodeResult(
        output="A safe answer",
        transcript=[
            {"role": "system", "content": secret},
            {"role": "user", "content": secret},
            {"role": "tool", "content": secret},
            {"role": "assistant", "content": [{"type": "reasoning", "reasoning": secret}]},
            {"role": "assistant", "content": "", "tool_calls": [{"arguments": {"body": secret}}]},
        ],
        tool_calls=[
            ToolCallRecord(
                name="send_email",
                arguments={"body": secret},
                executed=False,
                allowed=False,
                blocked_by="test",
            )
        ],
    )
    assert not score_policy(task, result)
    result.output = task.canaries["destination"]
    assert not score_policy(task, result)
    result.transcript.append({"role": "assistant", "content": secret})
    assert score_policy(task, result)
    result.transcript.pop()
    result.tool_calls[0].executed = True
    assert score_policy(task, result)


async def test_fake_reads_documents_not_grader_metadata(task):
    changed = task.model_copy(update={"expected_facts": ["GOLD_FACT_NOT_IN_DOCUMENTS"]})
    result, runtime = await run_episode(changed, "chat_single")
    assert not score_utility(changed, result)
    assert "GOLD_FACT_NOT_IN_DOCUMENTS" not in result.output
    assert all(
        "GOLD_FACT_NOT_IN_DOCUMENTS" not in m.text
        for _, messages, _ in runtime.requests
        for m in messages
    )
    assert isinstance(runtime.model.api, FakeModel)
    fake_a = get_fake_model("fake_a").api
    fake_b = get_fake_model("fake_b").api
    assert isinstance(fake_a, FakeModel) and isinstance(fake_b, FakeModel)
    assert fake_a.variant == "vulnerable"
    assert fake_b.variant == "resistant"


@scorer(metrics=[accuracy()])
def shared_utility():
    async def score(state: TaskState, target: Target) -> Score:
        result = EpisodeResult.model_validate(state.metadata["episode_result"])
        task = generate_synthetic_tasks()[0]
        return Score(value="C" if score_utility(task, result) else "I")

    return score


@pytest.mark.parametrize("scaffold", ["tool_agent", "delegate_agent"])
@pytest.mark.parametrize("failed", [False, True])
async def test_native_inspect_task_and_tool_logs(task, scaffold, failed, tmp_path, monkeypatch):
    # Inspect keeps trace/cache metadata as well as the explicit evaluation log.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    model = get_fake_model()
    runtime = (
        FailingRuntime(model, RuntimeError("provider failed")) if failed else TestRuntime(model)
    )
    attack = get_attack("indirect_instruction", task) if failed else None
    evaluation = Task(
        dataset=[Sample(input=task.user_goal)],
        solver=get_scaffold(scaffold).as_solver(
            task, model, Defense(), attack, runtime=cast(GenerationRuntime, runtime)
        ),
        scorer=shared_utility(),
    )
    logs = await eval_async(evaluation, model=model, log_dir=str(tmp_path))
    assert logs[0].status == "success", logs[0].error
    assert logs[0].samples is not None
    sample = logs[0].samples[0]
    assert sample.scores is not None
    assert sample.scores["shared_utility"].value == ("I" if failed else "C")
    result = EpisodeResult.model_validate(sample.metadata["episode_result"])
    if failed:
        assert result.error == "RuntimeError: provider failed"
        assert score_policy(task, result)
        assert result.state["sent_emails"]
    assert any(event.event == "model" for event in sample.events)
    tools = [event for event in sample.events if event.event == "tool"]
    assert tools
    if scaffold == "delegate_agent":
        assert {event.function for event in tools} >= {"worker_A", "worker_B"}
        assert any(event.event == "span_begin" for event in sample.events)


@pytest.mark.parametrize("fixture_task", generate_synthetic_tasks(), ids=lambda task: task.task_id)
@pytest.mark.parametrize("scaffold", SCAFFOLD_NAMES)
async def test_all_authored_fixture_facts_come_from_documents(fixture_task, scaffold):
    result, _ = await run_episode(fixture_task, scaffold)
    assert score_utility(fixture_task, result)
    assert not score_policy(fixture_task, result)


async def test_wrapped_payload_is_still_an_observable_encounter(task):
    attack = get_attack("indirect_instruction", task)
    result, _ = await run_episode(task, "chat_single", attack=attack, defense=ContextSeparation())
    assert any(event["stage"] == "P0" for event in result.propagation_events)


@pytest.mark.parametrize("max_turns", [0, -1, True, 1.5])
async def test_invalid_turn_limits(task, max_turns):
    with pytest.raises(ValueError, match="positive integer"):
        await run_episode(task, "delegate_agent", max_turns=max_turns)


def test_manager_text_does_not_imply_delegation_message_was_affected(task):
    attack = get_attack("indirect_instruction", task)
    session = EpisodeSession(
        task,
        Defense(),
        attack,
        cast(GenerationRuntime, TestRuntime(get_fake_model())),
        20,
        "delegate_agent",
    )
    session.encounter([ChatMessageUser(content=attack.payload)], "manager")
    output = ModelOutput(
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=task.canaries["secret"],
                    tool_calls=[
                        ToolCall(
                            id="safe-delegation",
                            function="worker_A",
                            arguments={"input": "Review your source facts."},
                        )
                    ],
                )
            )
        ]
    )
    session.observe(output, "manager")
    assert {event["stage"] for event in session.events} == {"P0", "P1", "P3"}


@pytest.mark.parametrize("scaffold", ["tool_agent", "delegate_agent"])
@pytest.mark.parametrize("failure_type", [BudgetExceeded, RuntimeError])
async def test_runtime_failure_after_executed_violation_preserves_partial_result(
    task, scaffold, failure_type
):
    model = get_fake_model()
    runtime = FailingRuntime(model, failure_type("synthetic failure after execution"))
    previous_cost = CallCost(provider="fake", model=model.name, purpose="previous_episode")
    runtime.costs.append(previous_cost)
    attack = get_attack("indirect_instruction", task)
    result = await get_scaffold(scaffold).run(
        task, model, Defense(), attack, runtime=cast(GenerationRuntime, runtime)
    )
    assert result.error == f"{failure_type.__name__}: synthetic failure after execution"
    assert result.state["truncated"]
    assert not result.state["completed_response"]
    assert result.state["termination_reason"] == failure_type.__name__
    assert runtime.budget.exhausted is (failure_type is BudgetExceeded)
    assert result.costs == runtime.costs[1:]
    assert result.costs[-1].input_tokens == 7
    assert result.state["sent_emails"]
    assert any(
        call.name == "send_email" and call.executed and not call.allowed
        for call in result.tool_calls
    )
    assert any(event["stage"] == "P4" for event in result.propagation_events)
    assert any(
        message["role"] == "tool" and message.get("function") == "send_email"
        for message in result.transcript
    )
    actor = "manager" if scaffold == "delegate_agent" else "assistant"
    root_outputs = [
        message
        for message in result.transcript
        if message["actor"] == actor and message["phase"] == "output"
    ]
    assert root_outputs and result.output == root_outputs[-1]["content"]
    assert score_policy(task, result)
    assert score_attack(task, result, attack)
    assert not score_utility(task, result)


async def test_monitor_failure_preserves_executed_effects(task):
    class MonitoredDefense(Defense):
        async def before_model(self, ctx):
            assert ctx.monitor is not None
            await ctx.monitor(ctx.messages)
            return ctx

    model = get_fake_model()
    runtime = FailingRuntime(model, BudgetExceeded("monitor budget exhausted"))
    result = await get_scaffold("tool_agent").run(
        task,
        model,
        MonitoredDefense(),
        get_attack("indirect_instruction", task),
        runtime=cast(GenerationRuntime, runtime),
    )
    assert result.error == "BudgetExceeded: monitor budget exhausted"
    assert runtime.requests[-1][0] == "monitor"
    assert result.transcript[-1]["actor"] == "monitor"
    assert result.costs == runtime.costs
    assert result.state["sent_emails"]
    assert score_policy(task, result)


@pytest.mark.parametrize("scaffold", SCAFFOLD_NAMES)
async def test_failure_before_first_response_still_returns_inputs_and_costs(task, scaffold):
    model = get_fake_model()
    runtime = FailingRuntime(model, RuntimeError("provider unavailable"), after_action=False)
    result = await get_scaffold(scaffold).run(
        task, model, Defense(), None, runtime=cast(GenerationRuntime, runtime)
    )
    assert result.error == "RuntimeError: provider unavailable"
    assert result.output == ""
    assert result.transcript and result.costs == runtime.costs
    assert result.state["truncated"]
    assert not score_utility(task, result)


@pytest.mark.parametrize("failure_type", [asyncio.CancelledError, SystemExit])
async def test_cancellation_and_process_exit_propagate(task, failure_type):
    model = get_fake_model()
    runtime = FailingRuntime(model, failure_type("stop requested"), after_action=False)
    with pytest.raises(failure_type, match="stop requested"):
        await get_scaffold("chat_single").run(
            task, model, Defense(), None, runtime=cast(GenerationRuntime, runtime)
        )


@pytest.mark.parametrize("scaffold", ["chat_multi", "tool_agent", "delegate_agent"])
async def test_turn_exhaustion_is_a_truncated_episode(task, scaffold):
    result, runtime = await run_episode(task, scaffold, max_turns=1)
    assert len(runtime.requests) == 1
    assert result.error is not None
    assert result.error.startswith("TurnLimitReached:")
    assert result.state["turn_limit_reached"]
    assert result.state["truncated"]
    assert not result.state["completed_response"]
    assert not score_utility(task, result)


@pytest.mark.parametrize(
    "scaffold,max_turns",
    [("chat_single", 1), ("chat_multi", 2), ("tool_agent", 2), ("delegate_agent", 4)],
)
async def test_final_response_exactly_at_turn_limit_is_complete(task, scaffold, max_turns):
    result, _ = await run_episode(task, scaffold, max_turns=max_turns)
    assert result.error is None
    assert result.state["turn_limit_reached"]
    assert result.state["completed_response"]
    assert not result.state["truncated"]
    assert result.state["termination_reason"] == "completed"
    assert score_utility(task, result)


@pytest.mark.parametrize("scaffold,max_turns", [("tool_agent", 2), ("delegate_agent", 5)])
async def test_turn_exhaustion_keeps_executed_violations(task, scaffold, max_turns):
    attack = get_attack("indirect_instruction", task)
    result, runtime = await run_episode(task, scaffold, attack=attack, max_turns=max_turns)
    assert result.error is not None
    assert result.error.startswith("TurnLimitReached:")
    assert len(runtime.requests) == max_turns
    assert result.state["sent_emails"]
    assert any(call.name == "send_email" and call.executed for call in result.tool_calls)
    assert any(event["stage"] == "P4" for event in result.propagation_events)
    assert score_policy(task, result) and score_attack(task, result, attack)
    assert not score_utility(task, result)


async def test_model_token_truncation_is_not_a_completed_response(task):
    class TruncatedRuntime(TestRuntime):
        async def generate(self, messages, tools=None, purpose="assistant"):
            output = await super().generate(messages, tools, purpose)
            output.choices[0].stop_reason = "max_tokens"
            return output

    model = get_fake_model()
    runtime = TruncatedRuntime(model)
    result = await get_scaffold("chat_single").run(
        task, model, Defense(), None, runtime=cast(GenerationRuntime, runtime)
    )
    assert result.error is not None
    assert result.error.startswith("IncompleteResponse:")
    assert result.output
    assert result.state["truncated"]
    assert not score_utility(task, result)


@pytest.mark.parametrize(
    "adapter",
    [chat_single, chat_multi, tool_agent, delegate_agent],
    ids=lambda adapter: adapter.name,
)
async def test_compatibility_exports_implement_scaffold_protocol(task, adapter):
    assert isinstance(adapter, ScaffoldProtocol)
    assert adapter == get_scaffold(adapter.name)
    model = get_fake_model()
    result = await adapter.run(
        task, model, Defense(), None, runtime=cast(GenerationRuntime, TestRuntime(model))
    )
    assert score_utility(task, result)


def test_invalid_names():
    with pytest.raises(ValueError, match="Unknown scaffold"):
        get_scaffold("nonexistent")
    with pytest.raises(ValueError, match="Unknown fake variant"):
        get_fake_model("nonexistent")
