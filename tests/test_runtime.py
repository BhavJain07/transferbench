"""Offline runtime/accounting contracts, exercised through a genuine Inspect Model.

No provider SDK, API key, network connection, or paid model is used. Safety
regressions assert the intended contract rather than blessing under-accounting.
"""

import asyncio
import copy
import math
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest
import yaml
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import Tool, ToolChoice, ToolInfo, tool
from pydantic import ValidationError
from tenacity import RetryError

from transferbench.models import registry as registry_module
from transferbench.models.registry import ModelRegistry, ModelSpec, Pricing, finite_cost, read_yaml
from transferbench.models.runtime import (
    BudgetExceeded,
    CostBudget,
    GenerationRuntime,
    UsageLimitExceeded,
)
from transferbench.runner import config as config_module
from transferbench.runner.config import (
    Experiment,
    Generation,
    Limits,
    MatrixConfig,
    TaskConfig,
    config_tasks,
    load_config,
    resolve_path,
)
from transferbench.tasks.dataset import generate_synthetic_tasks


class RetryableProbeError(RuntimeError):
    pass


class ProbeAPI(ModelAPI):
    """Controlled provider responses and a record of Inspect's effective config."""

    def __init__(self, model_name="controlled-v1", *, usage=None, omit_usage=False, error=None):
        super().__init__(model_name)
        self.usage = (
            usage
            if usage is not None
            else ModelUsage(input_tokens=10, output_tokens=4, total_tokens=14)
        )
        self.omit_usage = omit_usage
        self.error = error
        self.calls = []
        self.gate: asyncio.Event | None = None
        self.started = asyncio.Event()
        self.expected_started = 1

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        self.calls.append(
            {
                "messages": [message.model_copy(deep=True) for message in input],
                "tools": copy.deepcopy(tools),
                "config": config.model_copy(deep=True),
            }
        )
        if len(self.calls) >= self.expected_started:
            self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        output = ModelOutput.from_content(self.model_name, f"probe seed={config.seed}")
        output.usage = None if self.omit_usage else self.usage.model_copy(deep=True)
        return output

    def should_retry(self, ex: Exception) -> bool:
        return isinstance(ex, RetryableProbeError)

    def retry_wait(self):
        return lambda retry_state: 0


@modelapi("transferbench_runtime_probe")
def probe_api() -> type[ProbeAPI]:
    return ProbeAPI


def make_spec(**updates):
    data = {
        "alias": "probe",
        "provider": "transferbench_runtime_probe",
        "model": "controlled-v1",
        "family": "offline-probe",
        "pricing": {"input_per_million": 1.0, "output_per_million": 2.0},
        "max_input_tokens": 16384,
    }
    return ModelSpec.model_validate(data | updates)


def make_runtime(
    *, spec=None, budget=None, model_config=None, usage=None, omit_usage=False, error=None, **kwargs
) -> tuple[GenerationRuntime, ProbeAPI]:
    # Inspect's decorator exposes a generic ModelAPI constructor, not ProbeAPI's signature.
    api = cast(Callable[..., ProbeAPI], probe_api)(usage=usage, omit_usage=omit_usage, error=error)
    model = Model(api=api, config=model_config or GenerateConfig())
    runtime = GenerationRuntime(
        model,
        spec or make_spec(),
        budget if budget is not None else CostBudget(1),
        max_tokens=64,
        **kwargs,
    )
    return runtime, api


@pytest.fixture
def messages() -> list[ChatMessage]:
    return [ChatMessageUser(content="Offline synthetic request.")]


def model_entry(**updates):
    return {
        "provider": "fake",
        "model": "vulnerable",
        "pricing": {"input_per_million": 0, "output_per_million": 0},
    } | updates


def matrix_data(**updates):
    return {"models": ["fake_a"], "scaffolds": ["chat_single"]} | updates


def yaml_file(tmp_path, data, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# Reservation invariants -----------------------------------------------------


@pytest.mark.parametrize("value", [-1, math.inf, -math.inf, math.nan])
def test_budget_and_reservation_reject_invalid_costs(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        CostBudget(value)
    budget = CostBudget(1)
    with pytest.raises(ValueError, match="finite nonnegative"):
        budget.reserve(value)
    assert budget.charged == budget.reserved == 0
    assert not budget.exhausted


@pytest.mark.parametrize("value", [0, 0.001, 2.5])
def test_finite_cost_accepts_nonnegative_finite_values(value):
    assert finite_cost(value) == value


def test_reservations_cannot_exceed_hard_cap():
    budget = CostBudget(0.3)
    first = budget.reserve(0.1)
    second = budget.reserve(0.2)
    assert budget.charged + budget.reserved == pytest.approx(budget.maximum)
    with pytest.raises(BudgetExceeded, match="next-call bound"):
        budget.reserve(0.000001)
    assert budget.reserved == pytest.approx(0.3)
    assert budget.charged == 0
    assert budget.exhausted
    budget.settle(first, 0.01)
    budget.settle(second, 0.02)
    assert budget.reserved == pytest.approx(0, abs=1e-12)
    assert budget.charged == pytest.approx(0.03)
    with pytest.raises(BudgetExceeded):
        budget.reserve(0)


def test_settlement_releases_only_unused_bound():
    budget = CostBudget(1)
    first, second = budget.reserve(0.4), budget.reserve(0.5)
    budget.settle(first, 0.1)
    assert budget.charged == pytest.approx(0.1)
    assert budget.reserved == pytest.approx(0.5)
    third = budget.reserve(0.4)
    budget.settle(second, None)
    budget.settle(third, 0.2)
    assert budget.charged == pytest.approx(0.8)
    assert budget.reserved == pytest.approx(0)
    assert budget.uncertain and not budget.exhausted


def test_reported_cost_over_reservation_is_charged_and_stops_ledger():
    budget = CostBudget(1)
    bound = budget.reserve(0.1)
    with pytest.raises(UsageLimitExceeded, match="exceeded reserved bound"):
        budget.settle(bound, 0.2)
    assert budget.charged == pytest.approx(0.2)
    assert budget.reserved == 0
    assert budget.uncertain and budget.exhausted
    with pytest.raises(BudgetExceeded):
        budget.reserve(0.01)


def test_concurrent_thread_reservations_are_atomic():
    budget = CostBudget(8)
    barrier = threading.Barrier(32)

    def reserve():
        barrier.wait(timeout=5)
        try:
            return budget.reserve(1)
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(lambda _: reserve(), range(32)))
    accepted = [result for result in results if result is not None]
    assert len(accepted) == 8
    assert budget.reserved == 8 and budget.charged == 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda amount: budget.settle(amount, 0.25), accepted))
    assert budget.reserved == 0 and budget.charged == 2


# Genuine Inspect generation and shared accounting ---------------------------


async def test_success_records_actual_usage_and_releases_reservation(messages):
    runtime, api = make_runtime()
    expected = runtime.spec.pricing.estimate(10, 4)
    output = await runtime.generate(messages, purpose="worker_A")
    assert output.completion == "probe seed=None"
    assert len(api.calls) == len(runtime.costs) == 1
    assert runtime.budget.charged == pytest.approx(expected)
    assert runtime.budget.reserved == 0
    assert not runtime.budget.uncertain and not runtime.failed
    cost = runtime.costs[0]
    assert (cost.input_tokens, cost.output_tokens, cost.cached_tokens) == (10, 4, 0)
    assert cost.estimated_cost_usd == pytest.approx(expected)
    assert cost.provider == runtime.spec.provider and cost.model == runtime.spec.model
    assert cost.purpose == "worker_A" and cost.latency_ms >= 0


async def test_repeated_successes_charge_each_call_without_cache(messages):
    runtime, api = make_runtime()
    for _ in range(3):
        await runtime.generate(messages)
    assert len(api.calls) == len(runtime.costs) == 3
    assert runtime.budget.charged == pytest.approx(
        sum(call.estimated_cost_usd for call in runtime.costs)
    )
    assert runtime.budget.reserved == 0


async def test_insufficient_bound_prevents_any_provider_request(messages):
    runtime, api = make_runtime(budget=CostBudget(0))
    with pytest.raises(BudgetExceeded):
        await runtime.generate(messages)
    assert not api.calls and not runtime.costs
    assert runtime.budget.charged == runtime.budget.reserved == 0


async def test_concurrent_runtimes_share_one_hard_reservation_ledger(messages):
    spec = make_spec()
    bound = spec.pricing.estimate(spec.max_input_tokens, 64)
    budget = CostBudget(2 * bound)
    pairs = [make_runtime(spec=spec, budget=budget) for _ in range(8)]
    gate = asyncio.Event()
    for _, api in pairs:
        api.gate = gate
    tasks = [asyncio.create_task(runtime.generate(messages)) for runtime, _ in pairs]

    async def wait_for_admitted_calls():
        while sum(len(api.calls) for _, api in pairs) < 2:
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(wait_for_admitted_calls(), timeout=5)
        assert sum(len(api.calls) for _, api in pairs) == 2
        assert budget.charged + budget.reserved <= budget.maximum + 1e-12
    finally:
        gate.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
    assert sum(isinstance(result, ModelOutput) for result in results) == 2
    assert sum(isinstance(result, BudgetExceeded) for result in results) == 6
    assert budget.reserved == pytest.approx(0)
    assert budget.charged == pytest.approx(
        sum(cost.estimated_cost_usd for runtime, _ in pairs for cost in runtime.costs)
    )


@pytest.mark.parametrize(
    "error_type", [RuntimeError, TimeoutError, RetryableProbeError, asyncio.CancelledError]
)
async def test_paid_failure_retains_full_bound_and_never_retries(messages, error_type):
    runtime, api = make_runtime(error=error_type("synthetic provider failure"))
    bound = runtime.bound(runtime.spec)
    with pytest.raises((error_type, RetryError)) as caught:
        await asyncio.wait_for(runtime.generate(messages, purpose="worker_B"), timeout=5)
    # Inspect wraps retryable failures in RetryError even with zero retries.
    failure = (
        caught.value.last_attempt.exception()
        if isinstance(caught.value, RetryError)
        else caught.value
    )
    assert isinstance(failure, error_type)
    assert str(failure) == "synthetic provider failure"
    assert len(api.calls) == 1
    assert api.calls[0]["config"].max_retries == 0
    assert runtime.budget.reserved == 0
    assert runtime.budget.charged == pytest.approx(bound)
    assert runtime.budget.uncertain and runtime.failed
    assert len(runtime.costs) == 1
    assert runtime.costs[0].estimated_cost_usd == pytest.approx(bound)
    assert runtime.costs[0].purpose == "worker_B:unknown_usage_reserved"
    with pytest.raises(UsageLimitExceeded, match="earlier model/monitor call failed"):
        await runtime.generate(messages)
    assert len(api.calls) == 1
    assert runtime.budget.charged == pytest.approx(bound)


async def test_failed_calls_across_episodes_cannot_reuse_spent_bound(messages):
    spec = make_spec()
    bound = spec.pricing.estimate(spec.max_input_tokens, 64)
    budget = CostBudget(2 * bound)
    pairs = [
        make_runtime(spec=spec, budget=budget, error=RuntimeError("paid failure")) for _ in range(3)
    ]
    for runtime, _ in pairs[:2]:
        with pytest.raises(RuntimeError, match="paid failure"):
            await runtime.generate(messages)
    with pytest.raises(BudgetExceeded):
        await pairs[2][0].generate(messages)
    assert [len(api.calls) for _, api in pairs] == [1, 1, 0]
    assert budget.charged == pytest.approx(2 * bound)
    assert budget.reserved == pytest.approx(0)
    assert budget.exhausted and budget.uncertain


async def test_nonfinite_pricing_bound_cannot_send_a_request(messages):
    runtime, api = make_runtime(
        spec=make_spec(pricing={"input_per_million": 1e308, "output_per_million": 1e308})
    )
    with pytest.raises(ValueError, match="finite nonnegative"):
        await runtime.generate(messages)
    assert not api.calls and not runtime.costs
    assert runtime.budget.charged == runtime.budget.reserved == 0


async def test_paid_missing_usage_charges_bound_and_stops(messages):
    runtime, api = make_runtime(omit_usage=True)
    with pytest.raises(UsageLimitExceeded, match="omitted token usage"):
        await runtime.generate(messages)
    assert len(api.calls) == 1
    assert runtime.budget.charged == pytest.approx(runtime.bound(runtime.spec))
    assert runtime.budget.reserved == 0
    assert runtime.budget.uncertain and runtime.failed
    assert runtime.costs[0].purpose.endswith(":unknown_usage_reserved")


async def test_fake_zero_price_is_usable_under_zero_budget(messages):
    spec = ModelSpec(alias="fake_a", family="offline", **model_entry())
    runtime = GenerationRuntime(spec.resolve(), spec, CostBudget(0), max_tokens=64)
    output = await runtime.generate(messages)
    assert isinstance(output, ModelOutput)
    assert runtime.costs and runtime.costs[0].estimated_cost_usd == 0
    assert runtime.budget.charged == runtime.budget.reserved == 0
    assert not runtime.failed


@pytest.mark.parametrize("inputs,outputs", [(16385, 1), (10, 65), (32768, 128)])
async def test_provider_token_overage_fails_closed_with_actual_cost_retained(
    messages, inputs, outputs
):
    usage = ModelUsage(input_tokens=inputs, output_tokens=outputs, total_tokens=inputs + outputs)
    runtime, api = make_runtime(usage=usage)
    expected = runtime.spec.pricing.estimate(inputs, outputs)
    with pytest.raises(UsageLimitExceeded, match="exceeded"):
        await runtime.generate(messages)
    assert len(api.calls) == len(runtime.costs) == 1
    assert runtime.budget.charged == pytest.approx(expected)
    assert runtime.costs[0].estimated_cost_usd == pytest.approx(expected)
    assert runtime.budget.reserved == 0
    assert runtime.budget.exhausted and runtime.failed
    with pytest.raises(UsageLimitExceeded):
        await runtime.generate(messages)
    assert len(api.calls) == 1


async def test_overage_does_not_settle_another_inflight_calls_reservation(messages):
    spec = make_spec()
    bound = spec.pricing.estimate(spec.max_input_tokens, 64)
    budget = CostBudget(3 * bound)
    pending_runtime, pending_api = make_runtime(spec=spec, budget=budget)
    overrun_runtime, _ = make_runtime(
        spec=spec,
        budget=budget,
        usage=ModelUsage(
            input_tokens=2 * spec.max_input_tokens,
            output_tokens=128,
            total_tokens=2 * spec.max_input_tokens + 128,
        ),
    )
    pending_api.gate = asyncio.Event()
    pending = asyncio.create_task(pending_runtime.generate(messages))
    try:
        await asyncio.wait_for(pending_api.started.wait(), timeout=5)
        with pytest.raises(UsageLimitExceeded):
            await asyncio.wait_for(overrun_runtime.generate(messages), timeout=5)
        assert budget.reserved == pytest.approx(bound)
        assert budget.charged == pytest.approx(2 * bound)
        assert len(overrun_runtime.costs) == 1
    finally:
        pending_api.gate.set()
        await asyncio.wait_for(pending, timeout=5)
    assert budget.reserved == pytest.approx(0)
    assert budget.charged == pytest.approx(
        sum(
            cost.estimated_cost_usd
            for runtime in (pending_runtime, overrun_runtime)
            for cost in runtime.costs
        )
    )


async def test_cache_read_and_write_tokens_are_conservatively_counted(messages):
    usage = ModelUsage(
        input_tokens=10,
        input_tokens_cache_read=20,
        input_tokens_cache_write=30,
        output_tokens=4,
        total_tokens=64,
    )
    runtime, _ = make_runtime(usage=usage)
    await runtime.generate(messages)
    cost = runtime.costs[0]
    assert cost.input_tokens == 60 and cost.cached_tokens == 20
    assert cost.estimated_cost_usd == pytest.approx(runtime.spec.pricing.estimate(60, 4))


@tool
def probe_tool() -> Tool:
    async def execute(query: str) -> str:
        """Echo synthetic reference input.

        Args:
            query: Synthetic reference query.
        """
        return query

    return execute


@pytest.mark.parametrize("kind", ["text", "unicode", "tool_schema"])
async def test_serialized_input_bound_blocks_oversize_before_reserving(
    messages: list[ChatMessage], kind
):
    runtime, api = make_runtime(spec=make_spec(max_input_tokens=9000))
    tools = None
    if kind == "text":
        messages = [ChatMessageUser(content="x" * 1000)]
    elif kind == "unicode":
        messages = [ChatMessageUser(content="界" * 350)]
    else:
        from inspect_ai.tool import ToolDef

        tools = [ToolDef(probe_tool(), description="Synthetic " * 200).as_tool()]
    with pytest.raises(UsageLimitExceeded, match="no request sent"):
        await runtime.generate(messages, tools=tools)
    assert not api.calls and not runtime.costs
    assert runtime.budget.charged == runtime.budget.reserved == 0


async def test_tools_are_forwarded_to_inspect_without_execution(messages):
    runtime, api = make_runtime()
    await runtime.generate(messages, tools=[probe_tool()])
    assert [tool.name for tool in api.calls[0]["tools"]] == ["probe_tool"]
    assert "query" in api.calls[0]["tools"][0].parameters.properties


@pytest.mark.parametrize("supports_seed", [False, True])
async def test_effective_generation_config_and_deterministic_seed(messages, supports_seed):
    runtime, api = make_runtime(
        spec=make_spec(supports_seed=supports_seed), seed=123, temperature=0.4, timeout_seconds=7
    )
    first = await runtime.generate(messages)
    second = await runtime.generate(messages)
    assert first.completion == second.completion
    for call in api.calls:
        config = call["config"]
        assert config.seed == (123 if supports_seed else None)
        assert config.max_tokens == 64 and config.temperature == 0.4
        assert config.timeout == 7 and config.max_retries == 0
        assert config.parallel_tool_calls is False
        assert config.cache is False and config.cache_prompt is False


async def test_runtime_overrides_inherited_retry_and_output_limits(messages):
    runtime, api = make_runtime(
        model_config=GenerateConfig(
            max_retries=8,
            max_tokens=2048,
            temperature=1.9,
            cache=True,
            cache_prompt=True,
            parallel_tool_calls=True,
        )
    )
    await runtime.generate(messages)
    config = api.calls[0]["config"]
    assert config.max_retries == 0 and config.max_tokens == 64
    assert config.temperature == 0.2
    assert config.cache is config.cache_prompt is config.parallel_tool_calls is False


async def test_monitor_and_assistant_use_same_cost_ledger(messages):
    runtime, api = make_runtime(spec=make_spec(supports_seed=True), seed=31)
    monitor, monitor_api = make_runtime(
        spec=make_spec(
            alias="monitor",
            supports_seed=False,
            pricing={"input_per_million": 4, "output_per_million": 8},
        )
    )
    runtime.monitor_model, runtime.monitor_spec = monitor.model, monitor.spec
    await runtime.generate(messages, purpose="manager")
    await runtime.monitor(messages)
    assert len(api.calls) == len(monitor_api.calls) == 1
    assert [cost.purpose for cost in runtime.costs] == ["manager", "monitor"]
    assert runtime.costs[1].estimated_cost_usd == pytest.approx(
        4 * runtime.costs[0].estimated_cost_usd
    )
    assert runtime.budget.charged == pytest.approx(
        sum(cost.estimated_cost_usd for cost in runtime.costs)
    )
    assert runtime.budget.reserved == 0
    assert api.calls[0]["config"].seed == 31 and monitor_api.calls[0]["config"].seed is None
    assert not monitor.costs


@pytest.mark.parametrize("failure_at", ["assistant", "monitor"])
async def test_failure_stops_both_assistant_and_monitor(messages, failure_at):
    runtime, api = make_runtime(
        error=RuntimeError("assistant failed") if failure_at == "assistant" else None
    )
    monitor, monitor_api = make_runtime(
        error=RuntimeError("monitor failed") if failure_at == "monitor" else None
    )
    runtime.monitor_model, runtime.monitor_spec = monitor.model, monitor.spec
    operation = runtime.generate if failure_at == "assistant" else runtime.monitor
    with pytest.raises(RuntimeError, match=f"{failure_at} failed"):
        await operation(messages)
    for operation in (runtime.generate, runtime.monitor):
        with pytest.raises(UsageLimitExceeded, match="earlier model/monitor call failed"):
            await operation(messages)
    assert len(api.calls) + len(monitor_api.calls) == 1
    assert runtime.costs[0].purpose.startswith(
        "assistant" if failure_at == "assistant" else "monitor"
    )


async def test_monitor_reserves_before_call_and_cannot_use_separate_budget(messages):
    runtime, api = make_runtime()
    monitor, monitor_api = make_runtime()
    runtime.monitor_model, runtime.monitor_spec = monitor.model, monitor.spec
    runtime.budget = CostBudget(runtime.bound(runtime.spec))
    api.gate = asyncio.Event()
    pending = asyncio.create_task(runtime.generate(messages))
    try:
        await asyncio.wait_for(api.started.wait(), timeout=5)
        with pytest.raises(BudgetExceeded):
            await runtime.monitor(messages)
        assert not monitor_api.calls
    finally:
        api.gate.set()
        await asyncio.wait_for(pending, timeout=5)
    assert runtime.budget.reserved == 0


@pytest.mark.parametrize("which", ["both", "model", "spec"])
async def test_monitor_requires_complete_explicit_configuration(messages, which):
    runtime, api = make_runtime()
    if which == "model":
        runtime.monitor_model = runtime.model
    elif which == "spec":
        runtime.monitor_spec = runtime.spec
    with pytest.raises(ValueError, match="explicit monitor_model"):
        await runtime.monitor(messages)
    assert not api.calls and not runtime.costs


# Registry, pricing, YAML and selected-only environment expansion -------------


@pytest.mark.parametrize("field", ["input_per_million", "output_per_million"])
@pytest.mark.parametrize("value", [-1, math.nan, math.inf, -math.inf])
def test_pricing_rejects_negative_or_nonfinite_rates(field, value):
    with pytest.raises(ValidationError):
        Pricing.model_validate({"input_per_million": 1, "output_per_million": 2} | {field: value})


@pytest.mark.parametrize(
    "pricing",
    [
        None,
        {},
        {"input_per_million": 1},
        {"output_per_million": 1},
        {"input_per_million": 1, "output_per_million": 2, "discount": 0.5},
    ],
)
def test_model_pricing_is_required_complete_and_strict(pricing):
    with pytest.raises(ValidationError):
        make_spec(pricing=pricing)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "self_hosted"])
@pytest.mark.parametrize("field", ["input_per_million", "output_per_million"])
def test_every_nonsimulated_provider_requires_positive_rates(provider, field):
    pricing = {"input_per_million": 1, "output_per_million": 2, field: 0}
    with pytest.raises(ValidationError, match="positive conservative token prices"):
        make_spec(provider=provider, pricing=pricing)


def test_pricing_estimate_uses_per_million_units():
    pricing = Pricing(input_per_million=2, output_per_million=8)
    assert pricing.estimate(1_000_000, 500_000) == 6
    assert pricing.estimate(0, 0) == 0


@pytest.mark.parametrize("model", ["", "  ", "${UNRESOLVED}"])
def test_model_identifier_must_be_resolved_nonempty(model):
    with pytest.raises(ValidationError, match="resolved"):
        make_spec(model=model)


def test_fake_variant_and_input_limit_validation():
    with pytest.raises(ValidationError, match="vulnerable or resistant"):
        make_spec(provider="fake", model="other")
    with pytest.raises(ValidationError):
        make_spec(max_input_tokens=1023)


def test_registry_rejects_unknown_alias_and_selected_model_fields(tmp_path):
    path = yaml_file(tmp_path, {"models": {"fake_a": model_entry(extra_option=True)}})
    registry = ModelRegistry(path)
    with pytest.raises(ValueError, match="Unknown model alias"):
        registry.get("missing")
    with pytest.raises(ValidationError, match="extra_option"):
        registry.get("fake_a")


@pytest.mark.parametrize(
    "data", [{}, {"models": []}, {"models": {}, "extra": 1}, [], None, "scalar"]
)
def test_yaml_and_registry_require_exact_root_mapping(tmp_path, data):
    path = yaml_file(tmp_path, data)
    with pytest.raises(ValueError):
        ModelRegistry(path)


@pytest.mark.parametrize(
    "text",
    [
        "models: {}\nmodels: {}\n",
        "models:\n  alias: {}\n  alias: {}\n",
        "models:\n  alias:\n    pricing:\n      input_per_million: 1\n      input_per_million: 2\n",
        "models: [fake_a]\nscaffolds: [chat_single]\nlimits:\n  max_tokens: 32\n  max_tokens: 64\n",
    ],
)
def test_duplicate_yaml_keys_rejected_at_all_depths(tmp_path, text):
    path = tmp_path / "duplicate.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        read_yaml(path)


def test_yaml_loader_rejects_object_construction(tmp_path):
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object:builtins.object {}", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        read_yaml(path)


def test_only_selected_model_environment_is_expanded(tmp_path, monkeypatch):
    monkeypatch.delenv("TRANSFERBENCH_UNUSED_MODEL", raising=False)
    path = yaml_file(
        tmp_path,
        {
            "models": {
                "fake_a": model_entry(),
                "unused": model_entry(
                    provider="openai",
                    model="${TRANSFERBENCH_UNUSED_MODEL}",
                    pricing={"input_per_million": 1, "output_per_million": 2},
                ),
            }
        },
    )
    registry = ModelRegistry(path)
    selected = registry.get("fake_a")
    assert selected.alias == selected.family == "fake_a"
    assert selected.simulated and selected.inspect_id == "fake/vulnerable"
    with pytest.raises(ValueError, match="TRANSFERBENCH_UNUSED_MODEL"):
        registry.get("unused")


@pytest.mark.parametrize("env_value", [None, ""])
def test_selected_model_requires_nonempty_environment(tmp_path, monkeypatch, env_value):
    name = "TRANSFERBENCH_SELECTED_VERSION"
    if env_value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, env_value)
    path = yaml_file(
        tmp_path, {"models": {"selected": model_entry(model="${TRANSFERBENCH_SELECTED_VERSION}")}}
    )
    with pytest.raises(ValueError, match=name):
        ModelRegistry(path).get("selected")


def test_environment_expansion_is_lazy_nonmutating_and_supports_nested_pricing(
    tmp_path, monkeypatch
):
    entry = model_entry(
        provider="openai",
        model="${TRANSFERBENCH_VERSION}",
        base_url="https://${TRANSFERBENCH_HOST}/v1",
        pricing={"input_per_million": "${TRANSFERBENCH_INPUT_PRICE}", "output_per_million": 2},
    )
    path = yaml_file(tmp_path, {"models": {"selected": entry}})
    registry = ModelRegistry(path)
    original = copy.deepcopy(registry.entries)
    monkeypatch.setenv("TRANSFERBENCH_VERSION", "version-1")
    monkeypatch.setenv("TRANSFERBENCH_HOST", "example.invalid")
    monkeypatch.setenv("TRANSFERBENCH_INPUT_PRICE", "1.25")
    first = registry.get("selected")
    assert first.model == "version-1" and first.base_url == "https://example.invalid/v1"
    assert first.pricing.input_per_million == 1.25
    monkeypatch.setenv("TRANSFERBENCH_VERSION", "version-2")
    assert registry.get("selected").model == "version-2"
    assert registry.entries == original


def test_resolution_uses_inspect_identifier_and_base_url_without_sdk_calls(monkeypatch):
    sentinel = object()
    calls = []

    def resolve(identifier, *, base_url):
        calls.append((identifier, base_url))
        return sentinel

    monkeypatch.setattr(registry_module, "get_model", resolve)
    spec = make_spec(
        provider="openai", model="pinned-test-v1", base_url="https://example.invalid/v1"
    )
    assert spec.resolve() is sentinel
    assert calls == [("openai/pinned-test-v1", "https://example.invalid/v1")]


# Experiment configuration --------------------------------------------------


def test_minimal_config_defaults_and_yaml_roundtrip(tmp_path):
    config = load_config(yaml_file(tmp_path, matrix_data()))
    assert config == MatrixConfig.model_validate(matrix_data())
    assert config.include_controls and config.seed == 42 and config.repeats == 1
    assert config.tasks.suites == ["workspace", "delegation"]
    assert config.limits.max_cost_usd == 0
    assert config.experiment.transfer_mode == "fixed"


@pytest.mark.parametrize("section", [None, "experiment", "tasks", "generation", "limits"])
def test_config_rejects_unknown_fields_at_every_level(tmp_path, section):
    data = matrix_data()
    if section is None:
        data["unknown_option"] = True
    else:
        data[section] = {"unknown_option": True}
    with pytest.raises(ValidationError, match="unknown_option"):
        load_config(yaml_file(tmp_path, data))


@pytest.mark.parametrize(
    "field,value",
    [
        ("models", "fake_a"),
        ("scaffolds", "chat_single"),
        ("attacks", "indirect_instruction"),
        ("defenses", "none"),
        ("attack_artifacts", "frozen.json"),
    ],
)
def test_config_dimensions_reject_duplicates(field, value):
    with pytest.raises(ValidationError, match="duplicates"):
        MatrixConfig.model_validate(matrix_data(**{field: [value, value]}))


@pytest.mark.parametrize("field", ["models", "scaffolds", "defenses"])
def test_config_required_dimensions_cannot_be_empty(field):
    with pytest.raises(ValidationError):
        MatrixConfig.model_validate(matrix_data(**{field: []}))


@pytest.mark.parametrize("field", ["scaffolds", "attacks", "defenses"])
def test_config_rejects_unknown_components(field):
    with pytest.raises(ValidationError, match="Unknown"):
        MatrixConfig.model_validate(matrix_data(**{field: ["not_registered"]}))


def test_equivalent_attack_aliases_cannot_duplicate_family():
    with pytest.raises(ValidationError, match="same family"):
        MatrixConfig.model_validate(matrix_data(attacks=["indirect", "indirect_instruction"]))


@pytest.mark.parametrize("defense", ["monitor", "second_model_monitor", "sanitizer+monitor"])
def test_monitor_defenses_require_explicit_alias(defense):
    with pytest.raises(ValidationError, match="requires monitor_model"):
        MatrixConfig.model_validate(matrix_data(defenses=[defense]))
    config = MatrixConfig.model_validate(
        matrix_data(defenses=[defense], monitor_model="monitor_alias")
    )
    assert config.monitor_model == "monitor_alias"


@pytest.mark.parametrize("missing", ["attack_artifacts", "selection_manifest"])
def test_source_optimized_requires_frozen_artifacts_and_manifest(missing):
    data = matrix_data(
        experiment={"transfer_mode": "source_optimized"},
        attack_artifacts=["frozen.json"],
        selection_manifest="selection.json",
    )
    data.pop(missing)
    with pytest.raises(ValidationError, match=missing):
        MatrixConfig.model_validate(data)


def test_valid_source_optimized_config():
    config = MatrixConfig.model_validate(
        matrix_data(
            experiment={"transfer_mode": "source_optimized", "stage": "confirmatory"},
            attack_artifacts=["frozen.json"],
            selection_manifest="selection.json",
        )
    )
    assert config.experiment.stage == "confirmatory"


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_turns", 0),
        ("max_turns", 201),
        ("max_tokens", 31),
        ("max_tokens", 128001),
        ("max_cost_usd", -1),
        ("max_cost_usd", math.inf),
        ("max_cost_usd", math.nan),
        ("timeout_seconds", 0),
        ("timeout_seconds", 3601),
        ("timeout_seconds", math.inf),
        ("max_episodes", 0),
    ],
)
def test_resource_limits_reject_out_of_range_values(field, value):
    with pytest.raises(ValidationError):
        Limits.model_validate({field: value})


@pytest.mark.parametrize("temperature", [-0.1, 2.1, math.inf, math.nan])
def test_generation_temperature_is_finite_and_bounded(temperature):
    with pytest.raises(ValidationError):
        Generation(temperature=temperature)


@pytest.mark.parametrize(
    "field,value", [("repeats", 0), ("repeats", 1001), ("seed", -1), ("seed", 2**31)]
)
def test_seed_and_repeat_bounds(field, value):
    with pytest.raises(ValidationError):
        MatrixConfig.model_validate(matrix_data(**{field: value}))


@pytest.mark.parametrize(
    "values", [{"name": "../escape"}, {"stage": "unknown"}, {"transfer_mode": "unknown"}]
)
def test_experiment_identity_and_modes_are_validated(values):
    with pytest.raises(ValidationError):
        Experiment.model_validate(values)


@pytest.mark.parametrize("values", [{"suites": []}, {"limit": 0}])
def test_task_selection_config_bounds(values):
    with pytest.raises(ValidationError):
        TaskConfig.model_validate(values)


def test_task_loading_resolves_paths_relative_to_config_and_enforces_nonempty(
    tmp_path, monkeypatch
):
    seen = []
    fixture = generate_synthetic_tasks()[0]

    def loader(suites, *, limit, paths):
        seen.append((suites, limit, paths))
        return [fixture]

    monkeypatch.setattr(config_module, "load_suite", loader)
    config = MatrixConfig.model_validate(
        matrix_data(tasks={"suites": ["workspace"], "paths": ["fixtures/tasks.jsonl"], "limit": 3})
    )
    assert config_tasks(config, tmp_path) == [fixture]
    assert seen == [(["workspace"], 3, [(tmp_path / "fixtures/tasks.jsonl").resolve()])]
    monkeypatch.setattr(config_module, "load_suite", lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match="empty"):
        config_tasks(config, tmp_path)


def test_builtin_suites_load_offline_and_unknown_suites_fail(tmp_path):
    config = MatrixConfig.model_validate(matrix_data(tasks={"limit": 2}))
    assert len(config_tasks(config, tmp_path)) == 2
    invalid = MatrixConfig.model_validate(matrix_data(tasks={"suites": ["unknown"]}))
    with pytest.raises(ValueError, match="Unknown task suites"):
        config_tasks(invalid, tmp_path)


def test_resolve_path_supports_absolute_relative_and_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_path(tmp_path / "config", "../result") == (tmp_path / "result").resolve()
    assert (
        resolve_path(tmp_path / "config", str(tmp_path / "absolute"))
        == (tmp_path / "absolute").resolve()
    )
    assert resolve_path(tmp_path / "config", "~/output") == (tmp_path / "output").resolve()


# Safety regression probes: production fixes belong to the owning agent. -----


@pytest.mark.parametrize("actual", [-1, math.nan, math.inf])
def test_invalid_settlement_cannot_release_a_live_reservation(actual):
    budget = CostBudget(1)
    reservation = budget.reserve(0.6)
    with pytest.raises(ValueError):
        budget.settle(reservation, actual)
    assert budget.charged + budget.reserved >= reservation


async def test_nonfinite_actual_estimate_cannot_erase_a_paid_reservation(messages):
    runtime, api = make_runtime(
        spec=make_spec(pricing={"input_per_million": 1e300, "output_per_million": 1}),
        budget=CostBudget(1e299),
        usage=ModelUsage(input_tokens=10**9, output_tokens=1, total_tokens=10**9 + 1),
    )
    bound = runtime.bound(runtime.spec)
    assert math.isfinite(bound)
    with pytest.raises((ValueError, UsageLimitExceeded)):
        await runtime.generate(messages)
    assert len(api.calls) == 1
    assert runtime.budget.charged + runtime.budget.reserved >= bound
    assert all(math.isfinite(cost.estimated_cost_usd) for cost in runtime.costs)


async def test_unsupported_seed_cannot_leak_from_inherited_model_config(messages):
    runtime, api = make_runtime(
        spec=make_spec(supports_seed=False), model_config=GenerateConfig(seed=999), seed=123
    )
    await runtime.generate(messages)
    assert api.calls[0]["config"].seed is None


async def test_single_response_reservation_disables_inherited_multiple_choices(messages):
    runtime, api = make_runtime(model_config=GenerateConfig(num_choices=3))
    await runtime.generate(messages)
    assert api.calls[0]["config"].num_choices in (None, 1)


async def test_paid_total_only_usage_cannot_be_treated_as_zero_cost(messages):
    runtime, _ = make_runtime(usage=ModelUsage(total_tokens=1000))
    with pytest.raises(UsageLimitExceeded):
        await runtime.generate(messages)
    assert runtime.budget.charged >= runtime.bound(runtime.spec)
    assert runtime.failed


async def test_reported_provider_cost_above_bound_fails_closed(messages):
    runtime, _ = make_runtime(
        usage=ModelUsage(input_tokens=10, output_tokens=4, total_tokens=14, total_cost=0.5)
    )
    with pytest.raises(UsageLimitExceeded):
        await runtime.generate(messages)
    assert runtime.budget.exhausted and runtime.failed
    assert runtime.budget.charged >= 0.5


def test_monitor_composition_whitespace_cannot_bypass_required_alias():
    with pytest.raises(ValidationError, match="requires monitor_model"):
        MatrixConfig.model_validate(matrix_data(defenses=["sanitizer + monitor"]))
