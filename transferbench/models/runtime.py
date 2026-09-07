"""One audited generation boundary around Inspect, not a provider wrapper.

The hard ceiling is on configured conservative estimates. Provider billing cannot
be guaranteed by a client: prices, hidden tokens and billing on failed requests
are external. Set a provider-side spending cap for an absolute invoice ceiling.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from math import ceil

from inspect_ai.model import ChatMessage, GenerateConfig, Model, ModelOutput
from inspect_ai.tool import Tool, ToolDef

from transferbench.models.registry import ModelSpec, finite_cost
from transferbench.tasks.schema import CallCost


class BudgetExceeded(RuntimeError):
    pass


class UsageLimitExceeded(RuntimeError):
    pass


@dataclass
class CostBudget:
    maximum: float
    charged: float = 0.0
    reserved: float = 0.0
    exhausted: bool = False
    uncertain: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        finite_cost(self.maximum)

    def reserve(self, amount: float) -> float:
        finite_cost(amount)
        with self._lock:
            if self.exhausted or self.charged + self.reserved + amount > self.maximum + 1e-12:
                self.exhausted = True
                raise BudgetExceeded(
                    f"Ceiling ${self.maximum:.6f}; charged ${self.charged:.6f}, reserved ${self.reserved:.6f}, next-call bound ${amount:.6f}"
                )
            self.reserved += amount
        return amount

    def settle(self, reservation: float, actual: float | None) -> None:
        finite_cost(reservation)
        if actual is not None:
            finite_cost(actual)
        with self._lock:
            self.reserved = max(0.0, self.reserved - reservation)
            if actual is None:
                self.charged += reservation
                self.uncertain = True
            else:
                finite_cost(actual)
                self.charged += actual
                if actual > reservation + 1e-12:
                    self.exhausted = True
                    self.uncertain = True
                    raise UsageLimitExceeded(
                        "Provider usage exceeded reserved bound; stopped. Check provider billing and pricing configuration."
                    )
            if self.charged > self.maximum + 1e-12:
                self.exhausted = True


class GenerationRuntime:
    def __init__(
        self,
        model: Model,
        spec: ModelSpec,
        budget: CostBudget,
        *,
        seed: int = 0,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout_seconds: float = 120,
        monitor_model: Model | None = None,
        monitor_spec: ModelSpec | None = None,
    ):
        self.model, self.spec, self.budget = model, spec, budget
        # The runtime owns generation configuration. Inherited best_of, hidden
        # reasoning, fallback models, or extra request bodies bypass reservations.
        model.config = GenerateConfig()
        if monitor_model is not None:
            monitor_model.config = GenerateConfig()
        self.seed, self.temperature, self.max_tokens = seed, temperature, max_tokens
        self.timeout_seconds = timeout_seconds
        self.monitor_model, self.monitor_spec = monitor_model, monitor_spec
        self.costs: list[CallCost] = []
        self.failed = False

    def bound(self, spec: ModelSpec) -> float:
        return spec.pricing.estimate(spec.max_input_tokens, self.max_tokens)

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: list[Tool] | None = None,
        purpose: str = "assistant",
    ) -> ModelOutput:
        return await self._generate(self.model, self.spec, messages, tools, purpose)

    async def monitor(self, messages: list[ChatMessage]) -> ModelOutput | None:
        if self.monitor_model is None or self.monitor_spec is None:
            raise ValueError("monitor defense requires an explicit monitor_model alias")
        return await self._generate(
            self.monitor_model, self.monitor_spec, messages, None, "monitor"
        )

    async def _generate(
        self,
        model: Model,
        spec: ModelSpec,
        messages: list[ChatMessage],
        tools: list[Tool] | None,
        purpose: str,
    ) -> ModelOutput:
        if self.failed:
            raise UsageLimitExceeded(
                "An earlier model/monitor call failed; refusing further calls in this episode"
            )
        tools = tools or []
        payload = {
            "messages": [message.model_dump(mode="json") for message in messages],
            "tools": [
                {
                    "name": ToolDef(tool).name,
                    "description": ToolDef(tool).description,
                    "parameters": ToolDef(tool).parameters.model_dump(mode="json"),
                }
                for tool in tools
            ],
        }
        # Byte count is deliberately pessimistic for text tokenization; a generous
        # framing allowance covers provider chat templates. Multimodal is not enabled.
        upper_input = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 8192
        if not spec.simulated and upper_input > spec.max_input_tokens:
            raise UsageLimitExceeded(
                f"Serialized input bound {upper_input} exceeds configured {spec.max_input_tokens}; no request sent"
            )
        reservation = self.budget.reserve(self.bound(spec))
        started = time.perf_counter()
        settled = False
        try:
            config = GenerateConfig(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                max_retries=0,
                timeout=ceil(self.timeout_seconds),
                seed=self.seed if spec.supports_seed else None,
                parallel_tool_calls=False,
                num_choices=1,
                cache=False,
                cache_prompt=False,
            )
            output = await model.generate(messages, tools=tools, config=config, cache=False)
            usage = output.usage
            if usage is None and not spec.simulated:
                raise UsageLimitExceeded(
                    "Provider omitted token usage; entire reservation charged and episode stopped"
                )
            cached = (usage.input_tokens_cache_read or 0) if usage else 0
            cache_write = (usage.input_tokens_cache_write or 0) if usage else 0
            # Cache tokens may already be included by some providers: counting them
            # again is conservative. Configured input price must cover cache writes.
            inputs = (usage.input_tokens + cached + cache_write) if usage else 0
            outputs = usage.output_tokens if usage else 0
            if usage is not None and not spec.simulated:
                counters = [usage.input_tokens, outputs, cached, cache_write, usage.total_tokens]
                if (
                    any(value < 0 for value in counters)
                    or inputs + outputs == 0
                    or usage.total_tokens > inputs + outputs
                ):
                    raise UsageLimitExceeded(
                        "Provider supplied incomplete or inconsistent token usage; full bound charged"
                    )
            cost = finite_cost(spec.pricing.estimate(inputs, outputs))
            if usage is not None and usage.total_cost is not None and not spec.simulated:
                cost = max(cost, finite_cost(usage.total_cost))
            self.costs.append(
                CallCost(
                    provider=spec.provider,
                    model=spec.model,
                    input_tokens=inputs,
                    cached_tokens=cached,
                    output_tokens=outputs,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    estimated_cost_usd=cost,
                    purpose=purpose,
                )
            )
            settled = True
            self.budget.settle(reservation, cost)
            if not spec.simulated and (inputs > spec.max_input_tokens or outputs > self.max_tokens):
                self.budget.exhausted = True
                raise UsageLimitExceeded("Provider token usage exceeded configured bounds; stopped")
            return output
        except BaseException:
            self.failed = True
            if not settled:
                self.budget.settle(reservation, None)
                self.costs.append(
                    CallCost(
                        provider=spec.provider,
                        model=spec.model,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        estimated_cost_usd=reservation,
                        purpose=purpose + ":unknown_usage_reserved",
                    )
                )
            raise
