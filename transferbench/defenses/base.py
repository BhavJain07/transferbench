"""Scaffold-independent async defense hooks and explicit trust metadata."""

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field

from inspect_ai.model import ChatMessage, ModelOutput

from transferbench.tasks.schema import TaskSpec, ToolCallRecord

Monitor = Callable[[list[ChatMessage]], Awaitable[ModelOutput | None]]
UNTRUSTED_METADATA_KEY = "transferbench_untrusted"


@dataclass
class DefenseContext:
    task: TaskSpec
    messages: list[ChatMessage]
    actor: str = "assistant"
    events: list[dict] = field(default_factory=list)
    monitor: Monitor | None = None


def mark_untrusted(message: ChatMessage, source: str = "document") -> ChatMessage:
    """Return a copy marked as external data; never elevate its chat role."""
    marked = message.model_copy(deep=True)
    marked.metadata = {
        **(marked.metadata or {}),
        UNTRUSTED_METADATA_KEY: True,
        "transferbench_untrusted_source": source,
    }
    return marked


def is_untrusted(message: ChatMessage) -> bool:
    metadata = message.metadata or {}
    return (
        message.role == "tool"
        or metadata.get(UNTRUSTED_METADATA_KEY) is True
        or metadata.get("untrusted") is True
    )


def event(ctx: DefenseContext, defense: str, action: str, **details: object) -> None:
    ctx.events.append({"defense": defense, "action": action, "actor": ctx.actor, **details})


class Defense:
    name = "none"

    async def before_model(self, ctx: DefenseContext) -> DefenseContext:
        return ctx

    async def after_model(self, output: ModelOutput, ctx: DefenseContext) -> ModelOutput:
        return output

    async def before_tool(self, call: ToolCallRecord, ctx: DefenseContext) -> bool:
        return True

    async def after_tool(self, result: str, ctx: DefenseContext) -> str:
        return result


class DefenseChain(Defense):
    """Apply hooks in declared order; tool authorization short-circuits on deny."""

    def __init__(self, defenses: Iterable[Defense]):
        self.defenses = tuple(defenses)
        self.name = "+".join(defense.name for defense in self.defenses) or "none"

    async def before_model(self, ctx: DefenseContext) -> DefenseContext:
        for defense in self.defenses:
            ctx = await defense.before_model(ctx)
        return ctx

    async def after_model(self, output: ModelOutput, ctx: DefenseContext) -> ModelOutput:
        for defense in self.defenses:
            output = await defense.after_model(output, ctx)
        return output

    async def before_tool(self, call: ToolCallRecord, ctx: DefenseContext) -> bool:
        for defense in self.defenses:
            if not await defense.before_tool(call, ctx):
                call.blocked_by = call.blocked_by or defense.name
                return False
        return True

    async def after_tool(self, result: str, ctx: DefenseContext) -> str:
        for defense in self.defenses:
            result = await defense.after_tool(result, ctx)
        return result
