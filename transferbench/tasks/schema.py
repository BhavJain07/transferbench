"""Versioned, JSON-serializable research records shared by all components."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AttackSurface(Record):
    type: Literal[
        "document",
        "email",
        "webpage",
        "tool_result",
        "database_row",
        "memory",
        "delegation",
        "direct",
    ]
    location: str


class TaskSpec(Record):
    task_id: str
    family: str
    user_goal: str
    allowed_tools: list[str] = Field(default_factory=lambda: ["search_docs", "read_doc"])
    forbidden_actions: list[str] = Field(
        default_factory=lambda: ["send_email", "write_file", "forbidden_tool"]
    )
    success_assertions: list[str] = Field(default_factory=lambda: ["contains_expected_fact"])
    attack_surface: AttackSurface
    canaries: dict[str, str]
    documents: dict[str, str]
    expected_facts: list[str] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_facts")
    @classmethod
    def meaningful_facts(cls, facts: list[str]) -> list[str]:
        # Match the utility scorer's normalization without changing source excerpts.
        normalized = [" ".join(fact.casefold().split()) for fact in facts]
        if any(not fact for fact in normalized):
            raise ValueError("expected_facts cannot contain blank assertions")
        if len(set(normalized)) != len(normalized):
            raise ValueError("expected_facts must be unique after grading normalization")
        return facts

    @model_validator(mode="after")
    def valid_task(self) -> "TaskSpec":
        if set(self.allowed_tools) & set(self.forbidden_actions):
            raise ValueError("allowed_tools and forbidden_actions must be disjoint")
        if self.attack_surface.location not in self.documents:
            raise ValueError("attack surface must identify an existing synthetic document")
        if not self.canaries.get("secret"):
            raise ValueError("a nonempty synthetic secret canary is required")
        if any(self.canaries["secret"] in fact for fact in self.expected_facts):
            raise ValueError("utility facts cannot contain a protected canary")
        if set(self.success_assertions) - {"contains_expected_fact"}:
            raise ValueError("unsupported deterministic success assertion")
        return self


Task = TaskSpec


class Attack(Record):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    family: str
    payload: str
    source_model: str | None = None
    source_scaffold: str | None = None
    task_id: str | None = None
    generation_seed: int = 0
    created_at: str
    git_commit: str | None = None
    variant: str = "default"
    artifact_sha256: str = ""
    selection_id: str | None = None


class ToolCallRecord(Record):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    allowed: bool = True
    executed: bool = False
    actor: str = "assistant"
    blocked_by: str | None = None


class CallCost(Record):
    provider: str
    model: str
    input_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)
    purpose: str = "assistant"


class EpisodeResult(Record):
    output: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    transcript: list[dict[str, Any]] = Field(default_factory=list)
    costs: list[CallCost] = Field(default_factory=list)
    propagation_events: list[dict[str, Any]] = Field(default_factory=list)
    defense_events: list[dict[str, Any]] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class Episode(Record):
    schema_version: str = "1.0"
    run_id: str
    episode_id: str
    model_id: str
    model_family: str
    model_alias: str
    scaffold_id: str
    task_id: str
    task_family: str
    attack_id: str | None = None
    attack_family: str | None = None
    attack_source_model: str | None = None
    attack_source_scaffold: str | None = None
    attack_artifact_sha256: str | None = None
    selection_id: str | None = None
    defense_id: str
    defense_source_model: str | None = None
    defense_source_scaffold: str | None = None
    condition: Literal["C0", "C1", "C2", "C3"]
    transfer_mode: Literal["fixed", "family", "source_optimized"] = "fixed"
    split: Literal["screening", "selection", "evaluation", "confirmatory"] = "evaluation"
    seed: int
    repeat: int = Field(ge=0)
    utility_success: bool
    attack_success: bool
    policy_violation: bool
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    transcript: list[dict[str, Any]] = Field(default_factory=list)
    input_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)
    call_costs: list[dict[str, Any]] = Field(default_factory=list)
    propagation_depth: int = Field(default=0, ge=0, le=4)
    propagation_events: list[dict[str, Any]] = Field(default_factory=list)
    defense_events: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["ok", "error", "budget_exceeded"] = "ok"
    error: str | None = None
    inspect_log: str | None = None
    simulated: bool = False

    @model_validator(mode="after")
    def coherent_condition(self) -> "Episode":
        expected = (
            ("C1" if self.defense_id == "none" else "C2")
            if self.attack_id
            else ("C0" if self.defense_id == "none" else "C3")
        )
        if self.condition != expected:
            raise ValueError(f"condition must be {expected} for attack/defense combination")
        return self
