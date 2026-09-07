"""Explicit JSON import boundary; not an AgentDojo execution or scoring bridge."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from transferbench.environments.synthetic import TOOL_REGISTRY
from transferbench.tasks.schema import TaskSpec


class BoundaryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class UpstreamProvenance(BoundaryRecord):
    project: Literal["agentdojo"] = "agentdojo"
    repository: Literal["https://github.com/ethz-spylab/agentdojo"] = (
        "https://github.com/ethz-spylab/agentdojo"
    )
    version: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    export_method: str = Field(min_length=1)

    @model_validator(mode="after")
    def nonblank(self) -> "UpstreamProvenance":
        for field in ("version", "revision", "suite", "task_id", "export_method"):
            if not getattr(self, field).strip():
                raise ValueError(f"Provenance {field} cannot be blank")
        return self


class SyntheticSnapshot(BoundaryRecord):
    """Caller attests this is sanitized synthetic data, never live workspace data."""

    schema_version: Literal["1"] = "1"
    synthetic: Literal[True]
    documents: dict[str, str]
    emails: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    calendar: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    invoices: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    crm: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_references(self) -> "SyntheticSnapshot":
        for collection in (self.emails, self.calendar, self.invoices, self.crm):
            for record in collection.values():
                doc_id = record.get("document_id")
                if doc_id is not None and (
                    not isinstance(doc_id, str) or doc_id not in self.documents
                ):
                    raise ValueError("Snapshot document_id must reference a snapshot document")
        for record in self.calendar.values():
            if "start" in record and not isinstance(record["start"], str):
                raise ValueError("Calendar start must be text")
        return self


def snapshot_sha256(snapshot: SyntheticSnapshot) -> str:
    encoded = json.dumps(
        snapshot.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def import_task_snapshot(
    task: TaskSpec,
    snapshot: SyntheticSnapshot | dict,
    provenance: UpstreamProvenance | dict,
    *,
    expected_snapshot_sha256: str,
) -> TaskSpec:
    """Import an explicitly mapped TaskSpec with an integrity-checked snapshot.

    No upstream tools, Python objects, callbacks, or scoring functions are loaded.
    The provenance is supplied by the exporter, not independently authenticated.
    """
    snapshot = SyntheticSnapshot.model_validate(snapshot)
    provenance = UpstreamProvenance.model_validate(provenance)
    digest = snapshot_sha256(snapshot)
    if expected_snapshot_sha256 != digest:
        raise ValueError("Snapshot SHA-256 mismatch")
    if task.metadata.get("synthetic") is not True:
        raise ValueError("Imported task must explicitly declare metadata.synthetic=true")
    if task.documents != snapshot.documents:
        raise ValueError("Task documents must exactly match the imported snapshot")
    unknown = (set(task.allowed_tools) | set(task.forbidden_actions)) - TOOL_REGISTRY.keys()
    if unknown:
        raise ValueError(f"Unsupported upstream tools require explicit mapping: {sorted(unknown)}")
    metadata = {
        **task.metadata,
        "workspace": snapshot.model_dump(
            mode="json", exclude={"schema_version", "synthetic", "documents"}
        ),
        "upstream_provenance": {
            **provenance.model_dump(mode="json"),
            "snapshot_sha256": digest,
            "adapter": "transferbench-explicit-snapshot-v1",
            "execution": "synthetic_in_memory",
            "scoring": "transferbench_deterministic_assertions",
            "upstream_scoring_supported": False,
        },
    }
    return TaskSpec.model_validate({**task.model_dump(mode="json"), "metadata": metadata})


class SnapshotImport(BoundaryRecord):
    task: TaskSpec
    snapshot: SyntheticSnapshot
    provenance: UpstreamProvenance
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_task_snapshot(path: str | Path) -> TaskSpec:
    """Trusted host-side JSON loader, deliberately not a model-visible tool."""
    bundle = SnapshotImport.model_validate_json(Path(path).read_text(encoding="utf-8"))
    return import_task_snapshot(
        bundle.task,
        bundle.snapshot,
        bundle.provenance,
        expected_snapshot_sha256=bundle.snapshot_sha256,
    )
