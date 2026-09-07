"""Validate scalar episode provenance against the run's frozen manifest records.

Manifest contract: ``attack_artifacts`` (alias ``artifacts``) and
``source_selections`` (alias ``selections``) are lists, or ID-keyed dictionaries.
Artifacts use Attack fields (id, artifact_sha256, source_model, source_scaffold,
selection_id, optional task_id). Selection records use selection_id,
attack_artifact_sha256 (or artifact_sha256), source_model, source_scaffold,
and split='selection'/'screening'. They must not have selected=False.
Publication additionally requires the runner's file_hashes for episodes.parquet,
inputs/{config,tasks,models,source_selections}.json and every frozen attack file.
verify_integrity lazily calls runner.manifest.verify_run and compares hashed
snapshots with manifest records. It hashes actual files, but does not decode
transcripts or independently authenticate the manifest/provider. Model IDs are
provider/model, not bare manifest model names. Ambiguity fails closed.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _records(value: Any, id_field: str) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [{id_field: key, **item} for key, item in value.items() if isinstance(item, dict)]
    return []


def verify_integrity(run_dir: Path, manifest: dict) -> dict:
    """Lazily use runner verification, require its input coverage, and bind snapshots.

    Hashes detect changes relative to the manifest, not a dishonest/replaced
    manifest. Do not import the runner during discovery's provenance-only checks.
    """
    errors = []
    hashes = manifest.get("file_hashes")
    checked = "file_hashes" in manifest
    if checked:
        try:
            from transferbench.runner.manifest import verify_run

            verification = verify_run(run_dir)
            if verification.get("valid") is not True:
                errors.extend(verification.get("errors") or ["runner verification failed"])
        except (OSError, ValueError, TypeError, AttributeError, ImportError) as exc:
            errors.append(f"run verification failed: {exc}")
    if not isinstance(hashes, dict) or not hashes:
        errors.append("file_hashes are required for publication")
        return {"checked": checked, "valid": False, "errors": errors}
    required = {"episodes.parquet", "inputs/tasks.json"}
    snapshots = {
        "inputs/config.json": manifest.get("resolved_config"),
        "inputs/models.json": manifest.get("models"),
        "inputs/source_selections.json": manifest.get("source_selections", []),
    }
    for artifact in _records(manifest.get("attack_artifacts", manifest.get("artifacts")), "id"):
        digest = artifact.get("artifact_sha256")
        if isinstance(digest, str) and SHA256.fullmatch(digest):
            snapshots[f"inputs/attacks/{digest}.json"] = artifact
    if manifest.get("defense_selection") is not None:
        snapshots["inputs/defense_selection.json"] = manifest["defense_selection"]
    required.update(snapshots)
    for relative in sorted(required):
        digest = hashes.get(relative)
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            errors.append(f"Missing file hash: {relative}")
    if not errors:
        for relative, expected in snapshots.items():
            try:
                actual = json.loads((run_dir / relative).read_text(encoding="utf-8"))
                if actual != expected:
                    errors.append(f"Manifest disagrees with hashed snapshot: {relative}")
            except (OSError, ValueError) as exc:
                errors.append(f"Invalid provenance snapshot {relative}: {exc}")
    if not errors and manifest.get("defense_selection") is not None:
        try:
            from transferbench.runner.defense_selection import load_defense_selection

            load_defense_selection(run_dir / "inputs/defense_selection.json")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"Invalid source-defense evidence: {exc}")
    return {"checked": checked, "valid": not errors, "errors": errors}


def publication_metadata_issues(manifest: dict) -> list[str]:
    """Check recorded runner attestations; never substitute current machine state."""
    issues = []
    blockers = manifest.get("publication_blockers")
    if not isinstance(blockers, list):
        issues.append("missing publication_blockers attestation")
    elif blockers:
        issues.extend(f"runner publication blocker: {reason}" for reason in blockers)
    if not manifest.get("commit") or manifest.get("dirty") is not False:
        issues.append("missing commit or clean-repository attestation")
    lock = manifest.get("lock_sha256")
    if (
        manifest.get("dependencies_pinned") is not True
        or not isinstance(manifest.get("dependencies"), dict)
        or not manifest["dependencies"]
        or not isinstance(lock, str)
        or not SHA256.fullmatch(lock)
    ):
        issues.append("missing pinned dependency/lock provenance")
    if manifest.get("status") != "completed" or manifest.get("budget_uncertain"):
        issues.append("run incomplete or usage/budget uncertain")
    models = manifest.get("models")
    if (
        not isinstance(models, dict)
        or not models
        or any(
            not isinstance(spec, dict)
            or spec.get("version_pinned") is not True
            or any(
                not isinstance(spec.get(key), str) or not spec[key]
                for key in ("provider", "model", "family")
            )
            for spec in models.values()
        )
    ):
        issues.append("missing immutable model alias/provider/model/family provenance")
    return issues


class Provenance:
    def __init__(self, manifest: dict, *, integrity_failed: bool = False):
        self.integrity_failed = integrity_failed
        models = manifest.get("models")
        self.models: dict[str, Any] = models if isinstance(models, dict) else {}
        self.artifacts: dict[str, list[dict]] = defaultdict(list)
        self.selections: dict[str, list[dict]] = defaultdict(list)
        for artifact in _records(manifest.get("attack_artifacts", manifest.get("artifacts")), "id"):
            digest = artifact.get("artifact_sha256", artifact.get("sha256"))
            if isinstance(digest, str) and SHA256.fullmatch(digest):
                self.artifacts[digest].append(artifact)
        for selection in _records(
            manifest.get("source_selections", manifest.get("selections")), "selection_id"
        ):
            selection_id = selection.get("selection_id", selection.get("id"))
            if isinstance(selection_id, str) and selection_id:
                self.selections[selection_id].append(selection)

    def check(self, row: dict) -> list[str]:
        problems = []
        if self.models and isinstance(row.get("model_alias"), str):
            spec = self.models.get(row["model_alias"])
            if (
                not isinstance(spec, dict)
                or row.get("model_id") != f"{spec.get('provider')}/{spec.get('model')}"
                or row.get("model_family") != spec.get("family")
            ):
                problems.append("episode model alias/provider/model/family does not match manifest")
        if not row.get("attack_id"):
            return problems
        if self.integrity_failed:
            problems.append("run file integrity failed")
        if not isinstance(row["attack_id"], str):
            return [*problems, "invalid attack_id provenance"]
        digest = row.get("attack_artifact_sha256")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            return [*problems, "missing or invalid attack_artifact_sha256"]
        artifacts = self.artifacts.get(digest, [])
        if len(artifacts) != 1:
            problems.append("frozen artifact missing or ambiguous in manifest")
        else:
            artifact = artifacts[0]
            fields = {
                "id": "attack_id",
                "source_model": "attack_source_model",
                "source_scaffold": "attack_source_scaffold",
                "selection_id": "selection_id",
            }
            for field, episode_field in fields.items():
                if artifact.get(field) != row.get(episode_field):
                    problems.append(f"artifact {field} does not match episode")
            if (
                "family" in artifact
                and "attack_family" in row
                and artifact["family"] != row["attack_family"]
            ):
                problems.append("artifact family does not match episode")
            if artifact.get("task_id") not in (None, row.get("task_id")):
                problems.append("artifact task_id does not match episode")
            if artifact.get("frozen") is False:
                problems.append("artifact explicitly not frozen")
        if row.get("transfer_mode") == "source_optimized":
            if not row.get("attack_source_model") or not row.get("attack_source_scaffold"):
                problems.append("source-optimized attack lacks source model/scaffold")
            selection_id = row.get("selection_id")
            selections = (
                self.selections.get(selection_id, [])
                if isinstance(selection_id, str) and selection_id
                else []
            )
            if len(selections) != 1:
                problems.append("source selection missing or ambiguous in manifest")
            else:
                selection = selections[0]
                selected_hash = selection.get(
                    "attack_artifact_sha256", selection.get("artifact_sha256")
                )
                if selected_hash != digest:
                    problems.append("source selection artifact does not match episode")
                for field in ("source_model", "source_scaffold"):
                    if selection.get(field) != row.get(f"attack_{field}"):
                        problems.append(f"selection {field} does not match episode")
                if selection.get("split") not in ("screening", "selection"):
                    problems.append("selection must identify a non-evaluation split")
                if selection.get("selected") is False:
                    problems.append("artifact was not selected")
                source_model = selection.get("source_model")
                source_spec = (
                    self.models.get(source_model) if isinstance(source_model, str) else None
                )
                if (
                    selection.get("source_model_id")
                    and isinstance(source_spec, dict)
                    and selection["source_model_id"]
                    != f"{source_spec.get('provider')}/{source_spec.get('model')}"
                ):
                    problems.append(
                        "selected source model version does not match evaluation manifest"
                    )
        return problems
