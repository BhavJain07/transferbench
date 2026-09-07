"""Run attestation, pinned input snapshots and crash-safe derived storage."""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from transferbench.tasks.schema import Episode


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def git_state(root: Path) -> dict:
    def git(*args: str) -> str | None:
        proc = subprocess.run(
            ["git", "--no-pager", "--no-optional-locks", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": commit,
        "dirty": status != "",
        "git_status": status,
        "repository": git("rev-parse", "--show-toplevel"),
    }


def dependency_state(root: Path) -> dict:
    installed = {
        dist.metadata["Name"].lower().replace("_", "-"): dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    lock = root / "uv.lock"
    mismatches = []
    if lock.is_file():
        pinned = {}
        for package in tomllib.loads(lock.read_text()).get("package", []):
            pinned.setdefault(package["name"].lower().replace("_", "-"), set()).add(
                package["version"]
            )
        for name, version in installed.items():
            if name not in pinned or version not in pinned[name]:
                mismatches.append(f"{name}=={version} not represented in uv.lock")
    else:
        mismatches.append("uv.lock missing")
    return {
        "dependencies": installed,
        "lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "dependency_mismatches": mismatches,
        "dependencies_pinned": not mismatches,
        "python_version": platform.python_version(),
        "inspect_version": importlib.metadata.version("inspect-ai"),
    }


def publication_reasons(manifest: dict) -> list[str]:
    reasons = []
    if not manifest.get("commit"):
        reasons.append("No Git commit identifies the implementation")
    if manifest.get("dirty"):
        reasons.append("Repository contains uncommitted or untracked source changes")
    if not manifest.get("dependencies_pinned"):
        reasons.append("Installed dependencies do not match uv.lock")
    if any(model.get("provider") == "fake" for model in manifest.get("models", {}).values()):
        reasons.append("Simulated models are not real-provider evidence")
    if any(not model.get("version_pinned") for model in manifest.get("models", {}).values()):
        reasons.append("Model versions are not explicitly attested as immutable")
    origins = manifest.get("source_selections", [])
    if any(selection.get("source_simulated") for selection in origins):
        reasons.append("Attack selection used simulated source-model evidence")
    defense_origin = manifest.get("defense_selection") or {}
    if defense_origin.get("simulated"):
        reasons.append("Defense selection used simulated source-model evidence")
    if defense_origin and (
        defense_origin.get("source_dirty") or not defense_origin.get("git_commit")
    ):
        reasons.append("Defense selection source code was not clean and committed")
    if manifest.get("budget_uncertain"):
        reasons.append("At least one request has uncertain usage or exceeded its configured bound")
    if manifest.get("status") != "completed":
        reasons.append("Run did not complete its planned episodes")
    return reasons


_JSON_COLUMNS = {"tool_calls", "transcript", "call_costs", "propagation_events", "defense_events"}


def write_episodes(path: Path, episodes: list[Episode]) -> None:
    """JSON encode variable nested records; scalar metric columns stay typed."""
    rows = []
    for episode in episodes:
        row = episode.model_dump(mode="json")
        for key in _JSON_COLUMNS:
            row[key] = json.dumps(row[key], sort_keys=True, ensure_ascii=False, allow_nan=False)
        rows.append(row)
    if not rows:
        return
    # Explicit scalar types avoid a null first batch defining nullable identifiers
    # as an unusable Arrow null type. Each rewrite is atomic and self-contained.
    table = pa.Table.from_pylist(rows)
    temp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temp, compression="zstd")
    temp.replace(path)


def read_episodes(path: Path) -> list[Episode]:
    episodes = []
    for row in pq.read_table(path).to_pylist():
        for key in _JSON_COLUMNS:
            if isinstance(row.get(key), str):
                row[key] = json.loads(row[key])
        episodes.append(Episode.model_validate(row))
    return episodes


def verify_run(run_dir: Path) -> dict:
    """Verify recorded file integrity before publication or reproducibility review."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        return {"valid": False, "errors": ["Manifest must be a JSON object"], "publishable": False}
    errors = []
    hashes = manifest.get("file_hashes", {})
    if not isinstance(hashes, dict):
        errors.append("file_hashes must be a mapping of relative paths to SHA-256 digests")
        hashes = {}
    if not hashes:
        errors.append("No file integrity records")
    root = run_dir.resolve()
    for relative, expected in hashes.items():
        if not relative or "\0" in relative or Path(relative).is_absolute():
            errors.append(f"Unsafe provenance path: {relative!r}")
            continue
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
            errors.append(f"Invalid SHA-256 digest: {relative}")
            continue
        try:
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                errors.append(f"Unsafe provenance path: {relative}")
            elif not path.is_file() or sha256_file(path) != expected.lower():
                errors.append(f"Missing or changed: {relative}")
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"Cannot verify {relative!r}: {exc}")
    return {
        "valid": not errors,
        "errors": errors,
        "publishable": not errors and manifest.get("publishable") is True,
    }
