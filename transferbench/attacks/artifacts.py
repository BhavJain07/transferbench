"""Content-addressed, create-only JSON attack artifacts (not environment tools)."""

import hashlib
import json
import os
from pathlib import Path

from transferbench.tasks.schema import Attack


def attack_sha256(attack: Attack) -> str:
    data = attack.model_dump(mode="json", exclude={"artifact_sha256"})
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def seal_attack(attack: Attack) -> Attack:
    digest = attack_sha256(attack)
    if attack.artifact_sha256 and attack.artifact_sha256 != digest:
        raise ValueError("Attack artifact SHA-256 mismatch")
    return attack.model_copy(update={"artifact_sha256": digest})


def save_attack(attack: Attack, path: str | Path) -> Attack:
    """Create a new artifact; never overwrite, including an existing symlink.

    Parents must already exist. O_EXCL arbitrates competing writers. The digest
    detects modification, not authorship: this is not a cryptographic signature.
    """
    sealed = seal_attack(attack)
    data = (sealed.model_dump_json(indent=2) + "\n").encode("utf-8")
    fd = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return sealed


def load_attack(path: str | Path) -> Attack:
    attack = Attack.model_validate_json(Path(path).read_text(encoding="utf-8"))
    if not attack.artifact_sha256:
        raise ValueError("Attack artifact is missing SHA-256")
    return seal_attack(attack)


save_attack_artifact = save_attack
load_attack_artifact = load_attack
