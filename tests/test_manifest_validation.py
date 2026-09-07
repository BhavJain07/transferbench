"""Malformed evidence records must fail closed, not crash integrity inspection."""

import json

import pytest

from transferbench.runner.manifest import sha256_file, verify_run


@pytest.mark.parametrize("manifest", [None, [], "not a manifest", 42])
def test_non_object_manifest_returns_invalid(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = verify_run(tmp_path)
    assert result == {
        "valid": False,
        "errors": ["Manifest must be a JSON object"],
        "publishable": False,
    }


@pytest.mark.parametrize("hashes", [None, [], "digest", 42, {}])
def test_missing_or_non_mapping_hashes_cannot_publish(tmp_path, hashes):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"publishable": True, "file_hashes": hashes})
    )
    result = verify_run(tmp_path)
    assert not result["valid"] and not result["publishable"]
    assert "No file integrity records" in result["errors"]


@pytest.mark.parametrize("digest", [None, 7, [], "", "abc", "g" * 64])
def test_malformed_digest_is_reported(tmp_path, digest):
    (tmp_path / "payload.txt").write_text("synthetic evidence")
    (tmp_path / "manifest.json").write_text(json.dumps({"file_hashes": {"payload.txt": digest}}))
    assert verify_run(tmp_path)["errors"] == ["Invalid SHA-256 digest: payload.txt"]


@pytest.mark.parametrize("relative", ["", "../outside.txt", "invalid\0name"])
def test_unsafe_path_is_rejected(tmp_path, relative):
    (tmp_path / "manifest.json").write_text(json.dumps({"file_hashes": {relative: "0" * 64}}))
    result = verify_run(tmp_path)
    assert not result["valid"]
    assert "Unsafe provenance path" in result["errors"][0]


def test_absolute_path_is_not_a_portable_integrity_record(tmp_path):
    payload = tmp_path / "payload.txt"
    payload.write_text("synthetic evidence")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "file_hashes": {str(payload): sha256_file(payload)},
            }
        )
    )
    assert not verify_run(tmp_path)["valid"]


def test_uppercase_hex_digest_is_accepted(tmp_path):
    payload = tmp_path / "payload.txt"
    payload.write_text("synthetic evidence")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "publishable": False,
                "file_hashes": {"payload.txt": sha256_file(payload).upper()},
            }
        )
    )
    assert verify_run(tmp_path) == {"valid": True, "errors": [], "publishable": False}


def test_symlink_loop_is_reported_without_crashing(tmp_path):
    (tmp_path / "loop").symlink_to("loop")
    (tmp_path / "manifest.json").write_text(json.dumps({"file_hashes": {"loop": "0" * 64}}))
    result = verify_run(tmp_path)
    assert not result["valid"] and result["errors"]
