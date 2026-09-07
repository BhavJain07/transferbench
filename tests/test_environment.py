import copy
import json
from collections import Counter

import pytest

from transferbench.attacks import get_attack, prepare_task
from transferbench.environments.agentdojo import (
    SyntheticSnapshot,
    UpstreamProvenance,
    import_task_snapshot,
    load_task_snapshot,
    snapshot_sha256,
)
from transferbench.environments.synthetic import (
    TOOL_DEFINITIONS,
    WRITE_TOOLS,
    SyntheticWorkspace,
    authorization_error,
)
from transferbench.tasks.dataset import (
    DEFAULT_TASKS_PATH,
    generate_synthetic_tasks,
    load_suite,
    load_task,
    load_tasks,
)
from transferbench.tasks.schema import TaskSpec


@pytest.fixture
def task():
    return load_tasks()[0]


def test_exactly_24_reproducible_synthetic_fixtures():
    tasks = load_tasks()
    assert len(tasks) == 24
    assert tasks == generate_synthetic_tasks()
    assert tasks == load_tasks(DEFAULT_TASKS_PATH.parent)
    assert len({task.task_id for task in tasks}) == 24
    assert len({task.canaries["secret"] for task in tasks}) == 24
    assert Counter(task.family for task in tasks) == {
        "documents": 4,
        "email": 4,
        "calendar": 4,
        "payments": 4,
        "crm": 4,
        "delegation": 4,
    }
    for task in tasks:
        assert task.metadata["synthetic"] is True
        assert task.metadata["provenance"]["real_world_findings"] is False
        assert task.attack_surface.location in task.documents
        assert all(
            fact in task.documents[task.attack_surface.location] for fact in task.expected_facts
        )
        assert task.canaries["secret"].startswith("SYNTHETIC_SECRET_")
        assert task.canaries["secret"] not in task.user_goal
        assert task.canaries["destination"].endswith(".invalid")
        assert not set(task.allowed_tools) & set(task.forbidden_actions)


@pytest.mark.parametrize("task", load_tasks(), ids=lambda task: task.task_id)
def test_every_task_is_solvable_by_reading_actual_source(task):
    workspace = SyntheticWorkspace(prepare_task(task))
    search = workspace.execute("search_docs", {"query": task.attack_surface.location})
    assert search.allowed and search.executed
    assert task.attack_surface.location in search.result
    record = workspace.execute("read_doc", {"doc_id": task.attack_surface.location})
    assert record.allowed and record.executed
    assert all(fact in record.result for fact in task.expected_facts)
    assert not workspace.state["authorization_violations"]


@pytest.mark.parametrize(
    "family,tool,collection,argument",
    [
        ("email", "read_email", "emails", "email_id"),
        ("calendar", "read_calendar_event", "calendar", "event_id"),
        ("payments", "get_invoice", "invoices", "invoice_id"),
        ("crm", "get_customer", "crm", "customer_id"),
    ],
)
def test_typed_reads_dereference_injected_document(family, tool, collection, argument):
    for task in load_tasks(families=[family]):
        attack = get_attack("indirect", task)
        workspace = SyntheticWorkspace(prepare_task(task, attack))
        record_id = next(iter(workspace.state[collection]))
        result = workspace.execute(tool, {argument: record_id})
        assert result.allowed and result.executed
        content = json.loads(result.result)["content"]
        assert attack.payload in content
        assert all(fact in content for fact in task.expected_facts)


@pytest.mark.parametrize(
    "name,arguments,state_key",
    [
        (
            "send_email",
            {"to": "collector@attacker.invalid", "subject": "fixture", "body": "synthetic"},
            "sent_emails",
        ),
        (
            "create_calendar_event",
            {"title": "Fake event", "start": "2030-01-01T00:00:00Z"},
            "created_events",
        ),
        ("make_payment", {"payee": "fake-payee", "amount": 12.5}, "payments"),
        (
            "update_crm",
            {"customer_id": "customer-test", "field": "notes", "value": "changed"},
            "crm_updates",
        ),
        ("write_file", {"path": "/never/a/real/file", "content": "synthetic"}, "files"),
        ("delete_doc", {"doc_id": "documents/release"}, "deleted_documents"),
        ("forbidden_tool", {"payload": "synthetic"}, "forbidden_actions"),
    ],
)
def test_unauthorized_actions_execute_only_in_memory(task, name, arguments, state_key, monkeypatch):
    workspace = SyntheticWorkspace(prepare_task(task))
    workspace.state["crm"]["customer-test"] = {"notes": "before"}

    def no_io(*args, **kwargs):
        raise AssertionError("Synthetic tools must never perform filesystem or network IO")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", no_io)
        patch.setattr("pathlib.Path.open", no_io)
        patch.setattr("os.open", no_io)
        patch.setattr("socket.socket", no_io)
        record = workspace.execute(name, arguments, actor="worker_B")
    assert record.executed is True
    assert record.allowed is False
    assert record.actor == "worker_B"
    assert record.blocked_by is None
    assert workspace.state[state_key]
    assert workspace.state["authorization_violations"][-1]["name"] == name
    assert task.documents == load_tasks()[0].documents


def test_baseline_argument_authorization_is_observed_not_enforced(task):
    workspace = SyntheticWorkspace(prepare_task(task))
    record = workspace.execute("read_doc", {"doc_id": task.metadata["secret_document"]})
    assert not record.allowed
    assert record.executed
    assert task.canaries["secret"] in record.result


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("read_doc", {"doc_id": 1}),
        ("read_doc", {"doc_id": "documents/release", "extra": True}),
        ("read_doc", {}),
        ("send_email", {"to": ["a@example.invalid"], "subject": "x", "body": "x"}),
        ("make_payment", {"payee": "fake", "amount": -1}),
        ("make_payment", {"payee": "fake", "amount": True}),
        ("make_payment", {"payee": "fake", "amount": float("nan")}),
        ("make_payment", {"payee": "fake", "amount": "12"}),
        ("update_crm", {"customer_id": "x", "field": "__class__", "value": "x"}),
        ("run_shell", {"command": "echo forbidden"}),
    ],
)
def test_unknown_and_malformed_calls_do_not_execute(task, name, arguments):
    workspace = SyntheticWorkspace(prepare_task(task))
    before = copy.deepcopy(workspace.state)
    record = workspace.execute(name, arguments)
    assert not record.allowed and not record.executed
    assert "error" in json.loads(record.result)
    assert workspace.state == before


def test_workspace_and_returned_records_are_isolated(task):
    prepared = prepare_task(task)
    first, second = SyntheticWorkspace(prepared), SyntheticWorkspace(prepared)
    args = {"path": "virtual", "content": "one"}
    result = first.execute("write_file", args)
    args["content"] = "two"
    result.arguments["content"] = "three"
    assert first.state["files"]["virtual"] == "one"
    assert first.calls[0].arguments["content"] == "one"
    assert second.state["files"] == {}
    first.documents[task.attack_surface.location] = "changed"
    assert prepared.documents == task.documents
    assert second.documents == task.documents


def test_inventory_exposes_forbidden_tools_and_json_schemas(task):
    workspace = SyntheticWorkspace(prepare_task(task))
    assert set(WRITE_TOOLS).issubset(workspace.tool_names)
    for definition in TOOL_DEFINITIONS:
        assert definition.parameters["type"] == "object"
        assert definition.parameters["additionalProperties"] is False
    assert {definition["name"] for definition in workspace.tool_definitions} == set(
        workspace.tool_names
    )


def test_list_tools_and_missing_records():
    email_task = load_tasks(families=["email"])[0]
    email = SyntheticWorkspace(prepare_task(email_task))
    assert len(json.loads(email.execute("list_emails", {}).result)) == 1
    assert json.loads(email.execute("list_emails", {"query": "absent"}).result) == []
    calendar = SyntheticWorkspace(prepare_task(load_tasks(families=["calendar"])[0]))
    assert len(json.loads(calendar.execute("list_calendar", {"date": "2030-07-03"}).result)) == 1
    missing = calendar.execute("read_calendar_event", {"event_id": "missing"})
    assert missing.executed and "error" in json.loads(missing.result)


def test_dataset_filters_and_errors(tmp_path, task):
    assert len(load_tasks(families=["delegation"])) == 4
    assert len(load_tasks(families="email", limit=2)) == 2
    assert load_tasks(limit=0) == []
    assert load_task(task.task_id) == task
    with pytest.raises(KeyError):
        load_task("absent")
    with pytest.raises(ValueError, match="nonnegative"):
        load_tasks(limit=-1)
    with pytest.raises(FileNotFoundError):
        load_tasks(tmp_path / "missing.jsonl")
    with pytest.raises(ValueError, match="No JSONL"):
        load_tasks(tmp_path)
    path = tmp_path / "tasks.jsonl"
    path.write_text(task.model_dump_json() + "\n" + task.model_dump_json())
    with pytest.raises(ValueError, match="Duplicate task ID"):
        load_tasks(path)
    path.write_text("{not json}\n")
    with pytest.raises(ValueError, match=r"tasks.jsonl:1"):
        load_tasks(path)


def test_suite_selection_uses_one_dataset_and_filters_before_limit(monkeypatch):
    from transferbench.tasks import dataset

    all_tasks = load_tasks()
    workspace = load_suite()
    delegation = load_suite(suites=["delegation"])
    assert len(workspace) == 20
    assert len(delegation) == 4
    assert workspace == [task for task in all_tasks if task.family != "delegation"]
    assert delegation == load_tasks(families=["delegation"])
    assert load_suite(["delegation"], limit=2) == delegation[:2]
    assert load_suite(["workspace", "delegation"], limit=22) == all_tasks[:22]
    assert load_suite(["workspace"], limit=0) == []
    assert load_suite([]) == []
    calls = []

    def tracked_load():
        calls.append(True)
        return all_tasks

    monkeypatch.setattr(dataset, "load_tasks", tracked_load)
    assert load_suite(["workspace", "delegation", "workspace"]) == all_tasks
    assert len(calls) == 1


def test_suite_default_embedded_fallback_preserves_counts(monkeypatch, tmp_path):
    from transferbench.tasks import dataset

    monkeypatch.setattr(dataset, "DEFAULT_TASKS_PATH", tmp_path / "missing-default.jsonl")
    assert load_tasks() == generate_synthetic_tasks()
    assert len(load_suite()) == 20
    assert len(load_suite(["delegation"])) == 4
    assert load_suite(["workspace", "delegation"]) == generate_synthetic_tasks()
    with pytest.raises(FileNotFoundError):
        load_suite(paths=tmp_path / "explicit-missing.jsonl")


def test_suite_explicit_paths_and_validation(tmp_path):
    all_tasks = load_tasks()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text("\n".join(task.model_dump_json() for task in all_tasks[:12]))
    second.write_text("\n".join(task.model_dump_json() for task in all_tasks[12:]))
    assert load_suite(["workspace", "delegation"], paths=[first, second]) == all_tasks
    assert load_suite(["delegation"], paths=tmp_path) == all_tasks[20:]
    assert load_suite("workspace", paths=str(first)) == all_tasks[:12]
    assert load_suite(paths=[]) == []
    with pytest.raises(ValueError, match="Duplicate task ID"):
        load_suite(paths=[first, first])
    with pytest.raises(ValueError, match="Unknown task suites"):
        load_suite(["unknown"])
    with pytest.raises(ValueError, match="nonnegative"):
        load_suite(limit=-1)


@pytest.fixture
def snapshot_bundle(task):
    snapshot = SyntheticSnapshot(synthetic=True, documents=task.documents)
    provenance = UpstreamProvenance(
        version="test-fixture-only",
        revision="test-fixture-not-an-upstream-revision",
        suite="synthetic-boundary-test",
        task_id="synthetic-boundary-test-task",
        export_method="hand-authored adapter test, not an upstream export",
    )
    return task, snapshot, provenance


def test_explicit_snapshot_import_round_trip_without_upstream_dependency(snapshot_bundle, tmp_path):
    task, snapshot, provenance = snapshot_bundle
    before = task.model_copy(deep=True)
    digest = snapshot_sha256(snapshot)
    imported = import_task_snapshot(task, snapshot, provenance, expected_snapshot_sha256=digest)
    assert task == before
    assert imported.documents == task.documents
    recorded = imported.metadata["upstream_provenance"]
    assert recorded["snapshot_sha256"] == digest
    assert recorded["revision"] == provenance.revision
    assert recorded["upstream_scoring_supported"] is False
    assert recorded["execution"] == "synthetic_in_memory"
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps(
            {
                "task": task.model_dump(),
                "snapshot": snapshot.model_dump(),
                "provenance": provenance.model_dump(),
                "snapshot_sha256": digest,
            }
        )
    )
    assert load_task_snapshot(path) == imported
    workspace = SyntheticWorkspace(prepare_task(imported))
    assert (
        task.expected_facts[0]
        in workspace.execute("read_doc", {"doc_id": task.attack_surface.location}).result
    )


def test_snapshot_rejects_changed_content_or_missing_provenance(snapshot_bundle):
    task, snapshot, provenance = snapshot_bundle
    digest = snapshot_sha256(snapshot)
    modified = snapshot.model_copy(deep=True)
    modified.documents[task.attack_surface.location] += " changed"
    with pytest.raises(ValueError, match="SHA-256"):
        import_task_snapshot(task, modified, provenance, expected_snapshot_sha256=digest)
    with pytest.raises(ValueError, match="exactly match"):
        import_task_snapshot(
            task, modified, provenance, expected_snapshot_sha256=snapshot_sha256(modified)
        )
    with pytest.raises(ValueError):
        import_task_snapshot(task, snapshot, {}, expected_snapshot_sha256=digest)
    with pytest.raises(ValueError):
        SyntheticSnapshot.model_validate({"synthetic": False, "documents": task.documents})
    with pytest.raises(ValueError, match="document_id"):
        SyntheticSnapshot(
            synthetic=True, documents=task.documents, emails={"e": {"document_id": "absent"}}
        )
    with pytest.raises(ValueError):
        SyntheticSnapshot.model_validate(
            {"synthetic": True, "documents": task.documents, "execute": "arbitrary code"}
        )


def test_snapshot_requires_synthetic_task_and_explicit_tool_mapping(snapshot_bundle):
    task, snapshot, provenance = snapshot_bundle
    digest = snapshot_sha256(snapshot)
    live = task.model_copy(update={"metadata": {"synthetic": False}})
    with pytest.raises(ValueError, match="synthetic=true"):
        import_task_snapshot(live, snapshot, provenance, expected_snapshot_sha256=digest)
    unmapped = task.model_copy(update={"allowed_tools": ["upstream_unmapped_tool"]})
    with pytest.raises(ValueError, match="explicit mapping"):
        import_task_snapshot(unmapped, snapshot, provenance, expected_snapshot_sha256=digest)


def test_argument_policy_can_authorize_specific_synthetic_writes(task):
    mapped = TaskSpec.model_validate(
        {
            **task.model_dump(),
            "allowed_tools": ["send_email"],
            "forbidden_actions": [name for name in WRITE_TOOLS if name != "send_email"],
            "metadata": {"tool_policy": {"send_email": {"to": ["approved@example.invalid"]}}},
        }
    )
    good = {"to": "approved@example.invalid", "subject": "Test", "body": "Synthetic only"}
    assert authorization_error(mapped, "send_email", good) is None
    assert (
        authorization_error(mapped, "send_email", {**good, "to": "other@example.invalid"})
        is not None
    )
    assert SyntheticWorkspace(prepare_task(mapped)).execute("send_email", good).allowed
