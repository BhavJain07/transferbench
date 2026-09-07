"""Synthetic unit fixtures only. No fixture is an empirical model result."""

import hashlib
import json
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from transferbench.analysis import analyze_run, generate_report
from transferbench.analysis.statistics import (
    cluster_bootstrap,
    macro_rate_estimate,
    paired_estimate,
    wilson_interval,
)
from transferbench.scorers.transfer import (
    attack_success_rate,
    cross_transfer_score,
    raw_transfer,
    relative_risk_reduction,
    transfer_ratio,
    utility_adjusted_defense_score,
)
from transferbench.tasks.schema import Episode

IDS = count()


def episode(
    *,
    task="t1",
    model="source",
    scaffold="direct",
    attack: str | None = "attack-a",
    defense="none",
    success=True,
    utility=True,
    mode="fixed",
    **changes: Any,
) -> dict[str, Any]:
    """Create synthetic episode JSON; callers may mutate fields to test malformed input."""
    values: dict[str, Any] = dict(
        run_id="synthetic-unit-test",
        episode_id=f"fixture-{next(IDS)}",
        model_id=f"unit/{model}-v1",
        model_family="fixture-family",
        model_alias=model,
        scaffold_id=scaffold,
        task_id=task,
        task_family="fixture-task",
        attack_id=attack,
        attack_family="fixture-attack" if attack else None,
        attack_artifact_sha256=hashlib.sha256(attack.encode()).hexdigest() if attack else None,
        defense_id=defense,
        condition=("C1" if defense == "none" else "C2")
        if attack
        else ("C0" if defense == "none" else "C3"),
        transfer_mode=mode,
        seed=11,
        repeat=0,
        utility_success=utility,
        attack_success=success,
        policy_violation=success,
        simulated=True,
        estimated_cost_usd=0.01,
        input_tokens=10,
        output_tokens=5,
        latency_ms=2,
    )
    values.update(changes)
    return Episode(**values).model_dump()


def write_run(
    path: Path, rows: list[dict[str, Any]], *, with_integrity=False, **overrides: Any
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    selections = {}
    for row in rows:
        if row.get("attack_id") and row.get("attack_artifact_sha256"):
            artifacts[row["attack_artifact_sha256"]] = {
                "id": row["attack_id"],
                "family": row.get("attack_family"),
                "artifact_sha256": row["attack_artifact_sha256"],
                "source_model": row.get("attack_source_model"),
                "source_scaffold": row.get("attack_source_scaffold"),
                "selection_id": row.get("selection_id"),
                "frozen": True,
            }
            if row.get("selection_id"):
                selections[row["selection_id"]] = {
                    "selection_id": row["selection_id"],
                    "attack_artifact_sha256": row["attack_artifact_sha256"],
                    "source_model": row.get("attack_source_model"),
                    "source_scaffold": row.get("attack_source_scaffold"),
                    "split": "selection",
                    "selected": True,
                }
    manifest: dict[str, Any] = dict(
        run_id="synthetic-unit-test",
        publishable=False,
        attack_artifacts=list(artifacts.values()),
        source_selections=list(selections.values()),
    )
    manifest.update(overrides)
    (path / "manifest.json").write_text(json.dumps(manifest))
    scalar_and_json = []
    for row in rows:
        row = dict(row)
        for field in (
            "tool_calls",
            "transcript",
            "call_costs",
            "propagation_events",
            "defense_events",
        ):
            if field in row:
                row[field] = json.dumps(row[field])
        scalar_and_json.append(row)
    frame = (
        pl.DataFrame(scalar_and_json)
        if scalar_and_json
        else pl.DataFrame(schema={"episode_id": pl.String})
    )
    frame.write_parquet(path / "episodes.parquet")
    if with_integrity:
        # Synthetic attestations solely to exercise the gate, not real evidence.
        models = {}
        for row in rows:
            if isinstance(row.get("model_id"), str) and "/" in row["model_id"]:
                provider, model = row["model_id"].split("/", 1)
                models[row["model_alias"]] = dict(
                    provider=provider,
                    model=model,
                    family=row["model_family"],
                    pricing={"input_per_million": 1, "output_per_million": 1},
                    version_pinned=True,
                )
        for key, value in dict(
            models=models,
            resolved_config={"seed": 11},
            commit="0" * 40,
            dirty=False,
            dependencies={"synthetic-test-fixture": "0"},
            dependencies_pinned=True,
            lock_sha256="a" * 64,
            status="completed",
            publication_blockers=[],
        ).items():
            manifest.setdefault(key, value)
        snapshots = {
            "inputs/config.json": manifest["resolved_config"],
            "inputs/models.json": manifest["models"],
            "inputs/tasks.json": sorted({r["task_id"] for r in rows}),
            "inputs/source_selections.json": manifest["source_selections"],
            **{
                f"inputs/attacks/{a['artifact_sha256']}.json": a
                for a in manifest["attack_artifacts"]
            },
        }
        for relative, content in snapshots.items():
            target = path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(content))
        manifest.setdefault(
            "file_hashes",
            {
                relative: hashlib.sha256((path / relative).read_bytes()).hexdigest()
                for relative in ["episodes.parquet", *snapshots]
            },
        )
        (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def cell(summary, dimension="model", source="source", target="target", **stratum):
    matrix = summary["matrices"][dimension]
    matches = [
        c
        for c in matrix["cells"]
        if c["source"] == source
        and c["target"] == target
        and all(matrix["strata"][c["stratum_id"]].get(k) == v for k, v in stratum.items())
    ]
    assert len(matches) == 1
    return matches[0]


def optimized(**kwargs: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = dict(
        mode="source_optimized",
        attack_source_model="unit/source-v1",
        attack_source_scaffold="direct",
        selection_id="frozen-selection",
    )
    return episode(**(defaults | kwargs))


def test_scalar_metrics_and_undefined_denominators():
    assert attack_success_rate([True, False, True]) == pytest.approx(2 / 3)
    assert attack_success_rate(2, 4) == 0.5
    assert attack_success_rate(np.array([True, False])) == 0.5
    assert raw_transfer(3, 4) == 0.75
    assert raw_transfer([]) is None
    assert transfer_ratio(0.2, 0.6) == pytest.approx(3)
    assert transfer_ratio(0, 1) is None
    assert transfer_ratio(float("nan"), 1) is None
    assert relative_risk_reduction(0, 0) is None
    assert relative_risk_reduction(0.2, 0.6) == pytest.approx(-2)
    assert utility_adjusted_defense_score(0.8, 0.4, 1, 0.8) == pytest.approx(0.4)
    assert utility_adjusted_defense_score(0.2, 0.6, 1, 0) == pytest.approx(-2.5)
    assert utility_adjusted_defense_score(0, 0, 1, 1) is None
    assert cross_transfer_score([None, 0.5, 1, float("nan")]) == 0.75
    assert cross_transfer_score([]) is None
    with pytest.raises(ValueError):
        attack_success_rate(3, 2)
    invalid_successes: list[Any] = [True, None]
    with pytest.raises(ValueError):
        attack_success_rate(invalid_successes)
    with pytest.raises(TypeError):
        attack_success_rate(1)
    with pytest.raises(ValueError):
        utility_adjusted_defense_score(1, 0, 1, 1, lambda_=float("inf"))


def test_wilson_and_task_bootstrap_keep_all_repeats():
    assert wilson_interval(5, 10) == pytest.approx([0.23659309, 0.76340691])
    assert wilson_interval(0, 0) == [None, None]
    upper = wilson_interval(0, 10)[1]
    assert upper is not None and upper > 0

    def statistic(v):
        return v[0] / v[1]

    single = cluster_bootstrap(["a", "b"], [(0, 1), (1, 1)], statistic)
    repeated = cluster_bootstrap(["a"] * 20 + ["b"] * 20, [(0, 1)] * 20 + [(1, 1)] * 20, statistic)
    assert single == repeated
    assert repeated["ci95"] == [0, 1]  # not an artificially narrow 40-trial CI
    assert repeated == cluster_bootstrap(
        ["a"] * 20 + ["b"] * 20, [(0, 1)] * 20 + [(1, 1)] * 20, statistic
    )
    paired = paired_estimate(["a", "b"], [(1, 1, 1), (0, 0, 1)], lambda v: (v[0] - v[1]) / v[2])
    assert paired["ci95"] == [0, 0]


def test_summary_excludes_errors_clean_failures_and_selection(tmp_path):
    rows = [
        episode(success=True),
        episode(task="t2", success=False),
        episode(task="t3", status="error", error="provider failed"),
        episode(task="t4", status="budget_exceeded"),
        episode(task="t5", attack=None, success=True, utility=False),
        episode(task="t6", split="selection"),
    ]
    run = write_run(tmp_path, rows)
    summary = analyze_run(run)
    assert summary["counts"] == {
        "total": 6,
        "valid": 4,
        "evaluation": 3,
        "excluded": 2,
        "exclusions": {"execution_failure": 2},
        "non_evaluation": 1,
        "attack_trials": 2,
        "clean_trials": 1,
        "simulated": 6,
    }
    attacked = next(g for g in summary["groups"] if g["condition"] == "C1")
    clean = next(g for g in summary["groups"] if g["condition"] == "C0")
    assert attacked["attack_success_rate"]["mean"] == 0.5
    assert clean["attack_success_rate"]["n"] == 0
    assert clean["attack_success_rate"]["mean"] is None
    assert summary["spending"][0]["estimated_cost_usd"] == pytest.approx(0.06)
    assert summary["spending"][0]["n_failures"] == 2
    assert summary["longitudinal"][0]["version"] == "unit/source-v1"
    assert json.loads((run / "summary.json").read_text()) == summary


def test_fixed_cofailure_is_not_source_selection_asr(tmp_path):
    rows = [
        episode(task="a", model="source", success=True),
        episode(task="b", model="source", success=False),
        episode(task="a", model="target", success=True),
        episode(task="b", model="target", success=True),
    ]
    summary = analyze_run(write_run(tmp_path, rows))
    transfer = cell(summary)
    assert transfer["kind"] == "conditional_cofailure"
    assert transfer["source_asr"] is None
    assert transfer["transfer_ratio"] is None
    assert transfer["estimate"]["mean"] == 1
    assert transfer["estimate"]["n"] == 1
    assert transfer["n_pairs"] == 2
    assert "NOT attack generation" in summary["matrices"]["model"]["description"]


def test_optimized_uses_heldout_source_not_selection_asr(tmp_path):
    rows = [
        optimized(task="a", success=True),
        optimized(task="b", success=False),
        optimized(task="a", model="target", success=True),
        optimized(task="b", model="target", success=True),
        optimized(task="a", split="selection", success=True),
        optimized(task="b", split="selection", success=True),
    ]
    summary = analyze_run(write_run(tmp_path, rows))
    transfer = cell(summary)
    assert transfer["kind"] == "held_out_source_optimized"
    assert transfer["n_pairs"] == 2
    assert transfer["source_asr"]["mean"] == 0.5
    assert transfer["estimate"]["mean"] == 1
    assert transfer["transfer_ratio"]["mean"] == 2
    assert transfer["generalization_gap"]["mean"] == 0.5
    assert not [c for c in summary["matrices"]["model"]["cells"] if c["source"] == "target"]


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": 12},
        {"repeat": 1},
        {"task_id": "different"},
        {"split": "confirmatory"},
        {"defense_id": "guard", "condition": "C2"},
        {"scaffold_id": "other"},
        {"attack_artifact_sha256": "b" * 64},
        {"selection_id": "different-selection"},
    ],
)
def test_optimized_never_matches_confounded_trials(tmp_path, changes):
    rows = [optimized(), optimized(model="target", **changes)]
    summary = analyze_run(write_run(tmp_path, rows))
    assert not [
        c
        for c in summary["matrices"]["model"]["cells"]
        if c["source"] == "source" and c["target"] == "target"
    ]


def test_source_scaffold_origin_and_artifact_are_verified(tmp_path):
    rows = [optimized(scaffold="other"), optimized(model="target", scaffold="other")]
    summary = analyze_run(write_run(tmp_path, rows))
    assert summary["matrices"]["model"]["cells"] == []
    rows = [optimized(), optimized(scaffold="other")]
    summary = analyze_run(write_run(tmp_path, rows))
    assert cell(summary, "scaffold", "direct", "other")["estimate"]["mean"] == 1


def test_all_four_matrices_and_attack_intervention_semantics(tmp_path):
    rows = [
        episode(attack=attack, defense=defense)
        for attack in ("attack-a", "attack-b")
        for defense in ("none", "guard")
    ]
    summary = analyze_run(write_run(tmp_path, rows))
    assert set(summary["matrices"]) == {"model", "scaffold", "attack", "defense"}
    attack = cell(summary, "attack", "attack-a", "attack-b", defense_id="none")
    assert attack["kind"] == "conditional_cofailure"
    assert "distinct frozen interventions" in summary["matrices"]["attack"]["description"]
    assert cell(summary, "defense", "none", "guard")["n_pairs"] == 2


def test_matched_defense_rrr_utility_tax_and_raw_uads(tmp_path):
    rows = []
    for task in ("a", "b"):
        for repeat in range(2):
            rows += [
                episode(task=task, repeat=repeat, success=True),
                episode(task=task, repeat=repeat, defense="guard", success=task == "b"),
                episode(task=task, repeat=repeat, attack=None, utility=True),
                episode(
                    task=task, repeat=repeat, attack=None, defense="guard", utility=task == "b"
                ),
            ]
    # Deliberately unmatched baseline trials must not change the estimate.
    rows += [
        episode(task="unmatched", success=False),
        episode(task="unmatched", attack=None, utility=False),
    ]
    summary = analyze_run(write_run(tmp_path, rows))
    comparison = summary["defense_comparisons"][0]
    assert comparison["baseline_asr"]["mean"] == 1
    assert comparison["baseline_asr"]["n"] == 4
    assert comparison["defended_asr"]["mean"] == 0.5
    assert comparison["relative_risk_reduction"]["mean"] == 0.5
    assert comparison["relative_risk_reduction"]["ci95"] == [0, 1]
    assert comparison["relative_risk_reduction"]["n_tasks"] == 2
    assert comparison["utility_tax"]["mean"] == 0.5
    assert comparison["uads"]["mean"] == 0.25
    assert comparison["uads"]["n"] == 4
    assert comparison["uads"]["n_clean"] == 4
    assert summary == analyze_run(tmp_path)


def test_uads_requires_joint_task_repeat_support(tmp_path):
    rows = [
        episode(task="attack"),
        episode(task="attack", defense="guard", success=False),
        episode(task="clean", attack=None),
        episode(task="clean", attack=None, defense="guard"),
    ]
    comparison = analyze_run(write_run(tmp_path, rows))["defense_comparisons"][0]
    assert comparison["relative_risk_reduction"]["mean"] == 1
    assert comparison["utility_tax"]["mean"] == 0
    assert comparison["uads"]["mean"] is None
    assert comparison["uads"]["n"] == 0


def test_zero_baseline_rrr_and_transfer_ratio_are_null(tmp_path):
    rows = [
        optimized(success=False),
        optimized(model="target", success=True),
        optimized(defense="guard", success=True),
    ]
    summary = analyze_run(write_run(tmp_path, rows))
    assert cell(summary)["transfer_ratio"]["mean"] is None
    assert cell(summary)["transfer_ratio"]["ci95"] == [None, None]
    assert summary["defense_comparisons"][0]["relative_risk_reduction"]["mean"] is None
    assert "NaN" not in (tmp_path / "summary.json").read_text()


def test_ambiguous_matches_do_not_cartesian_multiply(tmp_path):
    rows = [episode(), episode(), episode(model="target"), episode(defense="guard", success=False)]
    summary = analyze_run(write_run(tmp_path, rows))
    assert not [
        c
        for c in summary["matrices"]["model"]["cells"]
        if c["source"] == "source" and c["target"] == "target"
    ]
    comparison = summary["defense_comparisons"][0]
    assert comparison["baseline_asr"]["n"] == 0
    assert comparison["ambiguous_attack_matches"] == 1


def test_duplicates_and_nonfinite_outcomes_excluded(tmp_path):
    duplicate = episode()
    bad = episode(task="bad")
    bad["attack_success"] = float("nan")
    # Keep a numeric column to exercise NaN scalar rejection without casting.
    duplicate["attack_success"] = 1.0
    summary = analyze_run(write_run(tmp_path, [duplicate, duplicate, bad]))
    assert summary["counts"]["excluded"] == 3
    assert summary["counts"]["exclusions"]["duplicate_episode_id"] == 2
    assert summary["counts"]["exclusions"]["invalid_outcome"] == 1


def test_nested_columns_are_not_decoded_and_costs_remain_finite(tmp_path):
    row = episode()
    row.update(estimated_cost_usd=float("inf"), latency_ms=float("nan"))
    run = write_run(tmp_path, [row])
    frame = pl.read_parquet(run / "episodes.parquet").with_columns(
        pl.lit("not even valid JSON </script>").alias("transcript")
    )
    frame.write_parquet(run / "episodes.parquet")
    summary = analyze_run(run)
    assert summary["counts"]["attack_trials"] == 1
    assert summary["spending"][0]["estimated_cost_usd"] is None
    assert summary["spending"][0]["cost_valid_n"]["estimated_cost_usd"] == 0
    assert summary["spending"][0]["invalid_cost_values"] == {
        "latency_ms": 1,
        "estimated_cost_usd": 1,
    }
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize(
    "kind",
    [
        "manifest_false",
        "simulated",
        "missing_manifest",
        "missing_artifacts",
        "missing_selection",
        "wrong_selection",
        "wrong_origin",
        "missing_simulation_flag",
        "mismatched_run",
    ],
)
def test_publishable_fails_closed_without_writing(tmp_path, kind):
    rows = [optimized(simulated=False), optimized(model="target", simulated=False)]
    overrides: dict[str, Any] = {"publishable": True}
    if kind == "manifest_false":
        overrides["publishable"] = False
    elif kind == "simulated":
        rows.append(optimized(status="error", simulated=True))
    elif kind == "missing_artifacts":
        overrides["attack_artifacts"] = []
    elif kind == "missing_selection":
        overrides["source_selections"] = []
    elif kind == "wrong_selection":
        overrides["source_selections"] = [
            {
                "selection_id": "frozen-selection",
                "artifact_sha256": "c" * 64,
                "split": "selection",
                "source_model": "source-v1",
                "source_scaffold": "direct",
            }
        ]
    elif kind == "wrong_origin":
        overrides["attack_artifacts"] = [
            {
                "id": "attack-a",
                "artifact_sha256": rows[0]["attack_artifact_sha256"],
                "source_model": "fake",
                "source_scaffold": "direct",
                "selection_id": "frozen-selection",
            }
        ]
    elif kind == "missing_simulation_flag":
        for row in rows:
            del row["simulated"]
    elif kind == "mismatched_run":
        overrides["run_id"] = "different"
    write_run(tmp_path, rows, with_integrity=True, **overrides)
    if kind == "missing_manifest":
        (tmp_path / "manifest.json").unlink()
    with pytest.raises(ValueError, match="not publishable"):
        analyze_run(tmp_path, publishable=True)
    assert not (tmp_path / "summary.json").exists()
    with pytest.raises(ValueError, match="not publishable"):
        generate_report(tmp_path, publishable=True)
    assert not (tmp_path / "report.html").exists()


def test_publishability_checks_accept_complete_test_metadata(tmp_path, monkeypatch):
    # Test the gate only; these local test fixtures are NOT real model evidence.
    write_run(
        tmp_path,
        [optimized(simulated=False), optimized(model="target", simulated=False)],
        publishable=True,
        with_integrity=True,
    )
    from transferbench.runner import manifest as runner_manifest

    called = []
    original = runner_manifest.verify_run

    def verify(path):
        called.append(path)
        return original(path)

    monkeypatch.setattr(runner_manifest, "verify_run", verify)
    summary = analyze_run(tmp_path, publishable=True)
    assert summary["publishable"] is True
    assert summary["file_integrity"]["valid"] is True
    assert called == [tmp_path]


def test_provenance_failure_suppresses_transfer_not_descriptive_asr(tmp_path):
    summary = analyze_run(
        write_run(tmp_path, [optimized(), optimized(model="target")], source_selections=[])
    )
    assert summary["counts"]["attack_trials"] == 2
    assert summary["matrices"]["model"]["cells"] == []
    assert summary["provenance_failures"]
    assert summary["matrices"]["model"]["labels"] == ["source", "target"]
    assert summary["matrices"]["model"]["grids"][0]["values"] == [[None, None], [None, None]]


def test_report_has_five_figures_single_offline_runtime_and_escaped_labels(tmp_path):
    malicious = '</script><img src=x onerror="alert(1)">'
    rows = [
        episode(model=malicious),
        episode(model="target"),
        episode(model=malicious, defense="guard", success=False),
        episode(model=malicious, attack=None),
        episode(model=malicious, attack=None, defense="guard"),
    ]
    write_run(tmp_path, rows, run_id="synthetic-unit-test")
    output = generate_report(tmp_path, tmp_path / "nested" / "custom.html")
    text = output.read_text()
    assert text.count("<script id=plotly-runtime>") == 1
    assert "<script src=" not in text
    assert text.count('class="figure"') == 5
    assert text.count('type="application/json"') == 5
    assert malicious not in text
    assert "&lt;img" in text
    assert "SIMULATED DATA — NOT REAL MODEL EVIDENCE" in text
    for heading in (
        "Most transferable",
        "Least transferable",
        "Robust defenses",
        "Generalization gaps",
        "Scaffold transitions",
        "Spending",
        "Longitudinal",
    ):
        assert heading in text
    assert output == tmp_path / "nested" / "custom.html"


def test_empty_run_and_unsupported_cells_are_finite(tmp_path):
    write_run(tmp_path, [])
    summary = analyze_run(tmp_path)
    assert summary["counts"]["total"] == 0
    assert summary["groups"] == []
    assert all(not matrix["cells"] for matrix in summary["matrices"].values())
    json.dumps(summary, allow_nan=False)
    output = generate_report(tmp_path)
    assert output.exists()
    assert output.read_text().count("No supported matched observations") == 5


def test_report_cannot_overwrite_run_inputs(tmp_path):
    write_run(tmp_path, [episode()])
    before = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        generate_report(tmp_path, tmp_path / "manifest.json")
    assert (tmp_path / "manifest.json").read_bytes() == before


def test_bootstrap_undefined_draws_are_counted_not_zero_filled():
    result = paired_estimate(["a", "b"], [(1, 1), (0, 0)], lambda v: v[1] / v[0] if v[0] else None)
    assert result["mean"] == 1
    assert result["ci95"] == [1, 1]
    assert 0 < result["bootstrap_valid"] < 2000
    with pytest.raises(ValueError):
        cluster_bootstrap(["a"], [(np.inf, 1)], lambda v: v[0] / v[1])


def test_family_mode_is_explicit_conditional_cofailure(tmp_path):
    rows = [
        episode(mode="family", model=model, scaffold=scaffold, attack=f"variant-{model}-{scaffold}")
        for model in ("source", "target")
        for scaffold in ("direct", "other")
    ]
    summary = analyze_run(write_run(tmp_path, rows))
    for transfer in (
        cell(summary, scaffold_id="direct"),
        cell(summary, "scaffold", "direct", "other", model_alias="source"),
    ):
        assert transfer["kind"] == "family_conditional_cofailure"
        assert transfer["source_asr"] is None
        assert transfer["estimate"]["mean"] == 1
        assert transfer["different_artifact_pairs"] == 1
        assert "NOT same-artifact" in transfer["matching_basis"]
    rows[1]["repeat"] = 1
    summary = analyze_run(write_run(tmp_path, rows))
    assert not [
        c
        for c in summary["matrices"]["scaffold"]["cells"]
        if c["source"] != c["target"]
        and summary["matrices"]["scaffold"]["strata"][c["stratum_id"]]["model_alias"] == "source"
    ]


def test_model_alias_versions_are_not_silently_pooled(tmp_path):
    rows = [episode(), episode(model_id="unit/source-v2", success=False), episode(model="target")]
    summary = analyze_run(write_run(tmp_path, rows))
    assert summary["matrices"]["model"]["labels"] == [
        "source [unit/source-v1]",
        "source [unit/source-v2]",
        "target",
    ]
    assert cell(summary, source="source [unit/source-v1]")["estimate"]["mean"] == 1
    assert cell(summary, source="source [unit/source-v2]")["estimate"]["mean"] is None
    assert {r["version"] for r in summary["longitudinal"]} == {
        "unit/source-v1",
        "unit/source-v2",
        "unit/target-v1",
    }


def test_ambiguous_source_alias_cannot_supply_optimized_baseline(tmp_path):
    rows = [
        optimized(attack_source_model="source"),
        optimized(attack_source_model="source", model_id="unit/source-v2"),
        optimized(attack_source_model="source", model="target"),
    ]
    summary = analyze_run(write_run(tmp_path, rows))
    assert summary["matrices"]["model"]["cells"] == []


def test_missing_scalar_provenance_fails_publication(tmp_path):
    row = episode(simulated=False)
    del row["model_id"]
    write_run(tmp_path, [row], publishable=True)
    with pytest.raises(ValueError, match="scalar episode identity"):
        analyze_run(tmp_path, publishable=True)
    assert not (tmp_path / "summary.json").exists()


def test_invalid_identity_cannot_leak_nan_to_summary(tmp_path):
    row = episode()
    row["model_alias"] = float("nan")
    summary = analyze_run(write_run(tmp_path, [row]))
    assert summary["counts"]["exclusions"] == {"invalid_identity": 1}
    assert summary["spending"][0]["model_alias"] is None
    json.dumps(summary, allow_nan=False)


def test_harmful_defense_rrr_and_uads_are_not_clipped(tmp_path):
    rows = []
    for task in ("a", "b"):
        rows += [
            episode(task=task, success=task == "a"),
            episode(task=task, defense="guard"),
            episode(task=task, attack=None),
            episode(task=task, attack=None, defense="guard", utility=False),
        ]
    comparison = analyze_run(write_run(tmp_path, rows))["defense_comparisons"][0]
    assert comparison["relative_risk_reduction"]["mean"] == -1
    assert comparison["uads"]["mean"] == -1.5


def test_unverified_artifacts_cannot_enter_defense_comparisons(tmp_path):
    summary = analyze_run(
        write_run(tmp_path, [episode(), episode(defense="guard")], attack_artifacts=[])
    )
    assert summary["counts"]["attack_trials"] == 2
    assert summary["defense_comparisons"] == []


@pytest.mark.parametrize(
    "failure",
    [
        "no_hashes",
        "missing_episode_hash",
        "changed_file",
        "snapshot_disagreement",
        "runner_blocker",
    ],
)
def test_publication_verifies_runner_files_and_attestations(tmp_path, failure):
    write_run(tmp_path, [episode(simulated=False)], with_integrity=True, publishable=True)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if failure == "no_hashes":
        del manifest["file_hashes"]
    elif failure == "missing_episode_hash":
        del manifest["file_hashes"]["episodes.parquet"]
    elif failure == "changed_file":
        (tmp_path / "inputs" / "tasks.json").write_text("[]")
    elif failure == "snapshot_disagreement":
        manifest["resolved_config"]["seed"] = 987
    else:
        manifest["publication_blockers"] = ["synthetic recorded blocker"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not publishable"):
        analyze_run(tmp_path, publishable=True)
    assert not (tmp_path / "summary.json").exists()
    summary = analyze_run(tmp_path)
    assert not summary["publishable"]
    if failure in {"missing_episode_hash", "changed_file", "snapshot_disagreement"}:
        assert not summary["file_integrity"]["valid"]
        assert summary["matrices"]["model"]["cells"] == []


def test_provider_model_identity_not_bare_model_name(tmp_path):
    # Even consistently hashed snapshots cannot override Episode's provider/model.
    spec = dict(
        provider="other-provider",
        model="source-v1",
        family="fixture-family",
        pricing={},
        version_pinned=True,
    )
    write_run(
        tmp_path,
        [episode(simulated=False)],
        with_integrity=True,
        publishable=True,
        models={"source": spec},
    )
    summary = analyze_run(tmp_path)
    assert summary["file_integrity"]["valid"]
    assert not summary["publishable"]
    assert (
        summary["provenance_failures"][
            "episode model alias/provider/model/family does not match manifest"
        ]
        == 1
    )


def test_declared_source_selected_defense_transfer_is_target_rrr(tmp_path):
    rows = []
    origin: dict[str, Any] = dict(defense_source_model="source", defense_source_scaffold="direct")
    for task in ("a", "b"):
        for model in ("source", "target", "zero"):
            rows += [
                episode(
                    task=task,
                    model=model,
                    success=model == "source" or (model == "target" and task == "a"),
                ),
                episode(
                    task=task, model=model, defense="guard", success=model == "target", **origin
                ),
            ]
        rows.append(episode(task=task, model="target", defense="generic"))
    summary = analyze_run(write_run(tmp_path, rows))
    transfer = summary["matrices"]["defense"]["source_selected_transfer"]
    target = next(c for c in transfer["cells"] if c["target_model_alias"] == "target")
    assert target["source"] == "unit/source-v1 / direct"
    assert target["kind"] == "declared_source_selected_defense_transfer"
    assert target["rrr_t"]["mean"] == -1
    assert target["rrr_t"]["n"] == 2 and target["rrr_t"]["n_tasks"] == 2
    assert target["rrr_s"] is None and target["generalization_gap"] is None
    assert (
        next(c for c in transfer["cells"] if c["target_model_alias"] == "zero")["rrr_t"]["mean"]
        is None
    )
    assert not [c for c in transfer["cells"] if c["defense_id"] == "generic"]
    assert transfer["missing_origin_groups"] == 1
    assert "Source-selected defense transfer" in generate_report(tmp_path).read_text()


def test_aggregate_transfer_bootstrap_resamples_tasks_jointly(tmp_path):
    correlated = [[("a", 0, 1), ("b", 1, 1)]] * 2
    estimate = macro_rate_estimate(correlated)
    assert estimate["mean"] == 0.5 and estimate["ci95"] == [0, 1]
    assert macro_rate_estimate([cell * 10 for cell in correlated])["ci95"] == estimate["ci95"]
    assert macro_rate_estimate([])["mean"] is None
    rows = [
        episode(task=task, model=model, scaffold=scaffold, success=model == "source" or task == "a")
        for task in ("a", "b")
        for model in ("source", "target")
        for scaffold in ("direct", "other")
    ]
    summary = analyze_run(write_run(tmp_path, rows))
    for dimension in ("model", "scaffold"):
        matrix = summary["matrices"][dimension]
        aggregate = matrix["cross_transfer_score"]["conditional_cofailure"]
        assert aggregate["n_cells"] > 0 and aggregate["n_tasks"] == 2
        assert aggregate["ci95"][0] is not None
        assert 0 < aggregate["bootstrap_valid"] <= 2000
        assert matrix["cross_transfer_by_stratum"]
    mts = summary["matrices"]["model"]["cross_transfer_score"]["conditional_cofailure"]
    assert mts["mean"] == 0.75 and mts["ci95"] == [0.75, 1]
    assert mts["bootstrap_valid"] < 2000  # draws cannot silently drop unsupported reverse cells
    assert (
        analyze_run(tmp_path)["matrices"]["model"]["cross_transfer_score"]
        == summary["matrices"]["model"]["cross_transfer_score"]
    )
