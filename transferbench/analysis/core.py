"""Run-level analysis. Only scalar Episode columns are loaded from Parquet."""

import json
from collections import Counter, defaultdict
from itertools import product
from math import isfinite
from pathlib import Path
from typing import Any

import polars as pl

from .provenance import Provenance, publication_metadata_issues, verify_integrity
from .statistics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    macro_rate_estimate,
    paired_estimate,
    rate,
    wilson_interval,
)

SCALAR_COLUMNS = {
    "schema_version",
    "run_id",
    "episode_id",
    "model_id",
    "model_family",
    "model_alias",
    "scaffold_id",
    "task_id",
    "task_family",
    "attack_id",
    "attack_family",
    "attack_source_model",
    "attack_source_scaffold",
    "attack_artifact_sha256",
    "selection_id",
    "defense_id",
    "defense_source_model",
    "defense_source_scaffold",
    "condition",
    "transfer_mode",
    "split",
    "seed",
    "repeat",
    "utility_success",
    "attack_success",
    "policy_violation",
    "input_tokens",
    "cached_tokens",
    "output_tokens",
    "latency_ms",
    "estimated_cost_usd",
    "propagation_depth",
    "status",
    "error",
    "simulated",
}
EVALUATION_SPLITS = {"evaluation", "confirmatory"}
IDENTITY = ["model_alias", "model_id", "scaffold_id"]
TRIAL = ["task_id", "seed", "repeat", "split", "transfer_mode"]
ARTIFACT = [
    "attack_id",
    "attack_artifact_sha256",
    "selection_id",
    "attack_source_model",
    "attack_source_scaffold",
]
COSTS = ["input_tokens", "cached_tokens", "output_tokens", "latency_ms", "estimated_cost_usd"]


def _key(row: dict, fields: list[str]) -> tuple:
    return tuple(row.get(field) for field in fields)


def _groups(rows: list[dict], fields: list[str]) -> list[tuple[dict, list[dict]]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_key(row, fields)].append(row)
    return [
        (dict(zip(fields, key, strict=True)), grouped[key]) for key in sorted(grouped, key=repr)
    ]


def _matched(
    left: list[dict], right: list[dict], fields: list[str]
) -> tuple[list[tuple[dict, dict]], int]:
    """One-to-one exact matching, never a many-to-many/cartesian join."""
    indices = []
    for rows in (left, right):
        index: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            index[_key(row, fields)].append(row)
        indices.append(index)
    pairs = []
    ambiguous = 0
    for key in sorted(indices[0].keys() & indices[1].keys(), key=repr):
        a, b = indices[0][key], indices[1][key]
        if len(a) == len(b) == 1:
            pairs.append((a[0], b[0]))
        else:
            ambiguous += 1
    return pairs, ambiguous


def _rate(rows: list[dict], metric: str) -> dict:
    return rate([row[metric] for row in rows], [row["task_id"] for row in rows])


def _load(run_dir: Path) -> tuple[dict, list[dict], list[str], set[str]]:
    issues = []
    try:

        def reject_constant(value: str) -> None:
            raise ValueError(f"nonfinite JSON constant: {value}")

        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8"), parse_constant=reject_constant
        )
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
    except (OSError, ValueError) as exc:
        manifest = {}
        issues.append(f"manifest unavailable or invalid: {exc}")
    parquet = run_dir / "episodes.parquet"
    schema = pl.read_parquet_schema(parquet)
    columns = sorted(SCALAR_COLUMNS & schema.keys())
    # Do not load or decode tool_calls, transcripts, call_costs, or event columns.
    rows = pl.read_parquet(parquet, columns=columns).to_dicts() if columns else []
    if not columns and pl.scan_parquet(parquet).select(pl.len()).collect().item():
        raise ValueError("episodes.parquet has no recognized scalar Episode columns")
    return manifest, rows, issues, set(schema)


def _invalid(row: dict) -> str | None:
    if row.get("status") != "ok" or row.get("error"):
        return "execution_failure"
    for field in [*IDENTITY, "run_id", "episode_id", "task_id", "model_family", "defense_id"]:
        if not isinstance(row.get(field), str) or not row[field]:
            return "invalid_identity"
    for field in [*ARTIFACT, "attack_family", "defense_source_model", "defense_source_scaffold"]:
        if row.get(field) is not None and not isinstance(row[field], str):
            return "invalid_provenance_scalar"
    for field in ("attack_success", "utility_success"):
        if not isinstance(row.get(field), bool):
            return "invalid_outcome"
    if row.get("transfer_mode") not in ("fixed", "family", "source_optimized"):
        return "invalid_transfer_mode"
    if row.get("split") not in (*EVALUATION_SPLITS, "screening", "selection"):
        return "invalid_split"
    if not isinstance(row.get("seed"), int) or isinstance(row["seed"], bool):
        return "invalid_seed_or_repeat"
    if (
        not isinstance(row.get("repeat"), int)
        or isinstance(row["repeat"], bool)
        or row["repeat"] < 0
    ):
        return "invalid_seed_or_repeat"
    expected = (
        ("C1" if row["defense_id"] == "none" else "C2")
        if row.get("attack_id")
        else ("C0" if row["defense_id"] == "none" else "C3")
    )
    if row.get("condition") != expected:
        return "incoherent_condition"
    return None


def _defense_comparisons(rows: list[dict]) -> list[dict]:
    records = []
    for identity, group in _groups(rows, [*IDENTITY, "split", "transfer_mode"]):
        for defense_id in sorted({r["defense_id"] for r in group} - {"none"}):
            defended = [r for r in group if r["defense_id"] == defense_id]
            # Partition by declared defense origin as well: tuned defenses from
            # different sources are not treated as one frozen intervention.
            for origin, defense_rows in _groups(
                defended, ["defense_source_model", "defense_source_scaffold"]
            ):
                attack_pairs, attack_ambiguous = _matched(
                    [r for r in group if r["condition"] == "C1"],
                    [r for r in defense_rows if r["condition"] == "C2"],
                    TRIAL + ARTIFACT,
                )
                clean_pairs, clean_ambiguous = _matched(
                    [r for r in group if r["condition"] == "C0"],
                    [r for r in defense_rows if r["condition"] == "C3"],
                    TRIAL,
                )
                tasks = [a["task_id"] for a, _ in attack_pairs]
                attack_values = [
                    (float(a["attack_success"]), float(b["attack_success"]), 1)
                    for a, b in attack_pairs
                ]
                rrr = paired_estimate(
                    tasks, attack_values, lambda v: 1 - v[1] / v[0] if v[0] else None
                )
                utility_tax = paired_estimate(
                    [a["task_id"] for a, _ in clean_pairs],
                    [
                        (float(a["utility_success"]), float(b["utility_success"]), 1)
                        for a, b in clean_pairs
                    ],
                    lambda v: (v[0] - v[1]) / v[2],
                )
                # UADS uses only evaluation task/seed/repeat strata with both
                # paired attack and paired clean observations. Repeated attacks
                # contribute to security, not repeated copies of clean utility.
                clean_index = {_key(a, TRIAL): (a, b) for a, b in clean_pairs}
                joint_attacks = [(a, b) for a, b in attack_pairs if _key(a, TRIAL) in clean_index]
                joint_keys = {_key(a, TRIAL) for a, _ in joint_attacks}
                joint_clean = [clean_index[key] for key in sorted(joint_keys, key=repr)]
                joint_tasks = [a["task_id"] for a, _ in joint_attacks + joint_clean]
                joint_values = [
                    (float(a["attack_success"]), float(b["attack_success"]), 1, 0, 0, 0)
                    for a, b in joint_attacks
                ] + [
                    (0, 0, 0, float(a["utility_success"]), float(b["utility_success"]), 1)
                    for a, b in joint_clean
                ]
                uads = paired_estimate(
                    joint_tasks,
                    joint_values,
                    lambda v: (
                        1 - v[1] / v[0] - 0.5 * (v[3] - v[4]) / v[5] if v[0] and v[5] else None
                    ),
                )
                uads.update(n=len(joint_attacks), n_clean=len(joint_clean), lambda_=0.5)
                records.append(
                    {
                        **identity,
                        **origin,
                        "defense_id": defense_id,
                        "baseline_asr": _rate([a for a, _ in attack_pairs], "attack_success"),
                        "defended_asr": _rate([b for _, b in attack_pairs], "attack_success"),
                        "baseline_utility": _rate([a for a, _ in clean_pairs], "utility_success"),
                        "defended_utility": _rate([b for _, b in clean_pairs], "utility_success"),
                        "relative_risk_reduction": rrr,
                        "utility_tax": utility_tax,
                        "uads": uads,
                        "ambiguous_attack_matches": attack_ambiguous,
                        "ambiguous_clean_matches": clean_ambiguous,
                        "unmatched_attack_rows": sum(r["condition"] == "C2" for r in defense_rows)
                        - len(attack_pairs),
                        "unmatched_clean_rows": sum(r["condition"] == "C3" for r in defense_rows)
                        - len(clean_pairs),
                    }
                )
    return records


def _source_selected_defense_transfer(
    comparisons: list[dict], rows: list[dict], provenance: Provenance
) -> dict:
    """RRR_T on target C1/C2 pairs for defenses with an explicit source origin.

    Unlike defense-ID co-failure, the axes are source and target environments.
    RRR_T needs no source outcome denominator. The schema declares origin but
    does not attest a frozen defense artifact/selection procedure, so no RRR_S
    or source-to-target generalization gap is manufactured from unequal support.
    """
    names: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        names[row["model_alias"]].add(row["model_id"])
        names[row["model_id"]].add(row["model_id"])
    for alias, spec in provenance.models.items():
        if isinstance(spec, dict) and spec.get("provider") and spec.get("model"):
            model_id = f"{spec['provider']}/{spec['model']}"
            names[alias].add(model_id)
            names[model_id].add(model_id)
    targets = sorted({f"{row['model_id']} / {row['scaffold_id']}" for row in rows})
    sources = set()
    strata, cells = [], []
    missing_origin = unresolved_origin = 0
    for stratum, group in _groups(comparisons, ["defense_id", "split", "transfer_mode"]):
        stratum_id = len(strata)
        strata.append({"id": stratum_id, **stratum})
        for comparison in group:
            model, scaffold = (
                comparison["defense_source_model"],
                comparison["defense_source_scaffold"],
            )
            if not model or not scaffold:
                missing_origin += 1
                continue
            if len(names[model]) != 1:
                unresolved_origin += 1
                continue
            source_id = next(iter(names[model]))
            source = f"{source_id} / {scaffold}"
            target = f"{comparison['model_id']} / {comparison['scaffold_id']}"
            sources.add(source)
            cells.append(
                {
                    "source": source,
                    "target": target,
                    "stratum_id": stratum_id,
                    "source_model": model,
                    "source_model_id": source_id,
                    "source_scaffold": scaffold,
                    "target_model_alias": comparison["model_alias"],
                    "target_model_id": comparison["model_id"],
                    "target_scaffold": comparison["scaffold_id"],
                    "defense_id": comparison["defense_id"],
                    "kind": "declared_source_selected_defense_transfer",
                    "provenance_level": "declared_source_origin_only; no defense selection/artifact attestation in Episode",
                    "rrr_t": comparison["relative_risk_reduction"],
                    "estimate": comparison["relative_risk_reduction"],
                    "baseline_asr_t": comparison["baseline_asr"],
                    "defended_asr_t": comparison["defended_asr"],
                    "n_target_pairs": comparison["relative_risk_reduction"]["n"],
                    "rrr_s": None,
                    "generalization_gap": None,
                }
            )
    sources = sorted(sources)
    grids = []
    for stratum in strata:
        # Alias duplicates of one provider/model can have different settings;
        # do not arbitrarily overwrite them in a source-target cell.
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for record in cells:
            if record["stratum_id"] == stratum["id"]:
                grouped[(record["source"], record["target"])].append(record)
        lookup = {
            key: items[0]["rrr_t"]["mean"] for key, items in grouped.items() if len(items) == 1
        }
        grids.append(
            {
                "stratum_id": stratum["id"],
                "values": [[lookup.get((s, t)) for t in targets] for s in sources],
            }
        )
    return {
        "axis": "model_id / scaffold_id",
        "source_labels": sources,
        "target_labels": targets,
        "strata": strata,
        "cells": cells,
        "grids": grids,
        "missing_origin_groups": missing_origin,
        "unresolved_origin_groups": unresolved_origin,
        "description": "RRR_T = 1 - ASR(C2,T,d_source) / ASR(C1,T), on matched target task/artifact/seed/repeat pairs. Source origin is declared, not independently proven selection. Null means unsupported, including zero target baseline risk. This is not defense-ID residual co-failure.",
    }


def _source_origin(row: dict, model_versions: dict[str, set[str]]) -> bool:
    source = row.get("attack_source_model")
    model_matches = source == row["model_id"] or (
        source == row["model_alias"] and len(model_versions[row["model_alias"]]) == 1
    )
    return model_matches and row.get("attack_source_scaffold") == row["scaffold_id"]


def _transfer_cell(
    pairs: list[tuple[dict, dict]], optimized: bool, family_variants: bool = False
) -> dict:
    reference = [a for a, _ in pairs]
    target = [b for _, b in pairs]
    tasks = [a["task_id"] for a in reference]
    values = [
        (
            float(a["attack_success"]),
            float(b["attack_success"]),
            float(a["attack_success"] and b["attack_success"]),
            1,
        )
        for a, b in pairs
    ]
    reference_rate, target_rate = (
        _rate(reference, "attack_success"),
        _rate(target, "attack_success"),
    )
    gap = paired_estimate(tasks, values, lambda v: (v[1] - v[0]) / v[3])
    ratio = paired_estimate(tasks, values, lambda v: v[1] / v[0] if v[0] else None)
    if optimized:
        estimate = target_rate
    else:
        estimate = paired_estimate(tasks, values, lambda v: v[2] / v[0] if v[0] else None)
        n = sum(a["attack_success"] for a in reference)
        successes = sum(a["attack_success"] and b["attack_success"] for a, b in pairs)
        estimate.update(
            n=n,
            successes=successes,
            n_tasks=len({a["task_id"] for a in reference if a["attack_success"]}),
            bootstrap_ci95=estimate["ci95"],
            ci95=wilson_interval(successes, n),
            ci_method="Wilson 95%",
        )
    return {
        "kind": "held_out_source_optimized"
        if optimized
        else ("family_conditional_cofailure" if family_variants else "conditional_cofailure"),
        "matching_basis": "family/task/seed/repeat; target-specific frozen variants, NOT same-artifact transfer"
        if family_variants
        else "exact trial and frozen artifact where applicable",
        "different_artifact_pairs": sum(
            a["attack_artifact_sha256"] != b["attack_artifact_sha256"] for a, b in pairs
        ),
        "n_pairs": len(pairs),
        "n_pair_tasks": len(set(tasks)),
        "estimate": estimate,
        "source_asr": reference_rate if optimized else None,
        "reference_asr": reference_rate,
        "target_asr": target_rate,
        "transfer_ratio": ratio if optimized else None,
        "generalization_gap": gap,
    }


def _matrices(rows: list[dict], provenance: Provenance) -> dict:
    # Strict provenance for transfer, even in exploratory mode. Ordinary outcome
    # summaries remain useful when producer metadata is incomplete.
    attacked = [r for r in rows if r["condition"] in {"C1", "C2"}]
    verified = {id(r) for r in attacked if not provenance.check(r)}
    model_versions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        model_versions[row["model_alias"]].add(row["model_id"])

    def axis_label(row: dict, axis: str) -> str:
        if axis == "model_alias" and len(model_versions[row[axis]]) > 1:
            return f"{row[axis]} [{row['model_id']}]"
        return row[axis]

    result = {}
    dimensions = {
        "model": "model_alias",
        "scaffold": "scaffold_id",
        "attack": "attack_id",
        "defense": "defense_id",
    }
    for name, axis in dimensions.items():
        common = ["split", "transfer_mode", "attack_source_model", "attack_source_scaffold"]
        partition = (
            common
            + {
                "model": [
                    "scaffold_id",
                    "defense_id",
                    "defense_source_model",
                    "defense_source_scaffold",
                    "condition",
                    "attack_family",
                ],
                "scaffold": [
                    "model_alias",
                    "model_id",
                    "defense_id",
                    "defense_source_model",
                    "defense_source_scaffold",
                    "condition",
                    "attack_family",
                ],
                "attack": [
                    *IDENTITY,
                    "defense_id",
                    "defense_source_model",
                    "defense_source_scaffold",
                    "condition",
                ],
                "defense": [*IDENTITY, "attack_family"],
            }[name]
        )
        labels = sorted({axis_label(r, axis) for r in attacked})
        cells = []
        strata = []
        aggregate_inputs = []
        partitioned = []
        for mode, mode_rows in _groups(attacked, ["transfer_mode"]):
            fields = partition
            if mode["transfer_mode"] == "family" and name in {"model", "scaffold"}:
                fields = [
                    field
                    for field in partition
                    if field not in {"attack_source_model", "attack_source_scaffold"}
                ]
            partitioned.extend(_groups(mode_rows, fields))
        for stratum, group in partitioned:
            stratum_id = len(strata)
            strata.append({"id": stratum_id, **stratum})
            group = [
                r
                for r in group
                if id(r) in verified
                and (name != "attack" or r["transfer_mode"] != "source_optimized")
            ]
            by_axis = {
                label: [r for r in group if axis_label(r, axis) == label] for label in labels
            }
            optimized = stratum["transfer_mode"] == "source_optimized" and name in {
                "model",
                "scaffold",
            }
            family_variants = stratum["transfer_mode"] == "family" and name in {"model", "scaffold"}
            match = TRIAL + (
                ["attack_family"] if family_variants else ([] if name == "attack" else ARTIFACT)
            )
            if family_variants and not stratum.get("attack_family"):
                continue
            for source, target in product(labels, repeat=2):
                left, right = by_axis[source], by_axis[target]
                if optimized:
                    # Source baseline is held-out evaluation of the *selected*
                    # artifact at its actual source model AND source scaffold.
                    left = [r for r in left if _source_origin(r, model_versions)]
                pairs, ambiguous = _matched(left, right, match)
                if not pairs:
                    continue
                cells.append(
                    {
                        "source": source,
                        "target": target,
                        "stratum_id": stratum_id,
                        "ambiguous_matches": ambiguous,
                        **_transfer_cell(pairs, optimized, family_variants),
                    }
                )
                if source != target:
                    aggregate_inputs.append(
                        {
                            "stratum_id": stratum_id,
                            "kind": cells[-1]["kind"],
                            "values": [
                                (
                                    a["task_id"],
                                    float(
                                        b["attack_success"]
                                        if optimized
                                        else a["attack_success"] and b["attack_success"]
                                    ),
                                    1.0 if optimized else float(a["attack_success"]),
                                )
                                for a, b in pairs
                            ],
                        }
                    )
        grids = []
        for stratum in strata:
            lookup = {
                (c["source"], c["target"]): c["estimate"]["mean"]
                for c in cells
                if c["stratum_id"] == stratum["id"]
            }
            grids.append(
                {
                    "stratum_id": stratum["id"],
                    "values": [[lookup.get((s, t)) for t in labels] for s in labels],
                }
            )
        off_diagonal = [c["estimate"]["mean"] for c in cells if c["source"] != c["target"]]
        kinds = sorted({c["kind"] for c in cells})
        result[name] = {
            "axis": axis,
            "labels": labels,
            "strata": strata,
            "cells": cells,
            "grids": grids,
            "description": (
                "Held-out source-optimized target ASR, or conditional co-failure P(target succeeds | reference succeeds). "
                "Fixed/family reference rows are NOT attack generation or selection sources. "
                "Family-mode model/scaffold cells match family/task/seed/repeat across target-specific frozen variants, NOT identical artifacts. "
                + (
                    "Attack-axis cells compare distinct frozen interventions on matched trials, not same-artifact transfer. "
                    if name == "attack"
                    else ""
                )
                + (
                    "Defense-axis cells are residual conditional co-failure, not causal defense efficacy; see defense_comparisons for C2 vs C1. "
                    if name == "defense"
                    else ""
                )
                + "Null cells are unsupported; strata must not be silently pooled."
            ),
            # Never average optimized ASRs together with conditional probabilities.
            "score_name": "MTS" if name == "model" else f"{name}_cross_transfer_score",
            "cross_transfer_score": {
                kind: macro_rate_estimate(
                    [item["values"] for item in aggregate_inputs if item["kind"] == kind]
                )
                for kind in kinds
            },
            "cross_transfer_by_stratum": [
                {**identity, **macro_rate_estimate([item["values"] for item in group])}
                for identity, group in _groups(aggregate_inputs, ["stratum_id", "kind"])
            ],
            "supported_off_diagonal_cells": sum(v is not None for v in off_diagonal),
        }
    return result


def _spending(rows: list[dict], valid_ids: set[int]) -> list[dict]:
    records = []
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        # Invalid identity columns must not leak NaN/nested values into JSON.
        grouped[
            tuple(row.get(field) if isinstance(row.get(field), str) else None for field in IDENTITY)
        ].append(row)
    for key in sorted(grouped, key=repr):
        identity, group = dict(zip(IDENTITY, key, strict=True)), grouped[key]
        totals = {}
        valid_costs = {}
        invalid_costs = Counter()
        for field in COSTS:
            values = []
            for row in group:
                value = row.get(field)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and isfinite(value)
                    and value >= 0
                ):
                    values.append(value)
                else:
                    invalid_costs[field] += 1
            total = sum(values)
            totals[field] = total if values and isfinite(total) else None
            valid_costs[field] = len(values)
        records.append(
            {
                **identity,
                "n": len(group),
                "n_valid": sum(id(r) in valid_ids for r in group),
                "n_failures": sum(id(r) not in valid_ids for r in group),
                **totals,
                "cost_valid_n": valid_costs,
                "invalid_cost_values": dict(invalid_costs),
            }
        )
    return records


def _cost_summary(rows: list[dict], valid: list[dict], manifest: dict) -> dict:
    def totals(group: list[dict], valid_n: int) -> dict:
        values = [row.get("estimated_cost_usd") for row in group]
        known = [
            value
            for value in values
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and value >= 0
        ]
        subtotal = sum(known)
        complete = bool(group) and len(known) == len(values) and isfinite(subtotal)
        total = subtotal if complete else None
        return {
            "n_episodes": len(group),
            "n_valid": valid_n,
            "cost_records_known": len(known),
            "complete": complete,
            "total_estimated_cost_usd": total,
            "cost_per_valid_episode_usd": total / valid_n
            if total is not None and valid_n
            else None,
        }

    valid_ids = {id(row) for row in valid}
    result = totals(rows, len(valid))
    for field in ("model_alias", "scaffold_id"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            label = row.get(field)
            groups[label if isinstance(label, str) else "unknown"].append(row)
        result["by_" + field] = {
            label: totals(group, sum(id(row) in valid_ids for row in group))
            for label, group in sorted(groups.items())
        }
    ledger = manifest.get("estimated_cost_usd")
    result["budget_ledger_charged_usd"] = (
        ledger
        if isinstance(ledger, (int, float))
        and not isinstance(ledger, bool)
        and isfinite(ledger)
        and ledger >= 0
        else None
    )
    projection = manifest.get("projection") or {}
    for key in ("conservative_maximum_usd", "hard_estimated_cost_ceiling_usd"):
        value = projection.get(key) if isinstance(projection, dict) else None
        result[key] = (
            value
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and value >= 0
            else None
        )
    return result


def _tables(matrices: dict, comparisons: list[dict]) -> dict:
    transfer = []
    for name in ("model", "scaffold", "attack", "defense"):
        for cell in matrices[name]["cells"]:
            if cell["source"] == cell["target"] or cell["estimate"]["mean"] is None:
                continue
            transfer.append(
                {"dimension": name, **cell, "stratum": matrices[name]["strata"][cell["stratum_id"]]}
            )
    # Rankings are separated by dimension, estimand, and stratum to avoid
    # manufacturing a leaderboard from incomparable conditional populations.
    rankings = []
    for identity, group in _groups(transfer, ["dimension", "kind", "stratum_id"]):
        ranked = sorted(group, key=lambda r: (r["estimate"]["mean"], r["source"], r["target"]))
        rankings.append({**identity, "most": ranked[-5:][::-1], "least": ranked[:5]})
    defenses = [r for r in comparisons if r["relative_risk_reduction"]["mean"] is not None]
    defenses.sort(key=lambda r: r["relative_risk_reduction"]["ci95"][0], reverse=True)
    return {
        "transferability": rankings,
        "robust_defenses": defenses,
        "generalization_gaps": [r for r in transfer if r["dimension"] in {"model", "scaffold"}],
        "scaffold_transitions": [r for r in transfer if r["dimension"] == "scaffold"],
    }


def analyze_run(run_dir: Path, publishable: bool = False) -> dict:
    """Read episodes.parquet + manifest.json; write/return JSON-safe summary.

    Publication requires manifest.publishable is True, an explicit simulation
    flag for every row, matching run IDs, and complete frozen attack/source
    selection provenance. Failure raises ValueError *before writing*. Evaluation
    and confirmatory splits are kept separate in matching; selection/screening
    outcomes never enter ASRs. See this package's API docstring for the contract.
    """
    run_dir = Path(run_dir)
    manifest, rows, issues, columns = _load(run_dir)
    integrity = verify_integrity(run_dir, manifest)
    provenance = Provenance(
        manifest, integrity_failed=integrity["checked"] and not integrity["valid"]
    )
    publication_issues = [*issues, *integrity["errors"], *publication_metadata_issues(manifest)]
    if manifest.get("publishable") is not True:
        publication_issues.append("manifest.publishable must be true")
    simulated = (
        manifest.get("simulated") is True
        or any(r.get("simulated") is True for r in rows)
        or any(
            isinstance(spec, dict) and spec.get("provider") == "fake"
            for spec in provenance.models.values()
        )
        or any(
            selection.get("source_simulated") is True
            for selections in provenance.selections.values()
            for selection in selections
        )
    )
    defense_origin = manifest.get("defense_selection") or {}
    simulated = simulated or defense_origin.get("simulated") is True
    for row in rows:
        if row.get("defense_source_model") or row.get("defense_source_scaffold"):
            if (
                not defense_origin
                or row.get("defense_id") != defense_origin.get("defense_id")
                or row.get("defense_source_model") != defense_origin.get("source_model")
                or row.get("defense_source_scaffold") != defense_origin.get("source_scaffold")
            ):
                publication_issues.append(
                    "Declared defense origin lacks matching frozen selection evidence"
                )
                break
    if simulated:
        publication_issues.append("SIMULATED episodes or manifest: not real model evidence")
    if rows and (
        "simulated" not in columns or any(not isinstance(r.get("simulated"), bool) for r in rows)
    ):
        publication_issues.append("missing explicit episode simulation provenance")
    if (
        not isinstance(manifest.get("run_id"), str)
        or not manifest["run_id"]
        or any(r.get("run_id") != manifest.get("run_id") for r in rows)
    ):
        publication_issues.append("missing or mismatched manifest/episode run_id")
    metadata_fields = [
        *IDENTITY,
        "episode_id",
        "model_family",
        "task_id",
        "defense_id",
        "split",
        "transfer_mode",
    ]
    if any(
        any(not isinstance(row.get(field), str) or not row[field] for field in metadata_fields)
        or any(
            not isinstance(row.get(field), int) or isinstance(row[field], bool)
            for field in ("seed", "repeat")
        )
        for row in rows
    ):
        publication_issues.append("missing or invalid scalar episode identity/matching provenance")
    provenance_failures = Counter()
    for row in rows:
        for problem in provenance.check(row):
            provenance_failures[problem] += 1
    publication_issues.extend(
        f"{problem} ({n} episodes)" for problem, n in sorted(provenance_failures.items())
    )
    if publishable and publication_issues:
        raise ValueError("Run is not publishable: " + "; ".join(publication_issues))

    exclusions = Counter()
    valid = []
    episode_counts = Counter(
        row["episode_id"] for row in rows if isinstance(row.get("episode_id"), str)
    )
    for row in rows:
        reason = _invalid(row)
        if isinstance(row.get("episode_id"), str) and episode_counts[row["episode_id"]] > 1:
            reason = "duplicate_episode_id"
        if reason:
            exclusions[reason] += 1
        else:
            valid.append(row)
    evaluation = [r for r in valid if r["split"] in EVALUATION_SPLITS]
    condition_groups = []
    for identity, group in _groups(
        evaluation, [*IDENTITY, "defense_id", "condition", "split", "transfer_mode"]
    ):
        attacked = [r for r in group if r["condition"] in {"C1", "C2"}]
        condition_groups.append(
            {
                **identity,
                "n": len(group),
                "n_tasks": len({r["task_id"] for r in group}),
                "attack_success_rate": _rate(attacked, "attack_success"),
                "utility_success_rate": _rate(group, "utility_success"),
            }
        )
    comparisons = _defense_comparisons([r for r in evaluation if not provenance.check(r)])
    matrices = _matrices(evaluation, provenance)
    matrices["defense"]["source_selected_transfer"] = _source_selected_defense_transfer(
        comparisons, evaluation, provenance
    )
    longitudinal = []
    for identity, group in _groups(
        evaluation,
        [
            "model_family",
            "model_id",
            "model_alias",
            "scaffold_id",
            "defense_id",
            "condition",
            "split",
            "transfer_mode",
        ],
    ):
        longitudinal.append(
            {
                **identity,
                "version": identity["model_id"],
                "attack_success_rate": _rate(
                    [r for r in group if r["condition"] in {"C1", "C2"}], "attack_success"
                ),
                "utility_success_rate": _rate(group, "utility_success"),
            }
        )
    warnings = list(publication_issues)
    if exclusions:
        warnings.append(
            f"Excluded {sum(exclusions.values())} errored/invalid episodes from outcome analysis."
        )
    if len(valid) != len(evaluation):
        warnings.append(
            f"Excluded {len(valid) - len(evaluation)} screening/selection episodes from evaluation metrics (spend retained)."
        )
    summary: dict[str, Any] = {
        "analysis_schema_version": "1.0",
        "run_id": manifest.get("run_id")
        if isinstance(manifest.get("run_id"), str)
        else run_dir.name,
        "publishable": not publication_issues,
        "simulated": simulated,
        "warnings": warnings,
        "counts": {
            "total": len(rows),
            "valid": len(valid),
            "evaluation": len(evaluation),
            "excluded": sum(exclusions.values()),
            "exclusions": dict(exclusions),
            "non_evaluation": len(valid) - len(evaluation),
            "attack_trials": sum(r["condition"] in {"C1", "C2"} for r in evaluation),
            "clean_trials": sum(r["condition"] in {"C0", "C3"} for r in evaluation),
            "simulated": sum(r.get("simulated") is True for r in rows),
        },
        "methods": {
            "bootstrap_unit": "task_id (all repeats and paired observations intact)",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "rate_ci": "Wilson 95%; task-cluster bootstrap CI also provided",
            "comparison_ci": "paired task-cluster percentile bootstrap 95%; undefined draws counted and omitted",
            "undefined": "JSON null; no imputation or clipping",
            "matching": "exact task_id, seed, repeat, split, transfer_mode, frozen artifact/selection where applicable; ambiguous matches excluded",
            "uads": "RRR - 0.5 * (C0 utility - C3 utility), raw, on joint matched support",
            "longitudinal": "model_family grouped by exact model_id/version; no inferred release chronology or version causality",
        },
        "file_integrity": integrity,
        "research_limitations": [
            "File hashes verify consistency with the manifest, not provider honesty or an independently signed provenance chain; external source-selection runs are not reverified here.",
            "Family co-failure compares target-specific frozen attack variants; it cannot isolate same-payload transfer or the effect of variant generation.",
            "Defense-origin labels alone are declarations. When manifest.defense_selection exists, its hashed snapshot and accessible source evidence are verified; unmatched origin labels cannot pass publication. No cross-source RRR gap is inferred from target effectiveness alone.",
            "Aggregate transfer scores are descriptive macro means over fixed supported cells. Task bootstrap is shared across cells; draws losing any cell denominator are omitted and counted. Sparse tasks/support and selection uncertainty limit inference.",
        ],
        "provenance_failures": dict(provenance_failures),
        "groups": condition_groups,
        "defense_comparisons": comparisons,
        "matrices": matrices,
        "spending": _spending(rows, {id(r) for r in valid}),
        "cost_summary": _cost_summary(rows, valid, manifest),
        "longitudinal": longitudinal,
        "tables": _tables(matrices, comparisons),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary
