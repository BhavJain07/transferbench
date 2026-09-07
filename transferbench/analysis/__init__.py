"""TransferBench analysis API (runner integrity verification is imported lazily).

Parent integration::

    from transferbench.analysis import analyze_run, generate_report
    summary = analyze_run(run_dir, publishable=False)  # writes summary.json
    report_path = generate_report(run_dir)            # also refreshes summary

Input: ``run_dir/episodes.parquet`` contains scalar Episode fields; nested
transcripts/tool_calls/call_costs/events may be JSON strings and are not loaded.
``manifest.json`` must be an object. Publication fails closed unless publishable
is literally true, run_id matches, every episode explicitly identifies whether
it is simulated, no simulation is present, and all attack provenance verifies.
See ``analysis.provenance`` for frozen artifact / source-selection record keys.
Unverified attacks remain in descriptive ASRs but cannot enter transfer cells
or matched defense comparisons. Explicit scalar identities/matching keys are
required for publication, as are runner file_hashes covering episodes and frozen
input snapshots. runner.manifest.verify_run hashes the actual recorded files;
snapshots must agree with manifest metadata. Publication also honors runner
blockers and commit/dirty/dependency/lock/version attestations. A replaceable
manifest is not independent proof of provider honesty or selection integrity.

Summary: counts/exclusions, per-condition groups, matched defense_comparisons,
model/scaffold/attack/defense matrices (labels, strata, cells, null-filled grids),
spending, longitudinal family/version groups, and report tables. ASR estimates
contain n, n_tasks, successes, mean, Wilson ci95 and task bootstrap_ci95. Paired
comparisons contain mean, ci95, n, n_tasks and bootstrap_valid. Bootstrap is
reproducible (seed 0, 2000 task-cluster draws); missing support is JSON null.
Spending includes failed/selection episodes and reports valid counts for each
cost field; entirely unknown costs are null, not zero. Reused model aliases are
version-qualified in model matrices rather than silently pooling model IDs.

Model and scaffold cells either measure held-out source-optimized target ASR
(with paired target/source transfer_ratio) or explicitly conditional co-failure
in fixed/family mode. No selection-source ASR is inferred for hand-authored
attacks. Family model/scaffold cells match family/task/seed/repeat, allowing
per-target frozen variants with different IDs/hashes; they are NOT same-artifact
transfer. Ambiguous variants are excluded. Attack-axis cells compare fixed/family
attack IDs on matched trials; defense-ID cells remain residual co-failure.

Each matrix cross_transfer_score now includes a joint task-bootstrap CI; model
scores are named MTS. cross_transfer_by_stratum avoids pooling nuisance strata.
Aggregate n counts pair memberships, not independent episodes; all point-supported
cells must retain denominators in a bootstrap draw or that draw is undefined.

matrices['defense']['source_selected_transfer'] is a separate source/target
model_id+scaffold matrix of RRR_T on matched target C1/C2 pairs. It requires both
defense_source fields and an unambiguous source model. Origin labels alone are
declarations. The runner's optional frozen defense_selection is checked against
its hashed snapshot and accessible source evidence for publication; unmatched
origin labels cannot pass publication. No RRR_S or generalization gap is invented.

C2/C1 RRR and C0/C3 utility tax use one-to-one matched pairs. UADS is raw
RRR - .5 * absolute utility tax on joint matched support. Zero baseline is
undefined and adverse RRR/ratios greater than one are retained. HTML embeds
Plotly once, requires no CDN, and escapes untrusted labels and embedded JSON.
"""

from transferbench.scorers.transfer import (
    attack_success_rate,
    cross_transfer_score,
    raw_transfer,
    relative_risk_reduction,
    transfer_ratio,
    utility_adjusted_defense_score,
)

from .core import analyze_run
from .report import generate_report

__all__ = [
    "analyze_run",
    "generate_report",
    "attack_success_rate",
    "raw_transfer",
    "transfer_ratio",
    "cross_transfer_score",
    "relative_risk_reduction",
    "utility_adjusted_defense_score",
]
