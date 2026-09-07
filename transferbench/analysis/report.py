"""Self-contained HTML reports: five figures, one inline Plotly runtime, no CDN."""

import html
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from .core import analyze_run


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _number(value: float | None) -> str:
    return "unsupported" if value is None else f"{value:.4g}"


def _estimate(estimate: dict | None) -> str:
    if not estimate:
        return "unsupported"
    low, high = estimate.get("ci95", [None, None])
    return (
        f"{_number(estimate.get('mean'))} [{_number(low)}, {_number(high)}]; "
        f"n={estimate.get('n', 0)}, tasks={estimate.get('n_tasks', 0)}"
    )


def _empty(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(title=title, template="plotly_white")
    figure.add_annotation(
        text="No supported matched observations",
        showarrow=False,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )
    return figure


def _heatmap(matrix: dict, title: str) -> go.Figure:
    supported = [
        s for s in matrix["strata"] if any(c["stratum_id"] == s["id"] for c in matrix["cells"])
    ]
    if not supported:
        return _empty(title)
    figure = go.Figure()
    labels = [_escape(label) for label in matrix["labels"]]
    buttons = []
    for index, stratum in enumerate(supported):
        lookup = {
            (c["source"], c["target"]): c
            for c in matrix["cells"]
            if c["stratum_id"] == stratum["id"]
        }
        z, hover = [], []
        for source in matrix["labels"]:
            z_row, hover_row = [], []
            for target in matrix["labels"]:
                cell = lookup.get((source, target))
                z_row.append(cell["estimate"]["mean"] if cell else None)
                hover_row.append(
                    _escape(
                        f"{cell['kind']}: {_estimate(cell['estimate'])}; matched pairs={cell['n_pairs']}"
                        if cell
                        else "Unsupported: no unambiguous matched trials"
                    )
                )
            z.append(z_row)
            hover.append(hover_row)
        figure.add_trace(
            go.Heatmap(
                x=labels,
                y=labels,
                z=z,
                zmin=0,
                zmax=1,
                colorscale="Blues",
                text=hover,
                hovertemplate="Reference/source: %{y}<br>Target: %{x}<br>%{text}<extra></extra>",
                visible=index == 0,
                colorbar={"title": "Rate"},
                hoverongaps=False,
            )
        )
        label = ", ".join(
            f"{key}={value}" for key, value in stratum.items() if key != "id" and value is not None
        )
        buttons.append(
            {
                "label": _escape(label),
                "method": "update",
                "args": [
                    {"visible": [i == index for i in range(len(supported))]},
                    {"title": {"text": _escape(title + " — " + label)}},
                ],
            }
        )
    figure.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Target",
        yaxis_title="Reference / provenance source",
        yaxis={"autorange": "reversed"},
        margin={"t": 130},
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "y": 1.2}],
    )
    return figure


def _error(estimate: dict) -> tuple[float | None, float | None]:
    mean = estimate["mean"]
    low, high = estimate["ci95"]
    if mean is None or low is None or high is None:
        return None, None
    return max(0.0, high - mean), max(0.0, mean - low)


def _defense_figure(comparisons: list[dict], axis: str, title: str) -> go.Figure:
    supported = [r for r in comparisons if r["relative_risk_reduction"]["mean"] is not None]
    if not supported:
        return _empty(title)
    other = "scaffold_id" if axis == "model_alias" else "model_alias"
    figure = go.Figure()
    # Keep model/scaffold strata visible in labels rather than pooling unmatched
    # baselines into a seemingly more precise defense score.
    for defense in sorted({r["defense_id"] for r in supported}):
        group = [r for r in supported if r["defense_id"] == defense]
        estimates = [r["relative_risk_reduction"] for r in group]
        errors = [_error(e) for e in estimates]
        figure.add_trace(
            go.Bar(
                name=_escape(defense),
                x=[
                    _escape(
                        f"{r[axis]} / {r[other]} / {r['model_id']} / {r.get('split', '')} / {r.get('transfer_mode', '')} / origin={r.get('defense_source_model')},{r.get('defense_source_scaffold')}"
                    )
                    for r in group
                ],
                y=[e["mean"] for e in estimates],
                error_y={
                    "type": "data",
                    "array": [e[0] for e in errors],
                    "arrayminus": [e[1] for e in errors],
                },
                text=[_escape(_estimate(e)) for e in estimates],
                hovertemplate="%{x}<br>%{fullData.name}<br>RRR: %{text}<extra></extra>",
            )
        )
    figure.update_layout(
        title=title,
        template="plotly_white",
        barmode="group",
        yaxis_title="RRR (C2 vs matched C1; negative means harm)",
    )
    return figure


def _security_utility(comparisons: list[dict]) -> go.Figure:
    title = "Security–utility trade-off"
    rows = [
        r
        for r in comparisons
        if r["relative_risk_reduction"]["mean"] is not None and r["utility_tax"]["mean"] is not None
    ]
    if not rows:
        return _empty(title)
    figure = go.Figure()
    for defense in sorted({r["defense_id"] for r in rows}):
        group = [r for r in rows if r["defense_id"] == defense]
        xerr = [_error(r["utility_tax"]) for r in group]
        yerr = [_error(r["relative_risk_reduction"]) for r in group]
        figure.add_trace(
            go.Scatter(
                mode="markers",
                name=_escape(defense),
                marker={"size": 11},
                x=[r["utility_tax"]["mean"] for r in group],
                y=[r["relative_risk_reduction"]["mean"] for r in group],
                error_x={
                    "type": "data",
                    "array": [e[0] for e in xerr],
                    "arrayminus": [e[1] for e in xerr],
                },
                error_y={
                    "type": "data",
                    "array": [e[0] for e in yerr],
                    "arrayminus": [e[1] for e in yerr],
                },
                text=[
                    _escape(
                        f"{r['model_alias']} / {r['model_id']} / {r['scaffold_id']}; security {_estimate(r['relative_risk_reduction'])}; utility tax {_estimate(r['utility_tax'])}; joint-support UADS {_estimate(r['uads'])}"
                    )
                    for r in group
                ],
                hovertemplate="%{text}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Utility tax: matched C0 − C3 (lower is better)",
        yaxis_title="RRR: matched C2 vs C1 (higher is better)",
    )
    return figure


def _table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "<p>No supported observations.</p>"
    return (
        "<div class=table-scroll><table><thead><tr>"
        + "".join(f"<th>{_escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
        )
        + "</tbody></table></div>"
    )


def _transfer_rows(rows: list[dict], metric: str = "estimate") -> list[list[object]]:
    return [
        [
            r["dimension"],
            r["kind"],
            r["source"],
            r["target"],
            json.dumps(r["stratum"], sort_keys=True),
            _estimate(r[metric]),
            r["n_pairs"],
        ]
        for r in rows
    ]


def _report_tables(summary: dict) -> str:
    tables = summary["tables"]
    headers = [
        "Dimension",
        "Estimand",
        "Reference/source",
        "Target",
        "Stratum",
        "Mean [95% CI]; n; tasks",
        "Matched pairs",
    ]
    sections = []
    for label, key in [
        ("Most transferable (within stratum)", "most"),
        ("Least transferable (within stratum)", "least"),
    ]:
        rows = [row for ranking in tables["transferability"] for row in ranking[key]]
        sections.append(f"<h2>{label}</h2>" + _table(headers, _transfer_rows(rows)))
    sections.append(
        "<h2>Robust defenses — descriptive ordering by lower RRR CI</h2><p>Compare within model/scaffold and provenance strata; this is not an adjusted cross-population leaderboard.</p>"
        + _table(
            [
                "Model alias / version",
                "Scaffold",
                "Defense / origin",
                "Split / mode",
                "C1 ASR",
                "C2 ASR",
                "RRR",
                "C0 − C3 utility tax",
                "Raw UADS (λ=0.5)",
            ],
            [
                [
                    f"{r['model_alias']} / {r['model_id']}",
                    r["scaffold_id"],
                    f"{r['defense_id']} / {r.get('defense_source_model')},{r.get('defense_source_scaffold')}",
                    f"{r['split']} / {r['transfer_mode']}",
                    _estimate(r["baseline_asr"]),
                    _estimate(r["defended_asr"]),
                    _estimate(r["relative_risk_reduction"]),
                    _estimate(r["utility_tax"]),
                    _estimate(r["uads"]),
                ]
                for r in tables["robust_defenses"]
            ],
        )
    )
    transfer = summary["matrices"]["defense"]["source_selected_transfer"]
    sections.append(
        "<h2>Source-selected defense transfer — declared-origin RRR_T</h2>"
        "<p>Source/target axes are model/scaffold environments, not defense IDs. RRR_T uses matched target C1/C2 trials. Origin fields alone do not prove defense selection or frozen parameters; no source RRR or cross-source gap is inferred.</p>"
        + _table(
            ["Declared source", "Target", "Defense", "Split / mode", "RRR_T [95% CI]; n; tasks"],
            [
                [
                    c["source"],
                    c["target"],
                    c["defense_id"],
                    f"{transfer['strata'][c['stratum_id']]['split']} / {transfer['strata'][c['stratum_id']]['transfer_mode']}",
                    _estimate(c["rrr_t"]),
                ]
                for c in transfer["cells"]
            ],
        )
    )
    aggregates = []
    for dimension in ("model", "scaffold"):
        matrix = summary["matrices"][dimension]
        for aggregate in matrix["cross_transfer_by_stratum"]:
            aggregates.append(
                [
                    matrix["score_name"],
                    aggregate["kind"],
                    json.dumps(matrix["strata"][aggregate["stratum_id"]], sort_keys=True),
                    _estimate(aggregate),
                    aggregate["n_cells"],
                    aggregate["bootstrap_valid"],
                ]
            )
    sections.append(
        "<h2>Aggregate transfer scores — task-bootstrap CIs</h2>"
        "<p>Macro means over fixed supported off-diagonal cells within each stratum. Tasks are drawn jointly across cells; n counts pair memberships, not independent episodes. Draws losing any required denominator are omitted and counted.</p>"
        + _table(
            [
                "Score",
                "Estimand",
                "Stratum",
                "Mean [95% CI]; n; tasks",
                "Cells",
                "Valid draws / 2000",
            ],
            aggregates,
        )
    )
    sections.append(
        "<h2>Generalization gaps — target minus matched reference ASR</h2>"
        + _table(headers, _transfer_rows(tables["generalization_gaps"], "generalization_gap"))
    )
    sections.append(
        "<h2>Scaffold transitions</h2>"
        + _table(headers, _transfer_rows(tables["scaffold_transitions"]))
    )
    sections.append(
        "<h2>Per-condition outcomes</h2>"
        + _table(
            [
                "Model",
                "Version",
                "Scaffold",
                "Defense",
                "Condition",
                "Split / mode",
                "ASR",
                "Utility",
            ],
            [
                [
                    r["model_alias"],
                    r["model_id"],
                    r["scaffold_id"],
                    r["defense_id"],
                    r["condition"],
                    f"{r['split']} / {r['transfer_mode']}",
                    _estimate(r["attack_success_rate"]),
                    _estimate(r["utility_success_rate"]),
                ]
                for r in summary["groups"]
            ],
        )
    )
    costs = summary["cost_summary"]
    sections.append(
        "<h2>Run cost summary</h2>"
        + _table(
            [
                "Total estimated USD",
                "Cost per valid episode USD",
                "Valid n",
                "Ledger charged USD",
                "Conservative planned bound USD",
                "Hard estimate ceiling USD",
                "All costs known",
            ],
            [
                [
                    _number(costs["total_estimated_cost_usd"]),
                    _number(costs["cost_per_valid_episode_usd"]),
                    costs["n_valid"],
                    _number(costs["budget_ledger_charged_usd"]),
                    _number(costs["conservative_maximum_usd"]),
                    _number(costs["hard_estimated_cost_ceiling_usd"]),
                    costs["complete"],
                ]
            ],
        )
    )
    sections.append(
        "<h2>Spending by alias / scaffold (including failed and selection episodes)</h2>"
        + _table(
            [
                "Alias",
                "Model ID",
                "Scaffold",
                "All n",
                "Valid n",
                "Failures / invalid",
                "Estimated USD",
                "Input tokens",
                "Cached tokens",
                "Output tokens",
                "Latency ms",
                "Valid cost n",
                "Invalid cost values",
            ],
            [
                [
                    r["model_alias"],
                    r["model_id"],
                    r["scaffold_id"],
                    r["n"],
                    r["n_valid"],
                    r["n_failures"],
                    _number(r["estimated_cost_usd"]),
                    r["input_tokens"],
                    r["cached_tokens"],
                    r["output_tokens"],
                    _number(r["latency_ms"]),
                    json.dumps(r["cost_valid_n"], sort_keys=True),
                    json.dumps(r["invalid_cost_values"], sort_keys=True),
                ]
                for r in summary["spending"]
            ],
        )
    )
    sections.append(
        "<h2>Longitudinal model-family / version groups</h2><p>Exact model IDs are version labels, not inferred release dates. Unmatched version differences are descriptive, not causal.</p>"
        + _table(
            [
                "Family",
                "Version",
                "Alias",
                "Scaffold",
                "Defense",
                "Condition",
                "Split / mode",
                "ASR",
                "Utility",
            ],
            [
                [
                    r["model_family"],
                    r["version"],
                    r["model_alias"],
                    r["scaffold_id"],
                    r["defense_id"],
                    r["condition"],
                    f"{r['split']} / {r['transfer_mode']}",
                    _estimate(r["attack_success_rate"]),
                    _estimate(r["utility_success_rate"]),
                ]
                for r in summary["longitudinal"]
            ],
        )
    )
    return "\n".join(sections)


def generate_report(run_dir: Path, output: Path | None = None, publishable: bool = False) -> Path:
    """Analyze a run and write HTML (default run_dir/report.html), returning its path.

    Revalidates publication eligibility even if a summary already exists. All
    data labels are HTML escaped, including Plotly text, and JSON embedded in
    script elements is escaped independently to prevent closing-script injection.
    """
    run_dir = Path(run_dir)
    output = Path(output) if output is not None else run_dir / "report.html"
    if output.resolve() in {
        (run_dir / name).resolve() for name in ("episodes.parquet", "manifest.json", "summary.json")
    }:
        raise ValueError("report output must not overwrite run inputs or summary.json")
    summary = analyze_run(run_dir, publishable=publishable)
    figures = [
        _heatmap(summary["matrices"]["model"], "Model transfer (stratified)"),
        _heatmap(summary["matrices"]["scaffold"], "Scaffold transfer (stratified)"),
        _defense_figure(
            summary["defense_comparisons"], "model_alias", "Defense effectiveness by model"
        ),
        _defense_figure(
            summary["defense_comparisons"], "scaffold_id", "Defense effectiveness by scaffold"
        ),
        _security_utility(summary["defense_comparisons"]),
    ]
    figure_html = []
    for index, figure in enumerate(figures):
        serialized = pio.to_json(figure)
        assert isinstance(serialized, str)
        payload = (
            serialized.replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )
        figure_html.append(
            f'<section><div id="figure-{index}" class="figure"></div><script type="application/json" id="data-{index}">{payload}</script></section>'
        )
    warnings = "".join(f"<li>{_escape(warning)}</li>" for warning in summary["warnings"])
    banner = (
        "<div class=warning><strong>SIMULATED DATA — NOT REAL MODEL EVIDENCE. Do not publish as empirical model results.</strong></div>"
        if summary["simulated"]
        else ""
    )
    eligibility = (
        "Eligible under recorded provenance checks"
        if summary["publishable"]
        else "Exploratory only — not publishable"
    )
    document = (
        "<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width, initial-scale=1'>"
        f"<title>TransferBench — {_escape(summary['run_id'])}</title>"
        "<style>body{font:15px system-ui,sans-serif;margin:2rem;line-height:1.5;color:#18202b}h1,h2{line-height:1.2}.warning{background:#fff0c2;border:2px solid #976600;padding:1rem;margin:1rem 0}.figure{min-height:520px}section{margin:2rem 0}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:.5rem;border:1px solid #cbd2db;text-align:left}th{background:#edf1f5}td{overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>"
        # Plotly is embedded exactly once; no external script tags or fetches.
        f"<script id=plotly-runtime>{get_plotlyjs()}</script></head><body>"
        f"<h1>TransferBench — {_escape(summary['run_id'])}</h1>{banner}<p><strong>{eligibility}</strong></p>"
        f"<ul>{warnings}</ul><h2>Included / excluded episodes</h2><pre>{_escape(json.dumps(summary['counts'], indent=2))}</pre>"
        "<h2>Methods and interpretation</h2><p>Rates show n, task count, mean and Wilson 95% CIs; summary.json also contains whole-task bootstrap intervals. Comparative error bars use paired task-cluster 95% intervals, retaining all repeats; seed=0, 2000 draws. Null means unsupported, not zero. Screening/selection outcomes and execution errors are not attack trials. Zero-baseline RRR is undefined; ratios and raw UADS are not clipped.</p>"
        "<p>Transfer figures use separate nuisance/provenance strata. Source-optimized cells are held-out target ASR on the exact selected artifact, matched to held-out source evaluation by task, seed and repeat. Fixed/family cells are conditional co-failure P(target succeeds | reference succeeds), NOT selection-source ASR. Family model/scaffold cells match family/task/seed/repeat across target-specific frozen variants with potentially different hashes: not same-artifact transfer. Attack-axis comparisons change frozen interventions; defense-ID comparisons describe residual co-failure, not source-selected defense transfer.</p>"
        "<p>Defense efficacy compares matched C2 vs C1; utility tax compares matched C0 vs C3. The security–utility plot shows their respective supports. UADS = RRR − 0.5 × utility tax uses their joint matched support. Singleton-task bootstrap intervals are descriptive and can be degenerate; no multiple-comparison correction is applied. Rankings are exploratory.</p>"
        + "<h2>File integrity</h2><pre>"
        + _escape(json.dumps(summary["file_integrity"], indent=2))
        + "</pre>"
        + "<h2>Research limitations</h2><ul>"
        + "".join(f"<li>{_escape(item)}</li>" for item in summary["research_limitations"])
        + "</ul>"
        + "\n".join(figure_html)
        + _report_tables(summary)
        + "<script>for(let i=0;i<5;i++){const f=JSON.parse(document.getElementById('data-'+i).textContent);Plotly.newPlot('figure-'+i,f.data,f.layout,{responsive:true,displaylogo:false});}</script></body></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
