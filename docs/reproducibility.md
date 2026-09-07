# Reproducibility, budgets and evidence

## Install and validate

Use Python 3.12 and uv. `uv sync --frozen --python 3.12` installs the lockfile, including development tools. `uv sync --frozen --extra providers` enables the real-provider SDKs used through Inspect. The separate `agentdojo` extra pins the native seed exporter. Run `uv run --frozen pytest -q` and `uv run --frozen ruff check .`. Optional native AgentDojo tests skip when its package is absent; core CI never needs API keys.

The CLI reads only the working directory's `.env`, without replacing exported environment variables. Real API credentials belong there or in your environment, never in configuration or tasks. `configs/models.yaml` expands only aliases selected for the run, so offline smoke does not require frontier-model variables. Unknown YAML keys, duplicate mappings, unsupported dimensions and unresolved selected model IDs fail before execution.

Paths in YAML resolve relative to that YAML file. `output_root` CLI options resolve from the current directory. Source-discovery exports snapshot the selected tasks and use resolved paths. Keep source runs and selection artifacts together: source-defense selection deliberately revalidates accessible source evidence instead of trusting an unverifiable origin label.

## Cost contract

Every assistant, manager, worker and monitor generation goes through `GenerationRuntime`, which calls Inspect's `Model.generate` directly. There are no custom provider clients or inference servers.

Before dispatch, the runtime reserves:

```
(max_input_tokens × conservative_input_rate
 + max_output_tokens × conservative_output_rate) / 1,000,000
```

All calls share one locked run ledger. A call that cannot reserve its entire bound is not dispatched. The CLI shows episode counts, worst-case spend and the ceiling before paid execution; unattended runs use `--yes` but cannot bypass the ledger. There is no adaptive concurrency or distributed scheduling. Batches are sequential; up to 64 already scheduled samples may be recorded as budget-stopped without calling a model.

Input admission uses serialized UTF-8 bytes plus a framing allowance, deliberately more conservative than ordinary text tokenization. Output and generation-attempt counts are bounded. Runtime configuration clears inherited request extras, disables caching/retries/parallel tool calls and limits the request to one choice. Unsupported seeds are not sent. This release does not enable multimodal inputs, separately configured reasoning budgets, provider-side tools, best-of sampling or fallback models.

Successful calls reconcile observed tokens with configured prices and reported provider cost when available. Cache tokens are conservatively counted; input rates must cover cache-write premiums and context tiers. Missing/inconsistent usage and failed requests retain their **entire reserved bound** as uncertain estimated spend. Overages stop the run; they are never erased from accounting.

**This is a hard ceiling on configured conservative estimates, not a guarantee about a third party's invoice.** Tokenizer overhead, provider billing rules, SDK/network retries, hidden usage and price changes remain external. Use conservative rates, inspect uncertain-cost flags, and set an account/provider-side spending cap for an absolute monetary limit. No paid rate defaults are invented. A zero-dollar ceiling runs the fake models but prevents any positive-cost real call.

`max_tokens` limits one generation, not the whole episode. `max_turns` limits all generation attempts across the episode, including both workers and the monitor. Failed/incomplete episodes remain in logs and cost totals but not headline ASR denominators.

## Evidence layout

```
results/<run_id>/
  inspect/*.eval            canonical Inspect events, messages, model usage, tools, agents, scores
  inputs/config.json        validated effective configuration
  inputs/models.json        aliases, resolved versions, rates, endpoint metadata (no API keys)
  inputs/tasks.json         exact fixtures and deterministic assertions
  inputs/attacks/*.json     content-hashed immutable artifacts
  inputs/source_selections.json
  inputs/defense_selection.json   when present
  episodes.parquet          typed scalar outcomes; variable nested records encoded as JSON strings
  manifest.json             Git/dependencies/model versions, counts, cost, integrity hashes
  summary.json              derived stratified metrics, uncertainty and tables
  report.html               self-contained interactive report, when requested
```

Artifacts use create-only writes and SHA-256; derived Parquet/JSON writes use atomic replacement. Reusing a run ID fails instead of overwriting evidence. Canonical Inspect sample metadata contains the full episode result and the derived episode, so the projection can be audited against the log. Every raw `.eval` records the local run ID and model alias. Inspect scores show deterministic outcomes, but its unfiltered aggregate should not replace TransferBench's valid-trial analysis.

A process interruption can leave more completed samples in canonical Inspect logs than in the latest batch-level Parquet checkpoint. Preserve both; inspect the raw logs rather than interpreting a partial projection as a completed run. This release does not implement automatic paid-run resume/retry or log-to-Parquet recovery CLI.

## Publication gate

```
uv run transferbench verify results/<run-id>
uv run transferbench analyze results/<run-id> --publishable
uv run transferbench report results/<run-id> --publishable
```

Publishing fails closed unless the run is complete, non-simulated, attributable to a clean committed implementation and a matching dependency lock, and every model is explicitly attested as version-pinned. Missing/failing integrity checks, mismatched input snapshots or unsupported attack provenance also block publication. `version_pinned: true` is a researcher attestation; do not set it on mutable `latest` aliases. Source-selected defenses also retain source evidence and implementation hashes.

These checks enforce consistency, not authorship, independent billing verification, or honesty of a manifest's author. They do not prove a claimed provider snapshot immutable. Archive raw logs, tasks, frozen artifacts, lockfile and exact commit with any paper. Do not publish simulated demonstrations as real model findings.

No commit is created automatically. Initial local runs in this new checkout are expected to be non-publishable until the researcher reviews and commits the implementation. Do not edit manifests to remove this limitation.

## Inspect tooling

Use Inspect's own viewer rather than a separate dashboard:

```
uv run inspect view --log-dir results/<run-id>/inspect
```

This is an interactive long-running viewer; start it yourself when needed. TransferBench's generated HTML reports work offline and embed Plotly once, with no CDN or analytics requests.
