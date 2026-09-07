# Contributing to TransferBench

TransferBench studies whether safety findings generalize across models and agent architectures. Contributions should preserve controlled comparisons, deterministic ground truth, and the boundary between simulated demonstrations and empirical results.

## Local setup

Use Python 3.12 and the checked-in dependency lock:

```sh
uv sync --frozen --python 3.12
make help
make plan
make check
```

Core tests use deterministic models and need no provider credentials. Dependency installation requires network access; the offline evaluation itself does not. To exercise the optional, pinned AgentDojo package integration:

```sh
make test-agentdojo
```

Run `make smoke` for an end-to-end four-scaffold demonstration, or `make reproduce` for the smaller two-scaffold subset with an HTML report. Both use fake models and incur no API charges.

## Choosing the right extension point

| Change | Location | Keep separate from |
| --- | --- | --- |
| Task fixtures and assertions | `transferbench/tasks/`, `tasks/` | Model prompts containing grader answers |
| Synthetic state and permissions | `transferbench/environments/` | Real accounts, external tools, host filesystem access |
| Attack placement and artifacts | `transferbench/attacks/` | Provider-specific generation code |
| Defense transforms | `transferbench/defenses/` | Scaffold-specific defense branches |
| Inspect solver/agent adapters | `transferbench/scaffolds/` | Scoring logic |
| Experimental design and provenance | `transferbench/runner/` | Unrecorded selection or target-specific changes |
| Metrics and reports | `transferbench/scorers/`, `transferbench/analysis/` | Silently pooled experimental strata |

Reuse Inspect's provider, tool, agent, and logging APIs. A new provider should normally be a registry entry, not a custom API wrapper.

## Tests and evidence

A behavioral change should include a focused regression test. Useful checks include:

- Clean tasks remain clean when no attack is present.
- Expected facts are meaningful, unique under grading normalization, and kept out of model-facing grader hints.
- Blocked tool attempts differ from executed unauthorized effects.
- C0/C3 utility comparisons and C1/C2 attack comparisons retain matched support.
- Frozen artifacts remain unchanged across targets in fixed/source-optimized mode.
- Selection trials do not reappear in held-out evaluation.
- Every model call, including monitors and workers, passes through the shared budget boundary.
- Failed and incomplete episodes retain their evidence and estimated costs without entering valid-trial denominators.

Do not add paid calls to the default test suite. Optional upstream integration tests should skip explicitly when their dependency is absent rather than silently substitute mocked upstream evidence.

## Before opening a pull request

```sh
make check
make build
```

Include a short explanation of the change, tests run, and any change to scoring or experimental interpretation. If a dependency changes, update `pyproject.toml` and `uv.lock` together. Keep commits focused on the behavior they introduce or fix.

Never attach API keys, private conversations, real customer data, or unredacted provider logs. For research results, retain canonical Inspect logs and the run manifest, state the model versions and sample sizes, and label simulated outputs clearly. Do not edit archived manifests to remove publication blockers or make historical evidence look reproducible.

See [methodology](docs/methodology.md), [reproducibility](docs/reproducibility.md), and [security reporting](SECURITY.md) for the detailed contracts.
