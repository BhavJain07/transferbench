# Security policy

TransferBench is pre-1.0 research software. Security reports are welcome for the current default branch; historical experiment snapshots are preserved as evidence and are not a promise of maintained releases.

## Reporting a vulnerability

Use the repository's private vulnerability reporting option on GitHub's **Security** tab if it is enabled. If private reporting is unavailable, open a minimal issue asking for a private reporting channel. Do not include exploit details, credentials, private datasets, or sensitive transcripts in that public request.

A useful private report includes:

- The affected commit, Python version, and dependency versions.
- The component and expected trust boundary.
- A minimal reproduction using synthetic data and fake credentials.
- The observed impact and whether a model/provider call was necessary.
- Relevant redacted logs, with original sensitive evidence kept private.

There is no published response-time guarantee or bug-bounty program.

## What counts as a security issue?

An attack causing an intentionally vulnerable fake model to perform a forbidden **synthetic** action is expected benchmark behavior. It is not, by itself, a vulnerability in the harness.

Harness issues worth reporting include:

- Model-controlled text reaching real network, account, shell, or filesystem operations outside the authorized synthetic interface.
- A bypass of deterministic tool-policy checks, artifact verification, or the model-call budget boundary.
- Secrets unexpectedly included in published artifacts or reports.
- Script execution from untrusted report labels or transcript content.
- A scoring or provenance bug that turns invalid evidence into a publishable safety claim.

Unexpected benchmark outcomes and methodological disagreements can normally be reported as ordinary issues using synthetic examples.

## Trust boundaries and operating guidance

- The built-in workspace mutates in-memory synthetic state. This isolates benchmark actions; it is **not an OS sandbox for arbitrary Python code**. Treat plugins, configuration, installed packages, and code you execute as trusted inputs requiring review.
- Built-in model tools do not provide real email, payment, shell, or host-file operations. Do not replace them with production integrations when running attack payloads.
- Provider evaluations send synthetic prompts to the selected external provider. Check provider data-retention policies before using any non-synthetic input.
- Keep API keys in your local environment or an untracked `.env` file. Do not place credentials in registry URLs, task documents, configuration committed to Git, or issue attachments.
- Inspect logs and reports are research artifacts, not a general-purpose secret-redaction system. Review artifacts before sharing them.
- The hard local budget ceiling applies to configured conservative cost estimates. Configure a provider-side spending cap for an absolute invoice limit.
- SHA-256 manifests detect changes relative to recorded evidence. They are not independently signed proof of authorship, trustworthy provider behavior, or immutable model versions.

See [the reproducibility and cost contract](docs/reproducibility.md) for additional operating limits.
