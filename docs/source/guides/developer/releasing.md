# Releasing

How a TorchFX version bump becomes a published release.

## Day-to-day flow

1. **Open a PR that bumps the version.** Edit `pyproject.toml`'s `version` field and add a matching entry to [`CHANGELOG`](https://github.com/matteospanio/torchfx/blob/master/CHANGELOG). Land any release-critical changes alongside the bump or beforehand.
2. **Merge the PR into `master`.** That's it on the human side.

When the merge commit lands on `master`, [`release.yml`](https://github.com/matteospanio/torchfx/blob/master/.github/workflows/release.yml) takes over:

- Reads the new version out of `pyproject.toml`.
- If a tag `v<version>` doesn't exist yet, it creates and pushes one (`v0.5.4`, `v0.6.0`, ...).
- Dispatches [`wheels.yml`](https://github.com/matteospanio/torchfx/blob/master/.github/workflows/wheels.yml) and [`wheels-cuda.yml`](https://github.com/matteospanio/torchfx/blob/master/.github/workflows/wheels-cuda.yml) against the new tag.

The wheel workflows then:

- **`wheels.yml`** — builds CPU wheels (Linux x86_64, macOS Intel + Apple Silicon, Windows x86_64) and the sdist via `cibuildwheel`, then publishes them to **PyPI** via trusted publishing.
- **`wheels-cuda.yml`** — builds Linux x86_64 CUDA wheels (`cu124`, `cu128`) and publishes them to the **GitHub Pages** PEP 503 index at `https://matteospanio.github.io/torchfx/wheels/`.

If you re-merge to `master` without changing the version, `release.yml` is a no-op (the tag already exists).

```{mermaid}
flowchart LR
    PR["Version-bump PR"] --> Merge["Merge to master"]
    Merge --> Release["release.yml<br/>(detects bump,<br/>tags vX.Y.Z)"]
    Release -->|workflow_dispatch| CPU["wheels.yml<br/>cibuildwheel × 4 OSes"]
    Release -->|workflow_dispatch| GPU["wheels-cuda.yml<br/>cu124 + cu128"]
    CPU --> PyPI["PyPI<br/>(trusted publishing)"]
    GPU --> Pages["GitHub Pages<br/>matteospanio.github.io/torchfx/wheels/"]
```

## One-time setup

These steps need to be done once per project, not per release.

### 1. PyPI trusted publishing

On [PyPI's *Manage* page for `torchfx`](https://pypi.org/manage/project/torchfx/settings/publishing/) → **Add a new pending publisher** (or trusted publisher if the project already exists), enter:

| Field | Value |
|-------|-------|
| Owner | `matteospanio` |
| Repository name | `torchfx` |
| Workflow name | `wheels.yml` |
| Environment name | `pypi` |

This authorises GitHub Actions runs of `wheels.yml` (in the `pypi` environment) to upload to PyPI without an API token.

### 2. GitHub `pypi` environment

In the GitHub repo, go to **Settings → Environments → New environment** and create one named `pypi`. Optional but recommended protection rules:

- **Deployment branches and tags**: restrict to `Selected branches and tags` matching `v*` so only tagged builds can publish.
- **Required reviewers**: require manual approval before each publish if the project policy calls for it.

Both protections are enforced by GitHub before the OIDC token is minted, so the publish job is gated even if a workflow change would otherwise allow it.

### 3. GitHub Pages source

`gh-pages` branch is already used by `docs.yml`. The CUDA wheel index deploys to the same branch under `wheels/cuXXX/` with `keep_files: true`, so docs and wheels coexist without conflicts.

Confirm under **Settings → Pages** that the source is "Deploy from a branch → `gh-pages` / `(root)`".

## Manual / out-of-band tagging

If you ever need to publish from a non-master branch, or skip the auto-tag step, push a tag by hand:

```bash
git tag -a v0.5.4 -m "Release v0.5.4"
git push origin v0.5.4
```

That fires the same `push: tags: ["v*"]` triggers on `wheels.yml` and `wheels-cuda.yml` directly, bypassing `release.yml`.

## Troubleshooting

**`release.yml` ran but no wheel build started.**
The auto-tag step uses `GITHUB_TOKEN` to push the tag; tag pushes from `GITHUB_TOKEN` do not retrigger workflows by design. The workflow then dispatches `wheels.yml` and `wheels-cuda.yml` explicitly via `gh workflow run --ref vX.Y.Z`. If those dispatch steps were skipped, check that `actions: write` is in the job's `permissions:` block (it is, in the shipped `release.yml`). Repository or org-level "Settings → Actions → General → Workflow permissions" must also allow workflows to dispatch other workflows.

**PyPI publish step fails with `403 Forbidden` or `Invalid OIDC token`.**
The trusted publisher entry on PyPI has not been created or the (workflow, environment) tuple does not match. Verify the values in [step 1](#1-pypi-trusted-publishing) above match exactly --- the workflow name is `wheels.yml`, the environment name is `pypi`.

**A re-run of `release.yml` after a force-push doesn't tag anything.**
The tag from the previous run is still on the remote. Delete it explicitly (`git push --delete origin vX.Y.Z`) before re-running.
