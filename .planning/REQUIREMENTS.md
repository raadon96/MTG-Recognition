# Requirements

_Milestone: MTG Recognition — Project Setup & Tooling_

## v1 Requirements

### Dataset Versioning (DVC)

- [ ] **DVC-01**: DVC is initialized in the repository (`dvc init`)
- [ ] **DVC-02**: A Google Drive remote is configured as the DVC remote storage
- [ ] **DVC-03**: `benchmarks/dataset/` is removed from git tracking and deleted from git history tracking (without history rewrite — gitignore + rm --cached)
- [ ] **DVC-04**: `benchmarks/dataset/` content is moved into `data/test/6_benchmark/`
- [ ] **DVC-05**: `data/test/` is tracked by DVC (`dvc add data/test`)
- [ ] **DVC-06**: Dataset is pushed to Google Drive remote (`dvc push`)
- [ ] **DVC-07**: Collaborator can run `dvc pull` to download all test/benchmark datasets
- [ ] **DVC-08**: `dvc push` workflow is documented so contributors can share new test images

### Reference Data Sync (mtg sync)

- [ ] **SYNC-01**: A `mtg sync` command exists that rebuilds MobileCache from latest MTGJSON data
- [ ] **SYNC-02**: `mtg sync` downloads reference card images that are missing from local (incremental — skip existing files)
- [ ] **SYNC-03**: `mtg sync` rebuilds the full pHash index after image download completes
- [ ] **SYNC-04**: `mtg sync` exits with a non-zero code and a clear error message if any step fails

### Unified CLI

- [ ] **CLI-01**: `mtg recognize` command runs the full recognition pipeline (wraps `pipeline.py` main)
- [ ] **CLI-02**: `mtg segment` command runs card segmentation (wraps `segmentation/cli.py` main)
- [ ] **CLI-03**: `mtg build-index` command rebuilds the pHash index standalone (wraps `PHashCardMatcher.build_index`)
- [ ] **CLI-04**: `mtg sync` is declared as a `[project.scripts]` entry point in `pyproject.toml`
- [ ] **CLI-05**: All `mtg` subcommands are declared as `[project.scripts]` entry points in `pyproject.toml`
- [ ] **CLI-06**: Existing `uv run python -m ...` invocations continue to work unchanged (additive only)

### Documentation

- [ ] **DOC-01**: README or CONTRIBUTING.md explains the first-time setup sequence: `dvc pull`, `mtg sync`
- [ ] **DOC-02**: README documents all `mtg` CLI commands with usage examples
- [ ] **DOC-03**: DVC push workflow is documented for contributors adding new test images

## v2 Requirements (deferred)

- Automated/scheduled MobileCache updates via GitHub Actions cron — manual trigger is sufficient for quarterly MTG releases
- Incremental pHash index rebuild (only new images) — full rebuild is acceptable for now
- DVC tracking of `data/reference/` (MobileCache + pHash index) — collaborators rebuild from source

## Out of Scope

- CI/CD pipeline changes — no automated triggers; all commands are manual
- Distributing ~1M reference card images via DVC — too large; collaborators download from Scryfall via `build_database.py`
- Git history rewrite to remove benchmarks/ from past commits — gitignore + `git rm --cached` going forward is sufficient
- New cloud storage costs — Google Drive free tier only

## Traceability

| Req ID | Phase |
|--------|-------|
| DVC-01 – DVC-08 | Phase 1 |
| SYNC-01 – SYNC-04 | Phase 2 |
| CLI-01 – CLI-06 | Phase 2 |
| DOC-01 – DOC-03 | Phase 3 |
