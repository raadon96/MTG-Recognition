# Roadmap: MTG Recognition — Project Setup & Tooling

## Overview

This milestone transforms a working but hard-to-reproduce ML project into one any collaborator can set up reliably. Phase 1 moves the benchmark dataset out of git into DVC-tracked storage. Phase 2 delivers the unified `mtg` CLI and the `mtg sync` command that rebuilds the full reference data stack in one shot. Phase 3 documents everything so a new collaborator can follow a written guide from zero to running recognition.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Dataset Versioning** - Move benchmark data out of git into DVC with Google Drive remote
- [ ] **Phase 2: CLI & Sync Command** - Deliver unified `mtg` CLI and one-command reference data sync
- [ ] **Phase 3: Documentation** - Write collaborator setup guide covering DVC pull, sync, and CLI usage

## Phase Details

### Phase 1: Dataset Versioning
**Goal**: All test and benchmark image data is tracked by DVC with a Google Drive remote, and no large files remain in git tracking
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: DVC-01, DVC-02, DVC-03, DVC-04, DVC-05, DVC-06, DVC-07, DVC-08
**Success Criteria** (what must be TRUE):
  1. `dvc init` has been run and `.dvc/` is committed to git
  2. A Google Drive remote is configured and `dvc push` has completed without error
  3. `benchmarks/dataset/` no longer appears in `git ls-files`; its contents are present under `data/test/6_benchmark/`
  4. A collaborator who clones the repo and runs `dvc pull` receives all test and benchmark images locally
  5. `git status` shows no large image files untracked or tracked in git after the migration
**Plans**: TBD

### Phase 2: CLI & Sync Command
**Goal**: A collaborator can invoke any pipeline operation via a short `mtg <subcommand>` command, and `mtg sync` rebuilds the full reference data stack (MobileCache → images → pHash index) in one shot
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, CLI-06, SYNC-01, SYNC-02, SYNC-03, SYNC-04
**Success Criteria** (what must be TRUE):
  1. `mtg recognize`, `mtg segment`, `mtg build-index`, and `mtg sync` are all available after `uv sync` with no extra install steps
  2. `mtg sync` runs to completion and leaves MobileCache.sqlite, reference images, and the pHash index in a consistent state
  3. `mtg sync` exits non-zero with a descriptive error message if any step (DB build, image download, index rebuild) fails
  4. Existing `uv run python -m mtg_recognition.engine.pipeline` and `uv run python -m mtg_recognition.engine.segmentation.cli` invocations still work unchanged
**Plans**: TBD

### Phase 3: Documentation
**Goal**: A new collaborator can follow a written guide to go from a fresh clone to a fully working development environment without asking anyone for help
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: DOC-01, DOC-02, DOC-03
**Success Criteria** (what must be TRUE):
  1. README or CONTRIBUTING.md contains a first-time setup section that covers `dvc pull` and `mtg sync` in order
  2. All four `mtg` subcommands are documented with at least one usage example each
  3. Contributors can find clear instructions for running `dvc push` to share new test images
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Dataset Versioning | 0/? | Not started | - |
| 2. CLI & Sync Command | 0/? | Not started | - |
| 3. Documentation | 0/? | Not started | - |
