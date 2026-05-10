# MTG Recognition — Project Setup & Tooling Milestone

## What This Is

MTG Recognition is a Python card-recognition pipeline that takes a photo of a Magic: The Gathering card, segments it, extracts text via OCR, and matches it against a ~1M-card SQLite database. It is served as a Streamlit web app via Docker.

This milestone addresses four infrastructure and developer-experience problems that make it hard for collaborators to reliably reproduce the development environment:

1. Test/benchmark image datasets are either git-tracked (bloating the repo) or not tracked at all.
2. The MobileCache SQLite database (card metadata) must be manually rebuilt and is not versioned or distributed.
3. When MobileCache is updated, the reference images and pHash index can drift out of sync.
4. Running module entry points requires a verbose `uv run python -m <module>` invocation with no unified CLI.

## Core Value

Every collaborator can reproduce the full development environment — datasets, reference database, and pHash index — with a predictable sequence of commands, without manual coordination.

## Requirements

### Validated

_Capabilities already working in the existing codebase:_

- ✓ Two-stage card recognition pipeline (YOLO segmentation → pHash + OCR matching) — existing
- ✓ Streamlit web app serving the pipeline — existing
- ✓ Docker-based deployment — existing
- ✓ Pipeline CLI (`uv run python -m mtg_recognition.engine.pipeline`) — existing
- ✓ Segmentation CLI (`uv run python -m mtg_recognition.engine.segmentation.cli`) — existing
- ✓ MobileCache build job (`build_database.py`): downloads MTGJSON, builds SQLite, downloads reference images (incremental — skips existing files) — existing
- ✓ pHash index builder (`PHashCardMatcher.build_index`) — existing

### Active

_What this milestone delivers:_

- [x] DVC initialized in the repo with a Google Drive remote configured — Validated in Phase 01 (2026-05-10)
- [x] `data/test/` fully tracked by DVC (no large files in git) — Validated in Phase 01 (2026-05-10)
- [x] `benchmarks/dataset` moved into `data/test/6_benchmark` and removed from git — Validated in Phase 01 (2026-05-10)
- [x] Collaborators can run `dvc pull` to get all test/benchmark datasets — Validated in Phase 01 (2026-05-10)
- [x] `dvc push` workflow documented so contributors can share new test images — Validated in Phase 01 (2026-05-10)
- [ ] `mtg sync` command: one command that runs MobileCache rebuild → downloads missing reference images → rebuilds full pHash index
- [ ] `mtg recognize` command: wraps the pipeline CLI
- [ ] `mtg segment` command: wraps the segmentation CLI
- [ ] `mtg build-index` command: standalone pHash index rebuild
- [ ] All `mtg` subcommands declared as `[project.scripts]` in `pyproject.toml`
- [ ] Setup documentation updated: README or CONTRIBUTING.md explains DVC pull, `mtg sync`, and CLI usage

### Out of Scope

- Automated / scheduled MobileCache updates (CI cron job) — manual trigger is sufficient; add later if needed
- Incremental pHash index rebuild (only new images) — full rebuild is acceptable; optimize if build time becomes painful
- Distributing reference images (~1M card images) via DVC — collaborators download from Scryfall via `build_database.py`; only test datasets go in DVC
- Storing MobileCache.sqlite in DVC — the sync command rebuilds it from MTGJSON on demand; no versioned binary tracking needed

## Context

- **Language / runtime**: Python 3.12+, managed with `uv`
- **Existing dataset layout**: `data/test/` (manual test images, 7 scenario categories), `benchmarks/dataset/` (benchmark images with UUID filenames + OCR ground truth), both are ~550 MB combined
- **Reference data**: `data/reference/` contains MobileCache.sqlite (~35 MB), ~1M card images, and the precomputed pHash numpy index — none is in git
- **Collaborators**: Small team; authentication via personal Google accounts is acceptable for DVC remote
- **pHash index size**: Covers ~1M cards. Full rebuild is accepted for now
- **Incremental image download**: Already implemented in `download_file_with_retry` (skips files that already exist on disk)

## Constraints

- **No CI/CD changes**: All automation is local / manual-trigger only
- **No new cloud costs**: Google Drive (free tier) for DVC remote
- **Backward compatibility**: Existing `uv run python -m ...` invocations must continue to work; new CLI is additive
- **No git history rewrite**: Remove large files from git tracking going forward, not via history rewrite

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| DVC + Google Drive remote | Free, simple auth for small team, DVC handles versioning | — Pending |
| `data/test/` as single DVC root | Unified pull command; cleaner than multiple dvc remotes | — Pending |
| Move benchmarks/ into data/test/6_benchmark | Eliminate top-level benchmarks/ dir; one DVC root for all image datasets | — Pending |
| `mtg sync` as manual one-command update | CI scheduling is overkill for quarterly MTG releases | — Pending |
| Full pHash index rebuild on sync | Simpler code; acceptable latency for infrequent manual runs | — Pending |
| `[project.scripts]` in pyproject.toml | Standard uv/pip mechanism; no extra tooling | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-10 — Phase 01 (dataset-versioning) complete*
