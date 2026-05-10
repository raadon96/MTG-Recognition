---
phase: 01-dataset-versioning
type: walking-skeleton
locked_decisions:
  - dvc-as-data-layer
  - gdrive-as-remote
  - personal-oauth-no-service-account
  - data-test-as-single-dvc-root
  - no-history-rewrite
  - secret-separation-via-config-local
  - no-reference-images-in-dvc
---

# Walking Skeleton: Dataset Versioning End-to-End

This document records the architectural decisions made when scaffolding the thinnest possible end-to-end DVC slice in Phase 1. Subsequent phases (CLI & Sync, Documentation) build on this skeleton without renegotiating these decisions.

## Phase Goal as User Story

**As a** collaborator who just cloned the repo,
**I want to** run a single `uv run dvc pull` command after `uv sync --dev`,
**so that** I receive every test image and benchmark dataset locally without coordinating manually with the project owner.

## Skeleton Slice (thinnest end-to-end stack)

The Walking Skeleton for Phase 1 is the smallest set of components that makes the user story TRUE end-to-end:

```
Developer machine
    |
    +-- pyproject.toml [dependency-groups] dev
    |       |
    |       +-- dvc[gdrive]>=3.67.1                     <-- Plan 01 Task 1
    |               |
    |               v   uv sync --dev
    |           dvc CLI on PATH
    |
    +-- .dvc/                                            <-- Plan 01 Task 2
    |       +-- config              (committed: remote 'origin' + gdrive_client_id)
    |       +-- config.local        (gitignored: gdrive_client_secret)
    |       +-- .gitignore          (auto: /config.local, /tmp, /cache)
    |
    +-- .dvcignore                                       <-- Plan 01 Task 2
    |
    +-- data/test/                                       <-- on disk (gitignored)
    |       +-- 0_error/ ... 5_scans_*/  (existing 7 categories)
    |       +-- 6_benchmark/             <-- migrated from benchmarks/  (Plan 02 Task 1)
    |               +-- images/raw/   (36 JPGs)
    |               +-- images/ready/ (36 JPGs)
    |               +-- ocr/          (20 JPGs)
    |
    +-- data/test.dvc                                    <-- Plan 02 Task 3 (committed pointer)
    |       (outs: with MD5 hash + size)
    |
    +-- data/.gitignore                                  <-- Plan 02 Task 3 (auto: /test)
    |
    +-- .gitignore                                       <-- Plan 02 Task 2 (benchmarks/ added)
    |
    +-- CONTRIBUTING.md                                  <-- Plan 03 Task 2 (workflow docs)
            |
            v   dvc push (Plan 02 Task 5)
        Google Drive: My Drive/decksray/MTG-Recognition/data/test/
            (hash-named blobs, reconstructed on pull)
            |
            v   dvc pull (Plan 03 Task 1 verifies)
        Collaborator's data/test/   (byte-identical reconstruction)
```

## Architectural Decisions (Locked for Future Phases)

These decisions are made once during Phase 1 and inherited unchanged by Phase 2 + 3 (and beyond).

### Data Layer

**Tool:** DVC (Data Version Control) 3.67.1+
**Why:** Industry standard for ML data versioning; integrates with git via pointer files; supports many remote backends including Google Drive; deduplicates by content hash; partial pulls.
**Alternatives rejected:** Custom S3/Drive sync scripts (reinvents pointer-files + dedup), git-lfs (charges for bandwidth, no Google Drive backend), DVC with local-only remote (defeats collaboration).

### Remote Storage

**Backend:** Google Drive via `dvc-gdrive` plugin
**URL:** `gdrive://root/decksray/MTG-Recognition/data/test`
**Folder hierarchy in My Drive:** must exist manually before first push (RESEARCH Pitfall 2).
**Why:** Free for small teams (15 GB tier); decision locked during `/gsd-discuss-phase` (D-09).
**Alternatives rejected:** S3 / Azure / GCS (cost), GitHub LFS (bandwidth metered), DVC HTTP remote (requires hosting a server).

### Authentication

**OAuth model:** Personal Google accounts via custom GCP OAuth Desktop app
**Why required:** DVC's default OAuth app is blocked by Google ("This app is blocked") since late 2023 — every developer must create their own GCP project + Desktop OAuth app (RESEARCH Pitfall 1).
**Token cache location:** `~/.cache/pydrive2fs/<client_id>/default.json` (Linux/Mac); per-developer.
**Service account:** Not used (D-09). Acceptable for a small team where each developer can do one-time OAuth.

### Secret Separation

**Public field** (`gdrive_client_id`) -> committed to `.dvc/config`.
**Secret field** (`gdrive_client_secret`) -> written via `dvc remote modify --local origin ...` -> stored in `.dvc/config.local`, which is auto-gitignored by `dvc init` (RESEARCH Pitfall 3).
**Verification gate:** `! grep -F 'gdrive_client_secret' .dvc/config` enforced in Plan 01 Task 4 acceptance criteria.

### Single DVC Root

**Tracked path:** `data/test/` (one `dvc add data/test`, one `data/test.dvc` pointer file).
**Why:** A single pull command for collaborators (D-06). All seven existing test categories plus the new `6_benchmark/` are tracked together.
**Not tracked by DVC:** `data/reference/` (~1M card images, multiple GB). Out of scope per REQUIREMENTS.md "out of scope" list — collaborators rebuild reference images via `build_database.py` in Phase 2.

### Git History Strategy

**Strategy:** Gitignore-going-forward; NO history rewrite (D-05).
**Implementation:** `git rm --cached benchmarks/` removes from current index only; old commits still contain the 73 benchmark JPGs.
**Acceptable because:** The benchmark JPGs are non-sensitive MTG card photos (publicly available); the 73 files at ~200-500 KB each represent a small, bounded git size penalty. The team accepted this tradeoff during `/gsd-discuss-phase` to avoid the complexity and rebase-pain of `git filter-branch` / `git filter-repo`.

### Migration Sequence (Safe Ordering)

**Order (Plan 02 Tasks 1-6, per RESEARCH Pattern 1):**
1. Move files (`mv benchmarks/dataset/{images,ocr} -> data/test/6_benchmark/`).
2. Untrack from git (`git rm -r --cached benchmarks/` + update `.gitignore`).
3. DVC-track (`dvc add data/test`).
4. Push to remote (`dvc push`).
5. Commit DVC metadata + `.gitignore` updates.

**Why this order:** Files moved before `git rm --cached` ensures zero data-loss window. If interrupted between 1 and 2, files exist on disk (in either old or new location); if interrupted between 2 and 5, files are still on disk and DVC cache, just not committed yet.

## What This Skeleton Enables (Future Phases Build on This)

### Phase 2 (CLI & Sync Command) inherits:
- `data/test/` is the canonical location for test images (no path renegotiation).
- DVC pull is the canonical first-time-setup step (referenced from `mtg sync` docs).
- Custom GCP OAuth setup is documented in CONTRIBUTING.md (Phase 2 doc work just adds CLI examples on top).
- `[dependency-groups] dev` is the canonical place for tooling deps (`mtg` CLI may go in `[project.scripts]`).

### Phase 3 (Documentation) inherits:
- CONTRIBUTING.md is the canonical contributor doc (already exists from Phase 1 with `## Dataset Setup` and `## Adding New Test Images` sections — Phase 3 just adds CLI usage sections and links to it from README).

## Out of Scope for This Skeleton (Explicit Non-Decisions)

- **Reference images in DVC** (~1M card images, multiple GB). Use `build_database.py` to rebuild locally.
- **CI-driven `dvc push`/`pull`.** All DVC operations are manual.
- **Service account authentication.** Personal accounts only.
- **Git history rewrite.** Old commits keep the 73 benchmark JPGs.
- **Automated MobileCache updates.** Manual quarterly trigger via `mtg sync` (Phase 2).

## Verification That the Skeleton Is Working

The Walking Skeleton is verifiable end-to-end by Plan 03 Task 1 (empty-cache `dvc pull` round-trip). After Phase 1 ships:
- A new collaborator clones, runs `uv sync --dev`, sets up GCP OAuth + `dvc remote modify --local`, runs `dvc pull`, and gets all 971 files restored.
- A contributor adds new test images, runs `dvc add data/test && dvc push && git commit data/test.dvc`, and others see their additions on next `dvc pull`.

Both flows are documented in `CONTRIBUTING.md` (Plan 03 Task 2) and verifiable by grep on the file content.
