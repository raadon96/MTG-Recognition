---
phase: 01-dataset-versioning
plan: 02
subsystem: data-versioning
tags:
  - dvc
  - migration
  - gdrive
  - benchmark

dependency_graph:
  requires:
    - 01-01  # DVC init + remote configured
  provides:
    - data/test.dvc tracking data/test/ on Google Drive
    - benchmarks/ fully gitignored (73 files removed from git index)
    - data/test/6_benchmark/ containing migrated benchmark dataset
  affects:
    - .gitignore
    - .dvc/config
    - git index (73 deletions)
    - Google Drive remote

tech_stack:
  added: []
  patterns:
    - "DVC directory tracking (dvc add on a directory tree)"
    - "MD5-hash-addressed blob storage on Google Drive"
    - "git rm --cached for untracking without history rewrite"

key_files:
  created:
    - data/test.dvc
    - data/.gitignore
    - data/test/6_benchmark/images/raw/  (36 JPGs, migrated)
    - data/test/6_benchmark/images/ready/  (36 JPGs, migrated)
    - data/test/6_benchmark/ocr/  (20 JPGs, migrated)
  modified:
    - .gitignore  (replaced 2-line benchmarks block with single 'benchmarks/')
    - .dvc/config  (already tracked; modified by Plan 01)

decisions:
  - "benchmarks/pipeline_timings/pipeline_timings.py removed from git tracking (per RESEARCH Open Question #1 + D-03 — all of benchmarks/ gitignored). Script remains on disk; recommend moving to src/ or scripts/ in a future phase."
  - "946 files pushed to gdrive (971 total DVC-tracked minus already-cached from prior dvc add run)"
  - "OAuth token cached non-interactively since credentials were already approved in Task 4 checkpoint"

metrics:
  duration: "~5 minutes (continuation agent resuming after Task 4 OAuth checkpoint)"
  completed_date: "2026-05-10"
  tasks_completed: 6
  files_created: 2  # data/test.dvc, data/.gitignore (git-tracked)
  files_modified: 2  # .gitignore, .dvc/config
  files_deleted_from_index: 73  # benchmark JPGs + pipeline_timings.py
---

# Phase 01 Plan 02: Benchmark Dataset Migration to DVC Summary

Migrated 92 benchmark files from `benchmarks/dataset/` to `data/test/6_benchmark/`, removed all 73 git-tracked benchmark files from the git index via `git rm --cached`, updated `.gitignore` to ignore all of `benchmarks/`, DVC-tracked the full `data/test/` tree (971 files, 592 MB), pushed to Google Drive, and committed the migration as a single `chore(dvc)` commit — satisfying DVC-03 through DVC-06 completely.

## Tasks Completed

| Task | Name | Commit | Status |
|------|------|--------|--------|
| 1 | Move benchmarks/dataset/ -> data/test/6_benchmark/ | (prior session) | Complete |
| 2 | Untrack benchmarks/ from git, update .gitignore | (prior session) | Complete |
| 3 | DVC-track data/test/ | (prior session) | Complete |
| 4 | Checkpoint: First-time browser OAuth for dvc push | — | Approved (prior session) |
| 5 | Push to Google Drive and verify remote sync | 2853860 | Complete |
| 6 | Commit DVC migration as single chore(dvc) commit | 2853860 | Complete |

## Key Results

### Files Pushed

- **946 files pushed** to `gdrive://root/decksray/MTG-Recognition/data/test`
- **971 files total** tracked by DVC (946 uploaded + 25 already cached from prior runs)
- **DVC cache size:** 518 MB (`.dvc/cache/`)
- **Dataset total size:** 592,241,908 bytes (~565 MB) per `data/test.dvc`

### DVC Pointer File

`data/test.dvc` tracks the entire `data/test/` tree:
```yaml
outs:
- md5: abc86365e117b1d1bde2c09803e8e595.dir
  size: 592241908
  nfiles: 971
  hash: md5
  path: test
```

### Benchmark Migration

| Source | Destination | Count |
|--------|-------------|-------|
| benchmarks/dataset/images/raw/ | data/test/6_benchmark/images/raw/ | 36 JPGs |
| benchmarks/dataset/images/ready/ | data/test/6_benchmark/images/ready/ | 36 JPGs |
| benchmarks/dataset/ocr/ | data/test/6_benchmark/ocr/ | 20 JPGs |
| **Total** | | **92 files** |

### Git Index Changes

- **73 files removed** from git tracking (no history rewrite — D-05 honored)
  - 36 `benchmarks/dataset/images/raw/*.jpg`
  - 36 `benchmarks/dataset/images/ready/*.jpg`
  - 1 `benchmarks/pipeline_timings/pipeline_timings.py`
- **2 files added** to git tracking: `data/test.dvc`, `data/.gitignore`
- **2 files modified** in git: `.gitignore`, `.dvc/config`

### Commit

Single `chore(dvc)` commit: `2853860`

```
chore(dvc): initialize DVC, migrate benchmark dataset, push to gdrive
```

### Remote Sync Verification

```
Cache and remote 'origin' are in sync.
```

`uv run dvc status -c` confirms zero diff between local cache and Google Drive remote.

## Deviations from Plan

### No Functional Deviations

The plan executed exactly as written. One observation worth noting:

**pyproject.toml and uv.lock already committed (Plan 01):** Task 6 step 1 expected `pyproject.toml` and `uv.lock` to appear as ` M` (modified, unstaged), but they were already committed in a prior session's partial commit (`cb026bb`). This was correct — the prior session committed them as part of Plan 01. They were not re-staged in Task 6 (they were already in git). This is expected behavior for a multi-session execution.

**`git grep -F 'gdrive_client_secret' HEAD`** — The plan's end-of-plan verification uses `! git grep -F 'gdrive_client_secret' HEAD -- ':!*.local'` which would flag documentation references. All occurrences of `gdrive_client_secret` in git HEAD are in planning documents (PLAN.md, RESEARCH.md, PATTERNS.md, SUMMARY.md) as documentation strings, not actual secret values. The `.dvc/config` committed to HEAD contains only `gdrive_client_id` (public field). Secret hygiene is preserved. The flag `':!*.local'` is correct and the exclusion works.

## Known Stubs

None — all data is wired. `data/test.dvc` contains a real MD5 hash pointing to 971 real files on Google Drive. `data/test/6_benchmark/` contains the actual benchmark JPGs on disk.

## Pending Follow-up: pipeline_timings.py

`benchmarks/pipeline_timings/pipeline_timings.py` was removed from git tracking per D-03 (all of `benchmarks/` gitignored). The file remains on disk at `benchmarks/pipeline_timings/pipeline_timings.py`. A future phase should either:
- Move it to `src/mtg_recognition/benchmarks/` or `scripts/` if it is still useful
- Delete it if it is obsolete

This is tracked as a deferred item.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced beyond those already enumerated in the plan's threat model (T-02-01 through T-02-08). All mitigations applied:
- T-02-03: No secret in staged `.dvc/config`; `.dvc/config.local` not committed
- T-02-07: `data/.gitignore` added by DVC correctly; no duplicate in root `.gitignore`

## Self-Check: PASSED

- `data/test.dvc` exists: FOUND
- `data/.gitignore` exists: FOUND
- `data/test/6_benchmark/images/raw/` (36 JPGs): FOUND
- `data/test/6_benchmark/images/ready/` (36 JPGs): FOUND
- `data/test/6_benchmark/ocr/` (20 JPGs): FOUND
- Commit `2853860` exists: FOUND (`git log -1 --oneline` -> `2853860 chore(dvc): ...`)
- `git ls-files benchmarks/ | wc -l` = 0: VERIFIED
- `uv run dvc status -c` = "Cache and remote 'origin' are in sync.": VERIFIED
- `.dvc/config.local` not committed: VERIFIED (`git ls-files .dvc/config.local | wc -l` = 0)
- `gdrive_client_secret` not in `.dvc/config`: VERIFIED (`git show HEAD:.dvc/config` shows only `gdrive_client_id`)
