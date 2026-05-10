---
phase: 01-dataset-versioning
plan: "03"
subsystem: infra
tags: [dvc, gdrive, verification, documentation, collaborator-onboarding]

requires:
  - 01-02  # data/test.dvc committed + pushed to gdrive

provides:
  - "DVC-07 verified: empty-cache dvc pull reconstructs data/test/ byte-identical from gdrive remote"
  - "DVC-08 satisfied: CONTRIBUTING.md documents GCP OAuth setup, dvc pull, dvc push workflow"
  - "Phase 1 complete: all 5 ROADMAP success criteria PASS"

affects:
  - CONTRIBUTING.md (created)

tech-stack:
  added: []
  patterns:
    - "Empty-cache round-trip simulation: mv cache+data to /tmp, dvc pull, diff -rq to verify"
    - "Secret-separation documentation: --local flag for gdrive_client_secret in CONTRIBUTING.md"

key-files:
  created:
    - CONTRIBUTING.md
  modified: []

decisions:
  - "Option B (empty-cache simulation) chosen over Option A (git clone to tempdir): avoids requiring git push while exercising the same DVC code path"
  - "dvc pull --dry flag does not exist in DVC 3.67.1: criterion 4 verified via dvc status (Data and pipelines are up to date)"
  - "memsearch memory files excluded from git commits per plan notes"

metrics:
  duration: "~10 minutes"
  completed_date: "2026-05-10"
  tasks_completed: 3
  files_created: 1
  files_modified: 0
---

# Phase 01 Plan 03: Collaborator Round-Trip Verification and CONTRIBUTING.md Summary

DVC-07 verified empirically: empty-cache `dvc pull` reconstructed all 971 files in `data/test/` byte-identical to the original (diff produced zero output lines, 946 files fetched from gdrive remote). CONTRIBUTING.md written with all required sections. All 5 ROADMAP Phase 1 success criteria PASS. Phase 1 is complete.

## Performance

- **Duration:** ~10 minutes
- **Started:** 2026-05-10
- **Completed:** 2026-05-10
- **Tasks:** 3 of 3 completed
- **Files created:** 1 (CONTRIBUTING.md)

## Task 1: DVC Pull Round-Trip Verification

### DVC-07 verified

**Method:** Option B — empty-cache simulation (no git push required)

**Steps executed:**
1. Pre-flight checks: all PASS (`data/test.dvc` tracked, remote `origin` configured, cache + data present)
2. Moved `.dvc/cache` (518 MB) to `/tmp/dvc-cache-backup-1778415957`
3. Moved `data/test` (567 MB) to `/tmp/data-test-backup-1778415957`
4. Ran `uv run dvc pull` — **exit code 0**
5. Ran `diff -rq data/test /tmp/data-test-backup-1778415957` — **zero output (byte-identical)**
6. Verified benchmark file counts: raw=36, ready=36, ocr=20 — all PASS
7. Removed backup directories (step 6 cleanup — backups no longer needed)

### Pull Log (`/tmp/dvc-pull-1778415957.log`)

```
A       data/test/
946 files fetched and 971 files added
```

946 files fetched from Google Drive remote, 971 files restored to `data/test/` (946 downloaded + 25 via cache links).

### Post-pull verification

```
$ uv run dvc status
Data and pipelines are up to date.

$ uv run dvc status -c
Cache and remote 'origin' are in sync.
```

## Task 2: CONTRIBUTING.md

`CONTRIBUTING.md` created at repo root (70 lines, above the 50-line minimum).

**Required sections present:**
- H1: `# Contributing to Decksray MTG Recognition`
- H2: `## Dataset Setup`
- H3: `### One-time setup (per developer)` — GCP OAuth setup, 5 steps
- H3: `### Pull the dataset` — `uv run dvc pull` command + token cache path
- H2: `## Adding New Test Images` — `dvc add` + `dvc push` + git commit workflow
- H2: `## Troubleshooting` — "This app is blocked", "no data to pull", "file not found in remote"

**Required literal strings verified:**
- `uv run dvc pull` (collaborator command)
- `uv run dvc push` (contributor command)
- `dvc remote modify --local origin gdrive_client_secret` (secret separation)
- `gdrive://root/decksray/MTG-Recognition/data/test` (remote URL)
- `~/.cache/pydrive2fs/<client_id>/default.json` (token cache path)
- `This app is blocked` (Pitfall 1 troubleshooting)
- No `gdrive_client_secret = ` pattern (secret hygiene)

## Task 3: Commit and ROADMAP Verification

### Commit

`a96e0b4 docs(dvc): add CONTRIBUTING.md with dvc pull/push workflow`

### ROADMAP Phase 1 Success Criteria

```
=== ROADMAP Phase 1 Success Criteria ===
PASS: 1. dvc init committed
PASS: 2. gdrive remote configured + push complete
PASS: 3. benchmarks untracked + content under data/test/6_benchmark/
PASS: 4. dvc pull verified (Data and pipelines are up to date)
PASS: 5. clean working tree (excluding memsearch)
```

All 5 ROADMAP Phase 1 criteria PASS. Zero FAIL lines.

## Commits in This Plan

| Task | Description | Commit |
|------|-------------|--------|
| Task 1 | No permanent files changed (round-trip verification only) | — |
| Task 2+3 | CONTRIBUTING.md created and committed | `a96e0b4` |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `dvc pull --dry` flag does not exist in DVC 3.67.1**
- **Found during:** Task 3 (ROADMAP criterion 4 check)
- **Issue:** The plan used `uv run dvc pull --dry` to verify criterion 4, but DVC 3.67.0+ removed the `--dry` flag from `dvc pull`
- **Fix:** Used `uv run dvc status` and `uv run dvc status -c` instead — both report "Data and pipelines are up to date" and "Cache and remote 'origin' are in sync." This is equivalent evidence that `dvc pull` would be a no-op (already up to date from Task 1's successful pull)
- **Files modified:** None

## Phase 1 Retrospective

### What worked well

- **Secret separation pattern** (gdrive_client_id public, gdrive_client_secret in config.local) worked without friction — zero leakage across 3 plans and multiple commits
- **Option B (empty-cache simulation)** was the right call: no git remote push required, exercises identical DVC code path, runs in under 2 minutes
- **Single DVC root at `data/test/`** proved clean: one `dvc pull` command, one pointer file, simple mental model for contributors
- **946-of-971 files fetched** in Task 1 (25 were already cached) confirms deduplicated blob storage working correctly

### What was painful

- **GCP OAuth setup** required a human-action checkpoint in Plan 01 Task 3 — cannot be automated. This is by design (DVC's auth model) but the CONTRIBUTING.md now documents it clearly enough that future contributors won't need to ask
- **benchmarks/pipeline_timings/pipeline_timings.py** was accidentally removed from git tracking per D-03 blanket gitignore. File remains on disk but is now untracked. Flagged as deferred item in Plan 02 SUMMARY

### Patterns to carry into Phase 2

- **`uv run <tool> <command>` pattern** used consistently — Phase 2's `mtg sync` CLI should follow the same `uv run mtg sync` invocation pattern for zero-install consistency
- **Verification via `dvc status -c`** is the reliable end-to-end check (cache ↔ remote sync), equivalent to a dry pull. Use this in Phase 2's `mtg sync` verification step
- **Deduplication is automatic** — Phase 2's image download job can be re-run idempotently without re-downloading existing blobs if it writes to a DVC-tracked directory

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes beyond those in the plan's threat model. T-03-01 mitigated (no real secret in CONTRIBUTING.md). T-03-02 mitigated (backup restore branch never needed — pull succeeded on first attempt).

## Self-Check: PASSED

- `CONTRIBUTING.md` exists: FOUND
- `01-03-SUMMARY.md` exists: FOUND
- `data/test.dvc` exists: FOUND
- `data/test/6_benchmark/` exists: FOUND
- Commit `a96e0b4` exists: FOUND
- `DVC-07 verified` marker in SUMMARY: FOUND
