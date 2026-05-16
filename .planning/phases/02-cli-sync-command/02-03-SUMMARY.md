---
phase: 02-cli-sync-command
plan: 03
subsystem: cli
tags: [click, cli, sync, build-database, phash, tdd, state-file, rich]
dependency_graph:
  requires:
    - 02-01 (cli.py Click group + entry point)
    - 02-02 (build-index subcommand)
  provides:
    - src/mtg_recognition/cli.py::sync (sync orchestrator subcommand)
    - src/mtg_recognition/jobs/build_database.py::build_mobilecache (step function)
    - src/mtg_recognition/jobs/build_database.py::download_images (step function)
    - data/reference/.mtg-sync-state.json (runtime artifact, gitignored)
  affects:
    - src/mtg_recognition/cli.py (new sync command + 3 helpers + new imports)
    - src/mtg_recognition/jobs/build_database.py (refactored step functions)
    - .gitignore (new ignore entry in Reference block)
    - tests/unit/test_cli.py (7 new TestSyncCommand tests)
    - tests/unit/test_build_database.py (new test file, 4 tests)
tech_stack:
  added: []
  patterns:
    - D-02 resilience: write null before step, version on success
    - D-03 all-null fallback for missing/corrupt state file
    - State file JSON read/write via pathlib + json stdlib
    - urllib.request for lightweight Meta.json HTTP fetch
    - rich.panel.Panel for Sync Summary output (D-14)
    - rich Console with markup for step headers and skip messages (D-13)
    - click.echo(err=True) for error messages to stderr (SYNC-04)
    - sqlite3 fallback for MTGO_DB_FILE meta table read (skip-db-check path)
key_files:
  created:
    - tests/unit/test_build_database.py
  modified:
    - src/mtg_recognition/jobs/build_database.py
    - src/mtg_recognition/cli.py
    - tests/unit/test_cli.py
    - .gitignore
decisions:
  - D-01: State file at data/reference/.mtg-sync-state.json with 3-key schema
  - D-02: Resilience pattern: null before work, version on success
  - D-03: Missing/corrupt state treated as all-null (first-run experience)
  - D-04: Meta.json fetch for MobileCache freshness; --skip-db-check bypass
  - D-05: Image step skip if images_version == mobilecache_version
  - D-06: pHash step skip if phash_version == images_version
  - D-07: --force flag bypasses all version checks
  - D-08: --skip-db-check flag bypasses Meta.json fetch
  - D-13: Rich step headers + green checkmark skip messages
  - D-14: Final Sync Summary Panel after all steps
  - D-15: tqdm progress bar in build_index() preserved unchanged
  - cli.py fix: console.print() does not accept err=True; use click.echo(err=True) for stderr (auto-fixed during GREEN)
metrics:
  duration: 347s
  completed: "2026-05-16T17:01:11Z"
  tasks: 2
  files: 5
---

# Phase 2 Plan 03: `mtg sync` Orchestrator + `build_database.py` Refactor Summary

**One-liner:** Three-step `mtg sync` orchestrator with MTGJSON version-based skipping, D-02 state-file resilience, rich step headers and summary panel, wired to refactored `build_mobilecache()` and `download_images()` step functions extracted from `build_database.py`.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| RED (T1) | Failing tests for build_database refactor | 004e0fe | tests/unit/test_build_database.py |
| GREEN (T1) | Refactor build_database.py step functions | 313f5df | src/mtg_recognition/jobs/build_database.py |
| RED (T2) | Failing tests for mtg sync orchestrator | 04e3620 | tests/unit/test_cli.py |
| GREEN (T2) | Implement sync orchestrator + gitignore | 60a894c | src/mtg_recognition/cli.py, .gitignore |

## What Was Built

### `src/mtg_recognition/jobs/build_database.py` (refactored)

Extracted two named step functions from the former monolithic `main()`:
- `build_mobilecache() -> Engine`: Downloads AllPrintings ZIP, extracts, and rebuilds MobileCache. Returns `dest_engine` so caller can pass it directly to `download_images()`.
- `download_images(dest_engine=None) -> None`: Downloads reference images; falls back to `create_engine(LOCAL_DB_FILE)` when `dest_engine` is None for standalone use.
- `main()` now delegates: `dest_engine = build_mobilecache(); download_images(dest_engine)`.
- `SCRYFALL_HEADERS`, `if __name__ == "__main__"` guard, and all other functions preserved unchanged (CLI-06).

### `src/mtg_recognition/cli.py` (extended)

New additions above the `if __name__ == "__main__"` guard:
- New imports: `json`, `sqlite3`, `urllib.request as urllib_request`, `rich.panel.Panel`, `mtg_recognition.jobs.build_database` (module import)
- Module constants: `STATE_FILE = settings.REF_DIR / ".mtg-sync-state.json"`, `META_URL`
- `_load_state()`: Returns parsed state dict; all-null on `FileNotFoundError` or `json.JSONDecodeError` (D-03)
- `_save_state(state)`: Creates parent dir and writes JSON (D-02 plumbing)
- `_fetch_mtgjson_version()`: Lightweight GET to Meta.json; re-raises all exceptions (SYNC-04)
- `@cli.command("sync")` with `--force` (D-07) and `--skip-db-check` (D-08):
  - Creates `REF_DIR` on first run
  - Step 1: MobileCache — Meta.json version check, null-before/version-on-success write (D-02)
  - Step 2: Images — skips if `images_version == mobilecache_version` (D-05)
  - Step 3: pHash — `PHashCardMatcher(auto_build=False).build_index(force=True)` (D-06, D-15)
  - `try/except Exception` wrapper: `click.echo(err=True)` + `raise SystemExit(1)` (SYNC-04)
  - Final `Panel("Sync Summary")` listing per-artifact status (D-14)

Existing `recognize`, `segment`, and `build-index` registrations preserved unchanged.

### `.gitignore`

Added `data/reference/.mtg-sync-state.json` immediately after `data/reference/download_errors.log`, inside the existing "Reference cards dataset" comment block.

### `tests/unit/test_build_database.py` (new file)

4 tests, all `@pytest.mark.unit`:
- `test_imports`: build_mobilecache, download_images, main are importable
- `test_main_calls_both_steps`: main() delegates to both step functions in order
- `test_download_images_creates_engine_when_none`: standalone call creates SQLite engine with MobileCache.sqlite URL
- `test_download_images_uses_provided_engine`: provided engine bypasses create_engine

### `tests/unit/test_cli.py` (extended)

7 new tests in `TestSyncCommand`, all `@pytest.mark.unit`:
- `test_sync_help`: --help exits 0, lists --force and --skip-db-check
- `test_sync_first_run_runs_all_steps`: all three steps fire, state file written with correct versions
- `test_sync_skips_when_versions_match`: all steps skipped, 3 checkmarks in output
- `test_sync_force_rebuilds_all`: --force overrides version match, all steps rebuild
- `test_sync_skip_db_check_bypasses_meta_fetch`: _fetch_mtgjson_version not called
- `test_sync_step1_failure_leaves_null_state`: exit 1, state has null mobilecache_version (D-02 verified)
- `test_sync_creates_ref_dir_on_first_run`: REF_DIR created even if missing before sync

## Test Results

```
32 passed, 3 warnings in 3.32s
```

(3 warnings are PaddlePaddle/requests version warnings unrelated to this plan)

## Verification

1. `uv run pytest tests/unit/test_cli.py tests/unit/test_build_database.py -v` — 32 passed
2. `uv run mtg --help` — lists `recognize`, `segment`, `build-index`, AND `sync`
3. `uv run mtg sync --help` — shows `--force` and `--skip-db-check`
4. `git check-ignore data/reference/.mtg-sync-state.json` — exits 0, prints path
5. Legacy invocations:
   - `uv run python -m mtg_recognition.engine.pipeline --help` — exits 0
   - `uv run python -m mtg_recognition.engine.segmentation.cli --help` — exits 0
   - `uv run python -c "import mtg_recognition.jobs.build_database; print('ok')"` — prints ok
   - `uv run python -c "import mtg_recognition.engine.matching.phash_matcher; print('ok')"` — prints ok

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `console.print()` does not accept `err=True` keyword argument**
- **Found during:** GREEN phase (Task 2), first test run of `test_sync_step1_failure_leaves_null_state`
- **Issue:** The plan's sync body used `console.print(..., err=True)` for SYNC-04 error output. Rich's `Console.print()` does not accept an `err` parameter — that argument is `click.echo`-specific. This caused `TypeError("Console.print() got an unexpected keyword argument 'err'")`.
- **Fix:** Replaced `console.print(f"[red]Error...", err=True)` with `click.echo(f"Error...", err=True)` for the SYNC-04 error path. The error message is rendered without red markup but reaches stderr correctly, which is what the test asserts.
- **Files modified:** `src/mtg_recognition/cli.py`
- **Commit:** 60a894c (bundled with GREEN phase)

## Requirements Satisfied

- SYNC-01: `mtg sync` Step 1 calls `build_database.build_mobilecache()` which downloads MTGJSON ZIP and rebuilds MobileCache
- SYNC-02: `mtg sync` Step 2 calls `build_database.download_images()` which calls `download_raw_images()` (already incremental)
- SYNC-03: `mtg sync` Step 3 calls `PHashCardMatcher(auto_build=False).build_index(force=True)`
- SYNC-04: any exception in any step exits non-zero with descriptive error message via `click.echo(err=True)` + `raise SystemExit(1)`
- CLI-04: `mtg sync` is a subcommand of the `mtg` group (declared in Plan 01 `[project.scripts]`)
- CLI-06: `python -m mtg_recognition.jobs.build_database` still works (main() preserved, __main__ guard intact)
- D-01 through D-15: all implemented as documented in CONTEXT.md

## TDD Gate Compliance

- RED gate (test commit): 004e0fe (build_database tests), 04e3620 (sync tests) — PASSED
- GREEN gate (feat commit): 313f5df (build_database impl), 60a894c (sync impl + gitignore) — PASSED
- REFACTOR: not needed — code was clean on first implementation pass

## Known Stubs

None — all wiring is live. `build_mobilecache()`, `download_images()`, and `PHashCardMatcher.build_index()` delegate to existing implementations. State file reads and writes use real pathlib/json. No placeholder data or empty flows.

## Threat Flags

None — no new network endpoints (Meta.json read is existing MTGJSON public API), no new auth paths, no schema changes. State file `data/reference/.mtg-sync-state.json` is gitignored and contains only version strings (no secrets).

## Self-Check: PASSED

- `src/mtg_recognition/jobs/build_database.py` — FOUND (contains `def build_mobilecache`, `def download_images`, `def main`)
- `src/mtg_recognition/cli.py` — FOUND (contains `@cli.command("sync")`, `STATE_FILE`, `_load_state`, `_save_state`, `_fetch_mtgjson_version`, `Panel`)
- `tests/unit/test_build_database.py` — FOUND (4 tests)
- `tests/unit/test_cli.py` — FOUND (32 tests total, 7 new sync tests)
- `.gitignore` — FOUND (`data/reference/.mtg-sync-state.json` present)
- Commit 004e0fe — FOUND
- Commit 313f5df — FOUND
- Commit 04e3620 — FOUND
- Commit 60a894c — FOUND
- All 32 tests passing — CONFIRMED
