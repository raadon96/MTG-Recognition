---
phase: 02-cli-sync-command
verified: 2026-05-16T00:00:00Z
status: passed
score: 11/11 must-haves verified
overrides_applied: 0
re_verification: null
gaps: []
human_verification: []
---

# Phase 02: CLI & Sync Command — Verification Report

**Phase Goal:** A collaborator can invoke any pipeline operation via a short `mtg <subcommand>` command, and `mtg sync` rebuilds the full reference data stack (MobileCache → images → pHash index) in one shot
**Verified:** 2026-05-16
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `mtg recognize`, `mtg segment`, `mtg build-index`, and `mtg sync` all available after `uv sync` | ✓ VERIFIED | `mtg --help` output shows all four; `[project.scripts]` declares `mtg = "mtg_recognition.cli:cli"` |
| 2 | `mtg sync` runs to completion (state file written, all three steps execute on first run) | ✓ VERIFIED | `test_sync_first_run_runs_all_steps` passes; state file written with all three version strings |
| 3 | `mtg sync` exits non-zero with descriptive error message if any step fails | ✓ VERIFIED | `try/except Exception` in `sync()` at line 263 of cli.py; `test_sync_step1_failure_leaves_null_state` passes (exit 1, "boom" in output) |
| 4 | Existing `uv run python -m mtg_recognition.engine.pipeline` and `uv run python -m mtg_recognition.engine.segmentation.cli` still work unchanged | ✓ VERIFIED | Both invocations exit 0 with help output; `python -c "import mtg_recognition.jobs.build_database"` succeeds |
| 5 | `mtg sync` skips up-to-date steps on second run | ✓ VERIFIED | `test_sync_skips_when_versions_match` passes; three `✓` checkmarks in output confirmed |
| 6 | `mtg sync --force` bypasses all version checks | ✓ VERIFIED | `test_sync_force_rebuilds_all` passes |
| 7 | `mtg sync --skip-db-check` bypasses Meta.json fetch | ✓ VERIFIED | `test_sync_skip_db_check_bypasses_meta_fetch` passes |
| 8 | State file resilience: null written BEFORE work begins per step | ✓ VERIFIED | Lines 187, 226, 252 of cli.py write null before calling build_mobilecache/download_images/build_index respectively |
| 9 | `mtg build-index` preflight exits non-zero when image dir missing/empty | ✓ VERIFIED | Lines 108-113 of cli.py; `test_build_index_preflight_missing_dir` and `test_build_index_preflight_empty_dir` both pass |
| 10 | `data/reference/.mtg-sync-state.json` is gitignored | ✓ VERIFIED | `git check-ignore data/reference/.mtg-sync-state.json` exits 0 and returns path |
| 11 | Rich step headers, skip checkmarks, and final Sync Summary Panel are implemented | ✓ VERIFIED | Lines 153, 212, 238 (step headers); lines 181, 221, 246 (✓ skip messages); line 278 (Panel with title "Sync Summary") |

**Score:** 11/11 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/mtg_recognition/cli.py` | Click group with recognize/segment/build-index/sync subcommands | ✓ VERIFIED | 283 lines; `@click.group()` at line 72; all four subcommands present; module-level `console = Console()` |
| `pyproject.toml` | `[project.scripts]` with `mtg = "mtg_recognition.cli:cli"` | ✓ VERIFIED | `[project.scripts]` at line 44; exact entry point string at line 45; positioned between `[project]` and `[dependency-groups]` |
| `tests/unit/test_cli.py` | CliRunner tests for all subcommands | ✓ VERIFIED | 450 lines; 28 test cases across 4 test classes (TestCliGroup, TestCliHelp, TestBuildIndexCommand, TestSyncCommand); all pass |
| `tests/unit/test_build_database.py` | Unit tests for build_mobilecache/download_images/main | ✓ VERIFIED | 86 lines; 4 test cases across 3 test classes; all pass |
| `src/mtg_recognition/jobs/build_database.py` | Refactored with `build_mobilecache()` and `download_images()` at module level | ✓ VERIFIED | `def build_mobilecache()` at line 261; `def download_images()` at line 293; `def main()` at line 307 delegates to both; `if __name__ == "__main__"` guard at line 313 |
| `.gitignore` | `data/reference/.mtg-sync-state.json` inside Reference cards dataset block | ✓ VERIFIED | Line 12 of .gitignore; within the "Reference cards dataset" comment block |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pyproject.toml [project.scripts]` | `src/mtg_recognition/cli.py::cli` | `mtg_recognition.cli:cli` | ✓ WIRED | Exact string at line 45 of pyproject.toml; `uv run mtg --help` resolves and lists all four subcommands |
| `src/mtg_recognition/cli.py` | `engine/pipeline.py::main` | `cli.add_command(_recognize_main, name="recognize")` | ✓ WIRED | Line 81; identity check test passes (`cli.commands["recognize"] is _recognize_main`) |
| `src/mtg_recognition/cli.py` | `engine/segmentation/cli.py::main` | `cli.add_command(_segment_main, name="segment")` | ✓ WIRED | Line 82; identity check test passes |
| `src/mtg_recognition/cli.py::build_index` | `phash_matcher.py::PHashCardMatcher.build_index` | `matcher.build_index(force=force, num_workers=workers)` | ✓ WIRED | Lines 117-118; `test_build_index_invokes_matcher_with_defaults` and `_with_flags` both pass |
| `src/mtg_recognition/cli.py::build_index` | `settings.py::settings.IMG_DIR` | preflight check on `img_dir.exists()` / `any(img_dir.iterdir())` | ✓ WIRED | Line 107-108; preflight tests pass |
| `src/mtg_recognition/cli.py::sync` | `jobs/build_database.py::build_mobilecache` | `build_database.build_mobilecache()` | ✓ WIRED | Line 190; `test_sync_first_run_runs_all_steps` asserts call |
| `src/mtg_recognition/cli.py::sync` | `jobs/build_database.py::download_images` | `build_database.download_images(dest_engine=dest_engine)` | ✓ WIRED | Line 229; test asserts call |
| `src/mtg_recognition/cli.py::sync` | `phash_matcher.py::PHashCardMatcher.build_index` | `PHashCardMatcher(auto_build=False)` then `.build_index(force=True)` | ✓ WIRED | Lines 256-257; `test_sync_first_run_runs_all_steps` asserts |
| `src/mtg_recognition/cli.py::_fetch_mtgjson_version` | `https://mtgjson.com/api/v5/Meta.json` | `urllib_request.urlopen` | ✓ WIRED | Lines 66-69; `META_URL` constant at line 35; `test_sync_skip_db_check_bypasses_meta_fetch` confirms it can be bypassed |
| `src/mtg_recognition/cli.py::sync` | `data/reference/.mtg-sync-state.json` | `_load_state()` / `_save_state()` | ✓ WIRED | `STATE_FILE` at line 32; helpers at lines 38-57; `test_sync_first_run_runs_all_steps` confirms state file written with correct content |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CLI-01 | 02-01 | `mtg recognize` wraps `pipeline.py` main | ✓ SATISFIED | `cli.add_command(_recognize_main, name="recognize")`; identity check test passes |
| CLI-02 | 02-01 | `mtg segment` wraps `segmentation/cli.py` main | ✓ SATISFIED | `cli.add_command(_segment_main, name="segment")`; identity check test passes |
| CLI-03 | 02-02 | `mtg build-index` wraps `PHashCardMatcher.build_index` standalone | ✓ SATISFIED | `@cli.command("build-index")`; preflight check; `PHashCardMatcher(auto_build=False)` |
| CLI-04 | 02-03 | `mtg sync` accessible via the `mtg` console-script entry point | ✓ SATISFIED | `sync` registered via `@cli.command("sync")`; reachable as `mtg sync` through single entry point |
| CLI-05 | 02-01 | All `mtg` subcommands accessible via single `[project.scripts]` entry point | ✓ SATISFIED | Single `mtg = "mtg_recognition.cli:cli"` entry point; all four subcommands attach via Click group |
| CLI-06 | 02-01, 02-02, 02-03 | Legacy `uv run python -m ...` invocations still work | ✓ SATISFIED | `pipeline.py` and `segmentation/cli.py` untouched; `build_database.py` preserves `main()` and `__name__` guard |
| SYNC-01 | 02-03 | `mtg sync` rebuilds MobileCache from latest MTGJSON | ✓ SATISFIED | Step 1 calls `build_database.build_mobilecache()` with version check via Meta.json |
| SYNC-02 | 02-03 | `mtg sync` downloads missing reference images (incremental) | ✓ SATISFIED | Step 2 calls `build_database.download_images()`; `download_raw_images` already incremental (skips existing files) |
| SYNC-03 | 02-03 | `mtg sync` rebuilds full pHash index after images | ✓ SATISFIED | Step 3 calls `PHashCardMatcher(auto_build=False).build_index(force=True)` |
| SYNC-04 | 02-03 | Non-zero exit + clear error message on any step failure | ✓ SATISFIED | `try/except Exception` at line 263; `click.echo(f"Error during sync: ...", err=True)` then `raise SystemExit(1)` |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `mtg --help` lists all four subcommands | `uv run mtg --help` | Output contains `recognize`, `segment`, `build-index`, `sync` | ✓ PASS |
| Full test suite passes | `uv run pytest tests/unit/test_cli.py tests/unit/test_build_database.py -v` | 32 passed, 3 warnings in 3.58s | ✓ PASS |
| Legacy pipeline invocation | `uv run python -m mtg_recognition.engine.pipeline --help` | Exit 0 | ✓ PASS |
| Legacy segmentation invocation | `uv run python -m mtg_recognition.engine.segmentation.cli --help` | Exit 0 | ✓ PASS |
| build_database module importable | `python -c "import mtg_recognition.jobs.build_database; print('ok')"` | `ok` | ✓ PASS |
| State file gitignored | `git check-ignore data/reference/.mtg-sync-state.json` | Returns path, exit 0 | ✓ PASS |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/mtg_recognition/jobs/build_database.py` | 256 | `# TODO download webp images instead` | ℹ️ Info | Pre-existing comment inside `download_raw_images()` — a function that predates Phase 02. Git log confirms this line existed before the Phase 02 refactor commit (`313f5df`). Not introduced by Phase 02. No follow-up issue reference required because Phase 02 did not introduce this debt. |

No blockers. No `TBD`, `FIXME`, or `XXX` markers in Phase 02 code. The single `TODO` is legacy, predates Phase 02, and is inside a pre-existing function not created by Phase 02.

### Human Verification Required

None. All behaviors are verifiable programmatically via the test suite and CLI invocation checks. `mtg sync` requires real MTGJSON network access and reference data to exercise the full happy path end-to-end, but all orchestration logic is covered by the mocked unit tests. This is a deliberate design choice (the tests mock external dependencies) and does not represent a gap.

### Gaps Summary

No gaps found. All 11 observable truths are VERIFIED. All 10 requirements (CLI-01 through CLI-06, SYNC-01 through SYNC-04) are satisfied by substantive, wired implementations with full test coverage (32 tests, all passing).

---

_Verified: 2026-05-16_
_Verifier: Claude (gsd-verifier)_
