---
phase: 02-cli-sync-command
plan: 02
subsystem: cli
tags: [click, cli, build-index, phash, tdd]
dependency_graph:
  requires:
    - 02-01 (cli.py Click group + entry point)
  provides:
    - src/mtg_recognition/cli.py::build_index (build-index subcommand)
  affects:
    - src/mtg_recognition/cli.py (new subcommand added)
    - tests/unit/test_cli.py (5 new test cases)
tech_stack:
  added: []
  patterns:
    - preflight check pattern: click.echo(err=True) + raise SystemExit(1)
    - PHashCardMatcher(auto_build=False) to prevent double-build
    - monkeypatch settings property for isolated test control
key_files:
  created: []
  modified:
    - src/mtg_recognition/cli.py
    - tests/unit/test_cli.py
decisions:
  - D-09: --force and --workers flags with cpu_count()-1 default, show_default=True
  - D-10: preflight check exits 1 with message directing user to run mtg sync when IMG_DIR missing/empty
  - D-15: tqdm progress bar from PHashCardMatcher.build_index passes through unchanged (no rich wrapping)
metrics:
  duration: 240s
  completed: "2026-05-16T16:55:00Z"
  tasks: 1
  files: 2
---

# Phase 2 Plan 02: `mtg build-index` Subcommand Summary

**One-liner:** `build-index` Click subcommand wrapping `PHashCardMatcher.build_index(force, num_workers)` with a preflight guard that exits 1 and directs users to `mtg sync` when the image directory is missing or empty.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| RED | Failing tests for build-index subcommand | 6ba4d53 | tests/unit/test_cli.py |
| GREEN | Implement build-index subcommand | e2c21ce | src/mtg_recognition/cli.py, tests/unit/test_cli.py |

## What Was Built

### `src/mtg_recognition/cli.py`

Extended with:
- `from multiprocessing import cpu_count` — for `--workers` default
- `from mtg_recognition.engine.matching.phash_matcher import PHashCardMatcher` — wired to the command body
- `from mtg_recognition.settings import settings` — for preflight check on `settings.IMG_DIR`
- New `@cli.command("build-index")` command `build_index(force, workers)`:
  - `--force` flag (`is_flag=True`, default `False`, `show_default=True`)
  - `--workers` integer option (default `cpu_count()-1`, `show_default=True`)
  - Preflight check: if `settings.IMG_DIR` missing or empty → `click.echo(err=True)` + `raise SystemExit(1)` with message containing `data/reference/images/` and `mtg sync`
  - `PHashCardMatcher(auto_build=False)` prevents constructor from triggering automatic build
  - `matcher.build_index(force=force, num_workers=workers)` delegates to existing implementation

Existing `recognize` and `segment` registrations preserved unchanged.

### `tests/unit/test_cli.py`

Added `TestBuildIndexCommand` class (5 new tests, all `@pytest.mark.unit`):
- `test_build_index_help` — `--help` exits 0, lists `--force` and `--workers`
- `test_build_index_preflight_missing_dir` — exits 1, output contains `data/reference/images/` and `mtg sync`
- `test_build_index_preflight_empty_dir` — exits 1, same output assertions
- `test_build_index_invokes_matcher_with_defaults` — `PHashCardMatcher(auto_build=False)` called once, `.build_index(force=False, num_workers=cpu_count()-1)` called once
- `test_build_index_invokes_matcher_with_flags` — `--force --workers 4` produces `build_index(force=True, num_workers=4)`

## Test Results

```
21 passed, 3 warnings in 3.14s
```

(3 warnings are PaddlePaddle/requests version warnings unrelated to this plan)

## Verification

1. `uv run mtg --help` — lists `recognize`, `segment`, AND `build-index`
2. `uv run mtg build-index --help` — shows `--force` and `--workers INTEGER [default: 15]`
3. `uv run pytest tests/unit/test_cli.py -v` — 21 passed
4. `uv run python -m mtg_recognition.engine.pipeline --help` — exits 0 (CLI-06 backward compat)
5. All 6 acceptance criteria grep checks return `1` (correct match count)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed `mix_stderr=False` incompatibility in test helpers**
- **Found during:** GREEN phase (first test run)
- **Issue:** Tests used `CliRunner(mix_stderr=False)` but Click 8.3.1 `CliRunner.__init__` does not accept this argument (it was removed). The plan's test snippet assumed a Click version that exposed this parameter. Tests that asserted `result.stderr` would always raise `TypeError`.
- **Fix:** Replaced `CliRunner(mix_stderr=False)` with plain `CliRunner()` and asserted `result.output` instead of `result.stderr`. In Click 8.3.1, stderr content is mixed into `result.output` by default — verified with a one-shot test script.
- **Files modified:** `tests/unit/test_cli.py`
- **Commit:** e2c21ce (bundled with GREEN phase commit)

## Requirements Satisfied

- CLI-03: `mtg build-index` rebuilds the pHash index standalone (wraps `PHashCardMatcher.build_index`)
- D-09: `--force` and `--workers` flags with `cpu_count()-1` default
- D-10: Preflight check exits non-zero with message directing user to `mtg sync`

## Known Stubs

None — command is fully wired to `PHashCardMatcher.build_index`. No placeholder text or empty data flows.

## Threat Flags

None — `build-index` only reads from `data/reference/images/` (read-only check) and writes to `data/reference/index/phash/` via the existing `PHashCardMatcher.build_index` implementation. No new network endpoints, auth paths, or schema changes.

## Self-Check: PASSED

- `src/mtg_recognition/cli.py` — FOUND
- `tests/unit/test_cli.py` — FOUND (21 tests)
- Commit 6ba4d53 (RED) — FOUND
- Commit e2c21ce (GREEN) — FOUND
- `@cli.command("build-index")` in cli.py — FOUND (grep count: 1)
- `PHashCardMatcher(auto_build=False)` in cli.py — FOUND (grep count: 1)
- All 21 tests passing — CONFIRMED
