---
phase: 02-cli-sync-command
plan: 01
subsystem: cli
tags: [click, cli, entry-point, tdd]
dependency_graph:
  requires: []
  provides:
    - src/mtg_recognition/cli.py::cli (Click group entry point)
    - mtg console script via [project.scripts]
  affects:
    - pyproject.toml (new [project.scripts] section)
tech_stack:
  added: []
  patterns:
    - click.group() + add_command() for zero-duplication subcommand registration (D-11)
    - Flat top-level module layout for CLI (D-12)
key_files:
  created:
    - src/mtg_recognition/cli.py
    - tests/unit/test_cli.py
  modified:
    - pyproject.toml
decisions:
  - D-11: add_command() reuse — recognize and segment inherit all existing options without duplication
  - D-12: flat cli.py module layout — consistent with database.py, settings.py
metrics:
  duration: 213s
  completed: "2026-05-16T16:46:49Z"
  tasks: 2
  files: 3
---

# Phase 2 Plan 01: mtg Click Group + recognize and segment Subcommands Summary

**One-liner:** `mtg` Click group with `recognize` and `segment` subcommands wired via `add_command()` registered as a console script entry point in `pyproject.toml`.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| RED | Failing tests for cli group | 62e4869 | tests/unit/test_cli.py |
| GREEN | Implement mtg click group | bc0d3f7 | src/mtg_recognition/cli.py |
| Task 2 | Add [project.scripts] entry point | 3c30118 | pyproject.toml |

## What Was Built

### `src/mtg_recognition/cli.py`

Top-level Click group (`@click.group()`) named `cli` with:
- `recognize` subcommand wired via `cli.add_command(_recognize_main, name="recognize")` — delegates to `engine/pipeline.py::main` with zero code duplication (D-11)
- `segment` subcommand wired via `cli.add_command(_segment_main, name="segment")` — delegates to `engine/segmentation/cli.py::main` with zero code duplication (D-11)
- Module-level `Console` instance (unused in this plan; stabilises import surface for plans 02 and 03)
- `if __name__ == "__main__": cli()` guard matching existing CLI convention

### `pyproject.toml`

Added `[project.scripts]` section between `[project]` and `[dependency-groups]`:

```toml
[project.scripts]
mtg = "mtg_recognition.cli:cli"
```

Also registered `unit` pytest marker to eliminate `PytestUnknownMarkWarning` (Rule 2 deviation).

### `tests/unit/test_cli.py`

16 Click `CliRunner` tests covering:
- Import correctness
- Group type assertion (`isinstance(cli, click.Group)`)
- Subcommand registration (`recognize`, `segment` keys present)
- Identity checks (commands are the same objects as pipeline/segmentation `main` functions)
- `--help` exit codes and option presence for all three levels

## Test Results

```
16 passed, 3 warnings in 3.16s
```

(3 warnings are PaddlePaddle/requests version warnings unrelated to this plan)

## Verification

1. `uv run pytest tests/unit/test_cli.py -v` — 16 passed
2. `uv run mtg --help` — lists `recognize` and `segment`
3. `uv run mtg recognize --help` — shows `--input`, `--k-phash`, `-k` (matches `python -m mtg_recognition.engine.pipeline --help`)
4. `uv run mtg segment --help` — shows `--input`, `--mode` (matches `python -m mtg_recognition.engine.segmentation.cli --help`)
5. `uv run python -m mtg_recognition.engine.pipeline --help` — exits 0 (CLI-06 backward compat)
6. `uv run python -m mtg_recognition.engine.segmentation.cli --help` — exits 0 (CLI-06 backward compat)
6. `git diff 3ba9d4b HEAD --name-only` shows only: `pyproject.toml`, `src/mtg_recognition/cli.py`, `tests/unit/test_cli.py`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Config] Registered `unit` pytest marker**
- **Found during:** Task 2
- **Issue:** `@pytest.mark.unit` used in new test file but not registered in `pyproject.toml` `[tool.pytest.ini_options].markers` — causes `PytestUnknownMarkWarning` on every test run
- **Fix:** Added `"unit: fast unit tests with no external dependencies"` to the markers list in `pyproject.toml`
- **Files modified:** `pyproject.toml`
- **Commit:** 3c30118 (bundled with Task 2 commit)

## Requirements Satisfied

- CLI-01: `mtg recognize` runs the existing recognition pipeline
- CLI-02: `mtg segment` runs the existing segmentation pipeline
- CLI-05 (partial): `mtg` entry point declared; `build-index` and `sync` added in plans 02 and 03
- CLI-06: existing `python -m mtg_recognition.engine.pipeline` and `python -m mtg_recognition.engine.segmentation.cli` invocations work unchanged

## Known Stubs

None — all wiring is live. `cli.add_command()` delegates directly to the existing `@click.command()` functions.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Self-Check: PASSED

- `src/mtg_recognition/cli.py` — FOUND
- `tests/unit/test_cli.py` — FOUND
- `pyproject.toml` contains `[project.scripts]` — FOUND
- Commit 62e4869 — FOUND
- Commit bc0d3f7 — FOUND
- Commit 3c30118 — FOUND
