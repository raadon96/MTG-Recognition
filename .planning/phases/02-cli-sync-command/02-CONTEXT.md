# Phase 2: CLI & Sync Command - Context

**Gathered:** 2026-05-16
**Status:** Ready for planning

<domain>
## Phase Boundary

A unified `mtg` click group (`recognize`, `segment`, `build-index`, `sync`) declared in `[project.scripts]` in `pyproject.toml`, plus the `mtg sync` orchestrator that rebuilds the full reference data stack (MobileCache → reference images → pHash index) in one shot, with intelligent version-based skipping so re-runs only do real work.

</domain>

<decisions>
## Implementation Decisions

### Sync State Management
- **D-01:** State file lives at `data/reference/.mtg-sync-state.json`. Schema: `{"mobilecache_version": str|null, "images_version": str|null, "phash_version": str|null}`. `null` = failed mid-step or never built.
- **D-02:** Resilience pattern: write `null` for an artifact at the **start** of its step; write the resolved version string on **success**. A crash or failure mid-step leaves `null` in the file — the next `mtg sync` run sees it as "needs rebuild" and re-attempts automatically.
- **D-03:** Missing or corrupt state file → treat as all-`null` → run all steps (same behaviour as `--force`). This is the correct first-run experience.

### mtg sync — Step-by-Step Behaviour
- **D-04:** **Step 1 (MobileCache):** Fetch `https://mtgjson.com/api/v5/Meta.json` (lightweight HTTP GET, ~200 bytes) to get the current MTGJSON version. Compare against `mobilecache_version` in state file. If versions match → skip rebuild. If different (or state is `null`) → download full AllPrintings ZIP, rebuild MobileCache, update state. `--skip-db-check` flag bypasses the Meta.json fetch and forces the rebuild check to use local ZIP if present.
- **D-05:** **Step 2 (Reference images):** Explicit version check — skip the image download pass entirely if `images_version == current mobilecache_version` in state file. If different → run `build_database.download_raw_images()` (already incremental: skips files that exist on disk) → update `images_version`.
- **D-06:** **Step 3 (pHash index):** Skip rebuild if `phash_version == images_version` in state file. If different (or `null`) → call `PHashCardMatcher.build_index(force=False)` → update `phash_version`.

### mtg sync — Flags
- **D-07:** `mtg sync --force` bypasses all version checks and rebuilds all three artifacts unconditionally. Useful for clean-slate recovery and when the state file has become untrustworthy.
- **D-08:** `mtg sync --skip-db-check` bypasses only the Meta.json HTTP fetch (Step 1 freshness check). Useful when offline or when the user knows no new MTGJSON version has been released.

### mtg build-index — Standalone Flags
- **D-09:** `mtg build-index` exposes `--force` (default `False`: skip if pHash index files already exist) and `--workers` (default `cpu_count()-1`). Both map directly to `PHashCardMatcher.build_index(force=..., num_workers=...)`.
- **D-10:** `mtg build-index` performs a pre-flight check: if `data/reference/images/` is missing or empty, exit non-zero with a clear message directing the user to run `mtg sync` first.

### Wrapper Architecture
- **D-11:** `mtg recognize` and `mtg segment` use `cli.add_command(existing_main, name="...")` — Click's native subcommand registration. Inherits all existing options automatically (`-i`, `-o`, `--k-phash`, `-k` for recognize; `-i`, `-o`, `--mode` for segment). Zero code duplication. `mtg recognize --help` and `python -m mtg_recognition.engine.pipeline --help` show identical options.
- **D-12:** All CLI code lives in `src/mtg_recognition/cli.py` (flat module, not a package). The click group + sync orchestration logic fit in one file — consistent with existing top-level modules (`settings.py`, `database.py`).

### Sync Output Style
- **D-13:** `mtg sync` uses `rich` formatting at the orchestration layer — step headers (e.g. `--- Step 1: MobileCache ---`), inline version-check results, and explicit skip messages (e.g. `✓ MobileCache is up-to-date (5.3.0+20260302) — skipping rebuild`). The `build_database.py` internal `print()` calls pass through unchanged.
- **D-14:** Final rich summary panel after all steps complete: one-line status per artifact (version resolved, rebuilt, or skipped).
- **D-15:** pHash index build: the existing `tqdm` progress bar inside `PHashCardMatcher.build_index()` is preserved as-is. No wrapping with rich progress — avoids mixing output libraries.

### Claude's Discretion
- How to surface `build_database.main()` internals as step functions callable by the sync orchestrator (refactor into sub-functions vs import and call monolithic `main()` with wrapping) — planner decides the safest refactoring approach.
- `data/reference/` directory creation on first run (sync should `mkdir -p` if it doesn't exist).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Planning
- `.planning/PROJECT.md` — Core value, key decisions, out-of-scope items (no CI/CD, no `data/reference/` in DVC, full pHash rebuild acceptable)
- `.planning/REQUIREMENTS.md` — CLI-01 through CLI-06, SYNC-01 through SYNC-04 with acceptance criteria
- `.planning/ROADMAP.md` — Phase 2 success criteria (4 conditions that must be TRUE)

### Entry Points Being Wrapped
- `src/mtg_recognition/engine/pipeline.py` — `main` is a `@click.command()` with `-i`, `-o`, `--k-phash`, `-k`; used as `mtg recognize` subcommand via `add_command()`
- `src/mtg_recognition/engine/segmentation/cli.py` — `main` is a `@click.command()` with `-i`, `-o`, `--mode`; used as `mtg segment` subcommand via `add_command()`
- `src/mtg_recognition/engine/matching/phash_matcher.py` — `PHashCardMatcher.build_index(force=False, num_workers=cpu_count()-1)`; used by `mtg build-index` and `mtg sync` Step 3
- `src/mtg_recognition/jobs/build_database.py` — `main()` (plain function, not click-decorated); drives Steps 1+2 of `mtg sync` (MobileCache rebuild + image download). Needs refactoring into step functions.

### Project Configuration
- `pyproject.toml` — `[project.scripts]` section does not exist yet — Phase 2 creates it; `[dependency-groups]` pattern for reference

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `pipeline.py::main` (`@click.command()`) — registered directly as `mtg recognize` subcommand; no new code needed for the wrapper
- `segmentation/cli.py::main` (`@click.command()`) — registered directly as `mtg segment` subcommand; no new code needed for the wrapper
- `PHashCardMatcher.build_index(force, num_workers)` — both flags are already implemented; `mtg build-index` just exposes them as click options
- `build_database.download_raw_images(dest_engine)` — already incremental (skips existing files); used in sync Step 2
- `rich` (`>=14.3.2`) — already in project dependencies; used in `pipeline.py` for colored output; available for sync orchestration layer

### Established Patterns
- Click `@click.command()` already used in `pipeline.py` and `segmentation/cli.py` — new `cli.py` uses a `@click.group()` and `add_command()` to compose them
- `[dependency-groups] dev` in `pyproject.toml` — Phase 2 adds `[project.scripts]` section (does not exist today)
- No formatter/linter configured — match surrounding code style

### Integration Points
- `data/reference/.mtg-sync-state.json` is a new artifact created and owned by `mtg sync`. It must be gitignored (add to `.gitignore` alongside `data/reference/`).
- `build_database.main()` is currently a monolithic function — the planner must decide how to expose its steps (MobileCache rebuild vs image download) as separately callable units without breaking the existing `python -m mtg_recognition.jobs.build_database` invocation (CLI-06 backward-compat applies here too).
- `data/reference/` may not exist on first run — `mtg sync` must create it via `mkdir -p`.

</code_context>

<specifics>
## Specific Ideas

- MTGJSON version is read from the `meta` table of `AllPrintings.sqlite` (columns: `date`, `version`; e.g. `"5.3.0+20260302"`). The lightweight check uses `https://mtgjson.com/api/v5/Meta.json` to get this version without downloading the full ZIP.
- State file schema is intentionally minimal: `{"mobilecache_version": str|null, "images_version": str|null, "phash_version": str|null}`. No timestamps, no checksums — the MTGJSON version string is the single source of truth for all three artifacts.
- Skip message style: `✓ MobileCache is up-to-date (5.3.0+20260302) — skipping rebuild` (checkmark + version + reason).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 2-CLI & Sync Command*
*Context gathered: 2026-05-16*
