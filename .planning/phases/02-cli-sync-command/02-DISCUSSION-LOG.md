# Phase 2: CLI & Sync Command - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-16
**Phase:** 2-CLI & Sync Command
**Areas discussed:** Sync pHash rebuild, Wrapper depth, Sync output style

---

## Sync pHash Rebuild

| Option | Description | Selected |
|--------|-------------|----------|
| Always rebuild (force=True) | Every mtg sync tears down and rebuilds the pHash index from scratch | |
| Skip if exists (force=False) | Skip rebuild if index is already up-to-date; user runs mtg build-index --force when needed | ✓ |

**User's choice:** Skip if exists — `mtg sync` calls `build_index(force=False)`; skip if `phash_version == images_version` in state file.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Both --force and --workers | Expose both flags on mtg build-index (map directly to existing build_index() params) | ✓ |
| --force only | Expose only --force, hardcode workers to cpu_count()-1 | |

**User's choice:** Both `--force` and `--workers` on `mtg build-index`.

---

**User clarification (free-text):** The user defined a full 3-step sync design with version-based freshness checks:

1. **Step 1 (MobileCache):** Check if MobileCache is present and up-to-date with MTGJSON. If not, rebuild and record version.
2. **Step 2 (Images):** Check if reference images match current MobileCache version. If not, download missing images.
3. **Step 3 (pHash index):** Check if pHash index matches images version. If not, rebuild.

The user proposed a single `.mtg-sync-state.json` state file to track versions of all three artifacts, using the MTGJSON `meta` table version string as the canonical version identifier.

**Claude's assessment and pushback:** Claude identified a chicken-and-egg problem in Step 1: knowing the "current MTGJSON version" without downloading the full ZIP. Recommended using MTGJSON's lightweight `Meta.json` endpoint (`/api/v5/Meta.json`, ~200 bytes) for the freshness check. User accepted this approach and added a `--skip-db-check` escape hatch for offline use.

For Step 2, Claude noted the incremental download is already a near-instant no-op when nothing has changed, questioning whether an explicit version check adds enough value. User chose explicit version check anyway for consistency and predictability.

| Option | Description | Selected |
|--------|-------------|----------|
| Always check Meta.json | One tiny HTTP GET at startup; fail gracefully offline | ✓ |
| Skip HTTP check, rebuild every run | Simpler; matches current build_database.py behavior | |

| Option | Description | Selected |
|--------|-------------|----------|
| Skip explicitly (version check) | Skip image download pass if images_version matches mobilecache_version | ✓ |
| Always run incremental pass | Near-instant no-op when unchanged; simpler | |

**State file location:** `data/reference/.mtg-sync-state.json` (co-located with the data it describes).

**Notes:** User required that the state file reflect failures — e.g., if a step fails mid-run, the corresponding artifact version must be set to `null`. Claude proposed the "write null at start, write version at success" pattern to handle crashes automatically without a separate failure tracking field. User accepted.

| Option | Description | Selected |
|--------|-------------|----------|
| mtg sync --force (bypass all checks) | Force-rebuild all three artifacts unconditionally | ✓ |
| Step-level flags only | Rely on --skip-db-check + manual state file deletion | |

| Option | Description | Selected |
|--------|-------------|----------|
| All-null on missing/corrupt state | Run all steps (safe first-run behavior) | ✓ |
| Abort with error | Require explicit --force on first run | |

---

## Wrapper Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Direct subcommand via add_command() | Inherit all existing click options; zero duplication | ✓ |
| Minimal shim | Expose only -i; hardcode defaults; simpler help output | |

**User's choice:** `cli.add_command(pipeline_main, "recognize")` — full option inheritance.

---

| Option | Description | Selected |
|--------|-------------|----------|
| src/mtg_recognition/cli.py (flat module) | Single file for click group + sync orchestration; follows existing pattern | ✓ |
| src/mtg_recognition/cli/ package | Separate files per command; more organized but adds directory overhead | |

**User's choice:** Flat `cli.py` module.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Fail fast with clear error | Check data/reference/images/ before starting; "run mtg sync first" | ✓ |
| Let PHashCardMatcher raise naturally | ValueError already raised if no images found | |

**User's choice:** Explicit pre-flight check with actionable error message.

---

## Sync Output Style

| Option | Description | Selected |
|--------|-------------|----------|
| Rich formatting | Step panels/headers, version-check results, skip messages | ✓ |
| Plain print (current style) | Bare print() statements consistent with build_database.py | |

**User's choice:** Rich formatting at the orchestration layer; `build_database.py` internal print() calls unchanged.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit skip message | e.g. ✓ MobileCache is up-to-date (5.3.0+20260302) — skipping rebuild | ✓ |
| Silent skip | No output when a step is skipped | |

**User's choice:** Explicit skip message per step.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Final summary panel (one-line per artifact) | Rebuilt/skipped + version at end of run | ✓ |
| No summary | Step output is sufficient | |

**User's choice:** Final rich summary panel.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Keep existing tqdm bar | PHashCardMatcher already has tqdm internally; pass through unchanged | ✓ |
| Replace with rich progress bar | Wrap build_index() with rich.progress; suppresses tqdm | |

**User's choice:** Keep tqdm — no mixing of output libraries.

---

## Claude's Discretion

- How to refactor `build_database.main()` into step functions callable by the sync orchestrator — planner decides the safest refactoring approach
- `data/reference/` directory creation on first run — sync should `mkdir -p`
- Exact rich formatting details (panel borders, colors, icons) — match pipeline.py style

## Deferred Ideas

None — discussion stayed within phase scope.
