# Phase 1: Dataset Versioning - Context

**Gathered:** 2026-05-10
**Status:** Ready for planning

<domain>
## Phase Boundary

All test and benchmark image data is tracked by DVC with a Google Drive remote, and no large files remain in git tracking. After this phase, collaborators can run `dvc pull` to get all datasets locally from a single Google Drive remote.

</domain>

<decisions>
## Implementation Decisions

### DVC Installation
- **D-01:** `dvc[gdrive]` is added to the `dev` dependency group in `pyproject.toml` (alongside ipykernel, sqlacodegen, etc.). Collaborators get DVC automatically via `uv sync --dev` — no separate install step.

### benchmarks/ Migration
- **D-02:** Move **all** of `benchmarks/dataset/` (images + OCR ground truth) to `data/test/6_benchmark/`. This includes `benchmarks/dataset/images/` and `benchmarks/dataset/ocr/`.
- **D-03:** After migration, gitignore `benchmarks/` entirely — treat all benchmark output as generated artifacts, remove any non-image benchmark content from git tracking.
- **D-04:** Planner determines the safest migration sequence to avoid data loss. Moving local files before `git rm --cached` is recommended to ensure no data loss.
- **D-05:** The 73 JPG files currently in `git ls-files benchmarks/dataset/images/raw/` must be removed from git tracking via `git rm --cached` (no history rewrite — gitignore going forward).
- **D-06:** Note: `data/test/` is already in `.gitignore` and NOT currently tracked in git. DVC will track this directory as a new artifact — no git-cleanup needed for `data/test/` itself.

### Google Drive Remote
- **D-07:** DVC remote named `origin` (mirrors git convention).
- **D-08:** Google Drive folder path: `decksray/MTG-Recognition/data/test` (nested under the decksray org folder). DVC will create this path if it doesn't exist.
- **D-09:** Authentication uses personal Google accounts (acceptable for this small team — no service account needed).

### Claude's Discretion
- Migration sequence order (move files → git rm --cached → dvc add → dvc push, or similar): planner chooses the safest sequence.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Planning
- `.planning/PROJECT.md` — Core value, key decisions, out-of-scope items (no history rewrite, no reference images in DVC)
- `.planning/REQUIREMENTS.md` — DVC-01 through DVC-08 (all Phase 1 requirements with acceptance criteria)
- `.planning/ROADMAP.md` — Phase 1 success criteria (5 conditions that must be TRUE)

### Codebase
- `pyproject.toml` — Current dependencies and dev group structure; DVC goes in `[dependency-groups] dev`
- `.gitignore` — Current exclusion patterns; `data/test/` is already ignored, `benchmarks/dataset/ocr` is excluded, `#benchmarks/**/*.jpg` is commented out (will need updating)
- `src/mtg_recognition/settings.py` — `DATA_DIR` setting that pipeline uses to locate `data/reference/` — do NOT modify; `data/test/` is a separate path

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/mtg_recognition/jobs/build_database.py` — The database build job; `mtg sync` in Phase 2 will wrap this. No changes needed in Phase 1.
- `src/mtg_recognition/engine/matching/phash_matcher.py` — `PHashCardMatcher.build_index()` used by `mtg build-index` in Phase 2. No changes in Phase 1.

### Established Patterns
- `uv` with `[dependency-groups]` for extras — DVC goes in `dev` group following this existing pattern
- No `[project.scripts]` section yet in `pyproject.toml` — Phase 2 adds this

### Integration Points
- `data/test/` is the single DVC root. All 7 existing categories + new `6_benchmark/` will be tracked under one `dvc add data/test`.
- `benchmarks/` is currently in git — migration must use `git rm --cached` to untrack, then add `benchmarks/` to `.gitignore`.

### Key Discovery
- `data/test/` already excluded from git via `.gitignore` — the DVC tracking work is additive (no git cleanup needed for this path, only for `benchmarks/`).
- `benchmarks/dataset/images/raw/` has exactly 73 files currently in git that need to be removed.

</code_context>

<specifics>
## Specific Ideas

- Google Drive folder path is explicitly `decksray/MTG-Recognition/data/test` (user specified)
- DVC remote name: `origin`
- All benchmark data (images + OCR ground truth) consolidates under `data/test/6_benchmark/`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 1-Dataset Versioning*
*Context gathered: 2026-05-10*
