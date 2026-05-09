# Phase 1: Dataset Versioning - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-10
**Phase:** 1-Dataset Versioning
**Areas discussed:** DVC install method, benchmarks/ post-migration, Google Drive folder layout

---

## DVC Install Method

| Option | Description | Selected |
|--------|-------------|----------|
| Dev dependency (pyproject.toml) | `dvc[gdrive]` added to dev group; collaborators get it via `uv sync --dev`. Adds ~50 MB of GDrive/DVC deps to venv. | ✓ |
| Standalone install (separate step) | Collaborators install DVC separately (e.g., `pip install dvc[gdrive]`). Documented in README. Keeps Python env cleaner. | |

**User's choice:** Dev dependency in existing `dev` group in `pyproject.toml`
**Notes:** Prioritizes frictionless collaborator setup over env cleanliness.

---

## benchmarks/ Post-Migration

**Q1: What happens to benchmarks/ after migration?**

| Option | Description | Selected |
|--------|-------------|----------|
| Keep benchmarks/ in git (non-image content) | pipeline_timings/ and non-image files stay in git; benchmarks/dataset/ gitignored | |
| Gitignore benchmarks/ entirely | All benchmark output treated as generated artifacts; removed from git | ✓ |
| Move pipeline_timings/ to data/test/ | Consolidate all dataset content under data/test/ and DVC | |

**Q2: Migration sequence?**

| Option | Description | Selected |
|--------|-------------|----------|
| Move first, then git rm --cached | Move images to data/test/6_benchmark/ before removing from git | |
| You decide the order | Let planner determine safest sequence | ✓ |

**Q3: Should benchmarks/dataset/ocr/ also move?**

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — move everything under benchmarks/dataset/ | All benchmark data (images + OCR ground truth) moves to data/test/6_benchmark/ | ✓ |
| No — images only | Only move benchmarks/dataset/images/ | |

**User's choices:** Gitignore benchmarks/ entirely; move ALL of benchmarks/dataset/ (images + ocr); planner decides safe sequence.
**Notes:** User wants a clean cut — no benchmark content remaining in git.

---

## Google Drive Folder Layout

**Q1: Where in Drive should DVC store the dataset?**

| Option | Description | Selected |
|--------|-------------|----------|
| Root-level folder | Single folder at Drive root, e.g. 'MTG-Recognition-DVC' | |
| Nested under project folder | e.g., 'decksray/MTG-Recognition/dvc-data' | ✓ |
| You decide | Claude picks a sensible default | |

**Q2: What should the nested path be?**

| Option | Description | Selected |
|--------|-------------|----------|
| decksray/MTG-Recognition/dvc-data | Matches repo name and decksray org umbrella | |
| MTG-Recognition/dvc-data | One level shallower | |
| Other | User specified custom path | ✓ |

**User's choice for path:** `decksray/MTG-Recognition/data/test`

**Q3: DVC remote name?**

| Option | Description | Selected |
|--------|-------------|----------|
| gdrive | Explicit, makes remote type obvious | |
| origin | Mirrors git convention | ✓ |
| storage | Neutral, works with multiple remotes | |

**User's choices:** Nested path `decksray/MTG-Recognition/data/test`; remote name `origin`.
**Notes:** User specified the exact Drive path directly.

---

## Claude's Discretion

- Migration sequence order for benchmarks/dataset/ → data/test/6_benchmark/: planner determines safest steps (move local files → git rm --cached → dvc add → dvc push is recommended pattern)

## Deferred Ideas

None — discussion stayed within Phase 1 scope.
