---
phase: 01-dataset-versioning
reviewed: 2026-05-10T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - .dvc/.gitignore
  - .dvc/config
  - .dvcignore
  - .gitignore
  - CONTRIBUTING.md
  - data/.gitignore
  - data/test.dvc
  - pyproject.toml
  - tests/regression/segmentation/test_rembg_reg.py
findings:
  critical: 1
  warning: 2
  info: 2
  total: 5
status: issues_found
---

# Phase 01: Dataset Versioning — Code Review Report

**Reviewed:** 2026-05-10
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

This phase adds DVC-based dataset versioning, migrates the benchmark dataset from `benchmarks/` to `data/test/6_benchmark/`, updates the corresponding regression test path, adds `dvc[gdrive]>=3.67.1` to the dev dependency group, and documents the contributor workflow in CONTRIBUTING.md.

The DVC config and gitignore scaffolding are structurally correct: `config.local` is excluded from git via `.dvc/.gitignore`, the committed `.dvc/config` contains only the public `gdrive_client_id` (no secret), and `data/test/` is excluded from git tracking by both `data/.gitignore` and the root `.gitignore`. The regression test path update is accurate.

One critical finding: CONTRIBUTING.md tells contributors to create their own GCP OAuth app (step 4) but then only provides the command to set `gdrive_client_secret` locally — it omits the corresponding `gdrive_client_id` override. A contributor following the guide verbatim will pair their own `client_secret` with the maintainer's `client_id`, producing an invalid OAuth credential pair that will fail authentication. Two warnings cover a redundant gitignore rule and a fragile relative-path assumption in the regression test.

---

## Critical Issues

### CR-01: CONTRIBUTING.md — missing `gdrive_client_id` local override causes OAuth failure

**File:** `CONTRIBUTING.md:16-23`

**Issue:** Step 4 of the one-time setup instructs contributors to create their own GCP Desktop OAuth application and note both `client_id` and `client_secret`. Step 5 then provides only one `dvc remote modify` command:

```bash
uv run dvc remote modify --local origin gdrive_client_secret '<your-client-secret>'
```

This sets the contributor's `client_secret` locally, but leaves `gdrive_client_id` pointing at the maintainer's OAuth app (committed in `.dvc/config`). A `client_id` and `client_secret` must belong to the same GCP application — mixing them produces an OAuth error at the Google consent screen ("This app is blocked" or an `invalid_client` 401).

The guide is internally inconsistent: it tells contributors to create their own credentials but then only configures half of them. The two supported models are:

**Option A — shared OAuth app (single maintainer's credentials):** Contributors do NOT create their own GCP app. The maintainer's `gdrive_client_id` is already in `.dvc/config`; only the matching `gdrive_client_secret` needs to be shared (e.g., via a team secret store). Step 4 should be removed or rewritten to "obtain the shared `client_secret` from a project maintainer."

**Option B — per-contributor OAuth app:** Contributors create their own GCP project and OAuth credentials. Both `client_id` and `client_secret` must be set locally. The setup command block must include:

```bash
uv run dvc remote modify --local origin gdrive_client_id '<your-client-id>'
uv run dvc remote modify --local origin gdrive_client_secret '<your-client-secret>'
```

Choose one model and update CONTRIBUTING.md to be consistent throughout, including the "This app is blocked" troubleshooting entry (line 64), which currently mentions both fields, suggesting Option B is the intent.

---

## Warnings

### WR-01: Redundant `data/test/` exclusion in root `.gitignore`

**File:** `.gitignore:8`

**Issue:** `data/test/` is already excluded by `data/.gitignore` (line 1: `/test`). The root `.gitignore` also has `data/test/` on line 8. The DVC-generated `data/.gitignore` is the canonical location for excluding DVC-tracked paths — it is generated automatically by `dvc add` and should be the single source of truth. Keeping a duplicate entry in the root `.gitignore` creates maintenance risk: if the DVC-tracked path is ever renamed (e.g., a second DVC-tracked dataset added), the root entry will silently persist and may exclude unintended paths.

**Fix:** Remove line 8 (`data/test/`) from `.gitignore`. The entry in `data/.gitignore` (generated and managed by DVC) is sufficient.

### WR-02: Regression test uses CWD-relative paths without a rootdir anchor

**File:** `tests/regression/segmentation/test_rembg_reg.py:23,29`

**Issue:** `FIXTURE_DIR` and `BENCHMARK_RAW_DIR` are constructed with bare relative `Path(...)` strings:

```python
FIXTURE_DIR = Path("tests/regression/segmentation/fixtures")   # line 23
BENCHMARK_RAW_DIR = Path("data/test/6_benchmark/images/raw")   # line 29
```

These resolve relative to the process working directory at test invocation time. When pytest is run from the project root (as documented) they work correctly. When run from any other directory — e.g., `cd tests && uv run pytest` or from an IDE that sets the working directory to the test file's directory — both paths silently resolve to non-existent locations. The `FIXTURE_DIR` case degrades gracefully (parametrize returns `[]`, tests are skipped), but `BENCHMARK_RAW_DIR` produces a hard `assert image_paths` failure (line 282) with a confusing "No benchmark images found" message rather than a clear path error.

**Fix:** Anchor paths to the repository root using the test file's `__file__`:

```python
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../MTG-Recognition/
FIXTURE_DIR = _REPO_ROOT / "tests/regression/segmentation/fixtures"
BENCHMARK_RAW_DIR = _REPO_ROOT / "data/test/6_benchmark/images/raw"
```

Alternatively, add a `conftest.py` at the project root that sets `rootdir` via a `pytest_configure` hook, or use `pyproject.toml`'s `[tool.pytest.ini_options] rootdir` setting.

---

## Info

### IN-01: CONTRIBUTING.md — inaccurate directory count after `dvc pull`

**File:** `CONTRIBUTING.md:36`

**Issue:** The pull description states: "The pull will populate `data/test/` with all 7 image categories plus the migrated `data/test/6_benchmark/`." The actual directory layout has 8 directories: `0_error`, `0_failure_cases`, `1_photos_single_card`, `2_photos_multiple_cards`, `3_scans_single_card_with_bg`, `4_scans_single_card_no_bg`, `5_scans_multiple_cards`, and `6_benchmark`. Additionally, `6_benchmark` is listed separately ("plus the migrated…") as if it is not one of the categories, which is confusing — it is in fact one of the 8 directories.

**Fix:** Update the sentence to accurately reflect the directory count and remove the redundant qualification:

```
The pull will populate `data/test/` with 8 subdirectories:
`0_error`, `0_failure_cases`, `1_photos_single_card`, `2_photos_multiple_cards`,
`3_scans_single_card_with_bg`, `4_scans_single_card_no_bg`, `5_scans_multiple_cards`,
and `6_benchmark` (migrated from `benchmarks/dataset/`).
```

### IN-02: `pyproject.toml` — `dvc[gdrive]` version constraint uses `>=` without upper bound

**File:** `pyproject.toml:55`

**Issue:** `"dvc[gdrive]>=3.67.1"` allows any future major version. DVC has historically introduced breaking changes to remote configuration schemas across minor and major versions (e.g., gdrive authentication methods changed significantly between 2.x and 3.x). For a dataset-management tool that is invoked explicitly rather than imported as a library, pinning to a known-compatible minor series (e.g., `>=3.67.1,<4`) provides more stable developer environments.

**Fix:**

```toml
"dvc[gdrive]>=3.67.1,<4",
```

This is an informational suggestion; the current unbounded constraint is not incorrect and is consistent with how other dev dependencies are pinned in this project.

---

_Reviewed: 2026-05-10_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
