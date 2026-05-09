# Phase 1: Dataset Versioning - Research

**Researched:** 2026-05-10
**Domain:** DVC (Data Version Control) + Google Drive remote + git large-file migration
**Confidence:** HIGH (core DVC mechanics verified via official docs; OAuth pitfall verified via GitHub issue)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** `dvc[gdrive]` added to `[dependency-groups] dev` in `pyproject.toml`. Collaborators get DVC via `uv sync --dev`.
- **D-02:** Move all of `benchmarks/dataset/` (images + OCR ground truth) to `data/test/6_benchmark/`. This includes `benchmarks/dataset/images/` and `benchmarks/dataset/ocr/`.
- **D-03:** After migration, gitignore `benchmarks/` entirely — treat all benchmark output as generated artifacts, remove any non-image benchmark content from git tracking.
- **D-04:** Planner determines the safest migration sequence to avoid data loss. Moving local files before `git rm --cached` is recommended.
- **D-05:** The 73 git-tracked files in `benchmarks/` must be removed via `git rm --cached` (no history rewrite).
- **D-06:** `data/test/` is already in `.gitignore` and NOT in git. DVC will track it as a new artifact — no git-cleanup needed for this path.
- **D-07:** DVC remote named `origin` (mirrors git convention).
- **D-08:** Google Drive folder path: `decksray/MTG-Recognition/data/test`. DVC will create this path if it doesn't exist.
- **D-09:** Authentication uses personal Google accounts (no service account).

### Claude's Discretion

- Migration sequence order (move files → git rm --cached → dvc add → dvc push, or similar): planner chooses the safest sequence.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DVC-01 | DVC is initialized (`dvc init`) | `dvc init` creates `.dvc/`, `.dvcignore` — must commit both to git |
| DVC-02 | Google Drive remote configured | `dvc remote add -d origin gdrive://root/decksray/MTG-Recognition/data/test` + custom OAuth app required (default DVC app blocked) |
| DVC-03 | `benchmarks/dataset/` removed from git tracking (no history rewrite) | `git rm -r --cached benchmarks/dataset/` + update `.gitignore` |
| DVC-04 | `benchmarks/dataset/` content moved to `data/test/6_benchmark/` | `mv benchmarks/dataset/* data/test/6_benchmark/` or equivalent |
| DVC-05 | `data/test/` tracked by DVC (`dvc add data/test`) | Creates `data/test.dvc` + `.gitignore` entry; `data/test/` already gitignored |
| DVC-06 | Dataset pushed to Google Drive (`dvc push`) | Requires auth to complete first; first push triggers browser OAuth |
| DVC-07 | Collaborator can `dvc pull` to get all datasets | Requires committing `data/test.dvc` to git so DVC knows what to pull |
| DVC-08 | `dvc push` workflow documented for contributors | CONTRIBUTING.md or README section |
</phase_requirements>

---

## Summary

This phase installs DVC into the project as a dev dependency, migrates 72 benchmark JPG images (plus 20 OCR ground-truth images currently only on disk) from `benchmarks/dataset/` into `data/test/6_benchmark/`, removes them from git tracking, tracks `data/test/` with DVC, and pushes the full dataset to a personal Google Drive folder. The result is a single `dvc pull` command for collaborators.

The migration is a Runtime State Inventory phase in miniature: `benchmarks/dataset/images/raw/` and `benchmarks/dataset/images/ready/` have 72 JPG files in git that must be untracked; `benchmarks/dataset/ocr/` has 20 JPG files on disk only (already gitignored) that still need to move. The `data/test/` tree (879 files across 7 categories) is already gitignored and will simply be added to DVC tracking.

**Critical pitfall:** The default DVC Google OAuth app is currently blocked by Google ("This app is blocked"). Users must create a custom Google Cloud Project with an OAuth Desktop app and configure `dvc remote modify` with `gdrive_client_id` and `gdrive_client_secret` before authentication will succeed. This is a required setup step, not optional.

**Primary recommendation:** Follow the sequence — install DVC, create custom GCP OAuth app, configure remote, move benchmark files, git rm --cached, dvc add data/test, dvc push, then commit all DVC metadata and .gitignore updates.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Package dependency management | Developer tooling (uv/pyproject.toml) | — | DVC is a dev dependency installed via uv |
| Data file versioning | DVC (local cache + .dvc files) | Git (tracks .dvc pointer files) | DVC owns data, git owns metadata pointers |
| Remote storage | Google Drive (via gdrive remote) | — | Decision locked to GDrive |
| Auth credentials | OS credential cache (pydrive2fs) | Google Cloud OAuth | Stored at `~/.cache/pydrive2fs/` on Linux |
| Git untracking | Git (rm --cached) | .gitignore | One-time operation, no history rewrite |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| dvc | 3.67.1 | Data versioning, remote push/pull | Industry standard ML data versioning tool |
| dvc-gdrive | 3.0.1 | Google Drive remote plugin for DVC | Official DVC remote plugin for GDrive |

[VERIFIED: PyPI — `curl -s https://pypi.org/pypi/dvc/json` + `https://pypi.org/pypi/dvc-gdrive/json`]

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pydrive2 | (pinned by dvc-gdrive) | Google OAuth + Drive API client | Installed transitively by dvc-gdrive |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Google Drive | S3/Azure/GCS | GDrive is free for small teams; decision locked |
| `dvc[gdrive]` extra | Manual `dvc-gdrive` install | Extra installs the plugin automatically; cleaner |

**Installation (exact pyproject.toml change):**

```toml
[dependency-groups]
dev = [
    "ipykernel>=7.1.0",
    "pytest>=9.0.2",
    "sqlacodegen>=3.2.0",
    "rapidocr>=3.7.0",
    "streamlit-webrtc>=0.64.5",
    "cv2-enumerate-cameras>=1.3.3",
    "pyarrow>=22.0.0",
    "ipyplot>=1.1.2",
    "dvc[gdrive]>=3.67.1",   # ← add this line
]
```

Then: `uv sync --dev`

---

## Architecture Patterns

### System Architecture Diagram

```
Developer machine
│
├── pyproject.toml  ──→  uv sync --dev  ──→  dvc (installed)
│
├── data/test/                        ← filesystem (gitignored)
│   ├── 0_error/ ... 5_scans_*/
│   └── 6_benchmark/                  ← migrated from benchmarks/dataset/
│         ├── images/raw/    (36 jpgs from benchmarks)
│         ├── images/ready/  (36 jpgs from benchmarks)
│         └── ocr/           (20 jpgs from benchmarks, disk-only)
│
├── data/test.dvc   ──────────────────→  git (tracked pointer file)
│
└── .dvc/config     ──────────────────→  git (remote config)
      remote.origin.url = gdrive://root/decksray/MTG-Recognition/data/test
      remote.origin.gdrive_client_id = <custom OAuth app ID>
      remote.origin.gdrive_client_secret = <stored separately>
            │
            ↓  dvc push
      Google Drive: My Drive/decksray/MTG-Recognition/data/test/
            │
            ↓  dvc pull (collaborator)
      Collaborator's data/test/   (all images restored)
```

### Recommended Project Structure (after phase)

```
.dvc/
├── .gitignore          # auto-created by dvc init (ignores /cache, /tmp)
└── config              # remote config (committed to git)
data/
└── test.dvc            # pointer file committed to git
benchmarks/             # gitignored entirely after migration
data/test/              # gitignored, DVC-tracked
└── 6_benchmark/        # new directory (migrated from benchmarks/dataset/)
    ├── images/
    │   ├── raw/        # 36 JPGs
    │   └── ready/      # 36 JPGs
    └── ocr/
        ├── common_layout/   # 4 JPGs
        └── uncommon_layout/ # 16 JPGs
```

### Pattern 1: Safe Migration Sequence

**What:** Order of operations that prevents data loss when moving git-tracked files to DVC.
**When to use:** Any time files are being moved out of git tracking into DVC.

```bash
# Source: doc.dvc.org/command-reference/add + iterative/dvc discuss.dvc.org/t/migration-tutorials/601

# Step 1: Move files first (while git still tracks them — safe to restore if interrupted)
mkdir -p data/test/6_benchmark
mv benchmarks/dataset/images data/test/6_benchmark/
mv benchmarks/dataset/ocr   data/test/6_benchmark/

# Step 2: Remove from git tracking (files already moved, no data loss risk)
git rm -r --cached benchmarks/dataset/

# Step 3: Update .gitignore to cover benchmarks/ entirely
# (replaces individual #benchmarks/**/*.jpg comment and benchmarks/dataset/ocr line)

# Step 4: DVC add the entire data/test directory
dvc add data/test
# Creates: data/test.dvc (commit this to git)
# Modifies: adds /data/test to .gitignore (harmless — already there)

# Step 5: Configure remote (if not already done)
dvc remote add -d origin "gdrive://root/decksray/MTG-Recognition/data/test"
dvc remote modify origin gdrive_client_id '<client-id>'
# gdrive_client_secret set via env or local config (not committed)

# Step 6: Push to Google Drive (triggers browser OAuth on first run)
dvc push

# Step 7: Commit all DVC metadata
git add .dvc/config data/test.dvc .gitignore pyproject.toml
git commit -m "chore(dvc): initialize DVC, migrate benchmark data, push to Google Drive"
```

[CITED: doc.dvc.org/command-reference/add, discuss.dvc.org/t/migration-tutorials/601]

### Pattern 2: Google Drive Remote with Custom OAuth App

**What:** Required setup because DVC's default OAuth app is blocked by Google.
**When to use:** Every new DVC + Google Drive setup as of late 2023+.

```bash
# Source: doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended

# 1. In Google Cloud Console:
#    - Create project, enable Google Drive API
#    - OAuth consent screen: "Desktop app", add your email as test user
#    - Create OAuth Client ID → download JSON → note client_id + client_secret

# 2. Configure remote (gdrive_client_secret goes in local config, NOT committed)
dvc remote add -d origin "gdrive://root/decksray/MTG-Recognition/data/test"
dvc remote modify origin gdrive_client_id '<your-client-id>'
dvc remote modify --local origin gdrive_client_secret '<your-client-secret>'

# --local writes to .dvc/config.local (gitignored), keeping secret out of git

# 3. First push triggers browser auth for each collaborator
dvc push   # opens browser → login → grant access → credentials cached at:
           # ~/.cache/pydrive2fs/{client_id}/default.json  (Linux)
```

[CITED: doc.dvc.org/user-guide/data-management/remote-storage/google-drive, github.com/treeverse/dvc/issues/10516]

### Anti-Patterns to Avoid

- **Running `git rm --cached` before moving files:** If interrupted, data is lost from git without existing elsewhere. Always move first, then untrack.
- **Using DVC's default OAuth app:** Google blocks it ("This app is blocked"). Custom GCP project is required.
- **Committing `gdrive_client_secret` to git:** Use `dvc remote modify --local` to write to `.dvc/config.local` which is gitignored by `dvc init`.
- **Using `gdrive://root` alone (no path):** Dumps DVC hash-named files directly into "My Drive" root — messy and harder to share. Always include a subfolder path.
- **Running `dvc add` before `git rm --cached`:** DVC will hash and cache the files, but git still tracks them — creates confusion until git status is cleaned up.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Large file remote storage | Custom S3/Drive sync scripts | DVC push/pull | DVC handles hashing, dedup, partial push, link types |
| Data pointer files | Custom manifest JSON | `.dvc` files | DVC pointer format is understood by all DVC commands |
| Credential caching | Custom OAuth token storage | pydrive2fs cache (via dvc-gdrive) | Handles token refresh, platform-specific paths |
| .gitignore management for data dirs | Manual edits | `dvc add` auto-updates .gitignore | DVC ensures the tracked dir is always gitignored |

**Key insight:** DVC's value is in the pointer-file + cache + remote triad. The `.dvc` file is the git artifact; the remote holds the data. Never bypass this by syncing data directly.

---

## Runtime State Inventory

> This is a migration phase — the inventory below covers everything that holds the `benchmarks/` path or related state.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | `benchmarks/dataset/images/raw/` — 36 JPGs in git | `git rm -r --cached` + move to `data/test/6_benchmark/` |
| Stored data | `benchmarks/dataset/images/ready/` — 36 JPGs in git | `git rm -r --cached` + move to `data/test/6_benchmark/` |
| Stored data | `benchmarks/dataset/ocr/` — 20 JPGs on disk only (gitignored) | Move to `data/test/6_benchmark/ocr/`, no git action needed |
| Live service config | None — no external services reference `benchmarks/` path | None |
| OS-registered state | None — no cron/task scheduler references found | None |
| Secrets/env vars | None — no env vars reference benchmark paths | None |
| Build artifacts | `benchmarks/pipeline_timings/` — `.py` file + `__pycache__/` + CSV results; NOT image data | Keep in git (benchmark scripts stay); ensure `benchmarks/` gitignore pattern excludes only the dataset subdirectory or covers all of benchmarks/ per D-03 |

**Key clarification on D-03 vs benchmark scripts:** Decision D-03 says gitignore `benchmarks/` entirely. However, `benchmarks/pipeline_timings/pipeline_timings.py` is currently tracked in git. The planner must decide: either (a) keep `benchmarks/pipeline_timings/pipeline_timings.py` in git and use a more targeted gitignore pattern (e.g., `benchmarks/dataset/`), or (b) remove `benchmarks/pipeline_timings/pipeline_timings.py` from git tracking too and gitignore all of `benchmarks/`. D-03 says "gitignore `benchmarks/` entirely" — this implies option (b), but the planner should confirm this interpretation does not lose needed scripts.

[VERIFIED: git ls-files benchmarks/ — local codebase inspection]

---

## Common Pitfalls

### Pitfall 1: Default DVC OAuth App Blocked

**What goes wrong:** Running `dvc push` with default DVC credentials shows "This app is blocked" and authentication fails — no data is pushed.
**Why it happens:** Google blocked DVC's OAuth app for policy compliance. Issue open since late 2023.
**How to avoid:** Set up a custom Google Cloud Project with a Desktop OAuth app before attempting any `dvc push`. Configure with `dvc remote modify origin gdrive_client_id`.
**Warning signs:** Browser opens to Google OAuth, shows "This app is blocked" instead of a consent screen.

[VERIFIED: github.com/treeverse/dvc/issues/10516]

### Pitfall 2: `gdrive://root/path` Requires Path to Already Exist

**What goes wrong:** If the Google Drive folder `decksray/MTG-Recognition/data/test` doesn't exist yet, DVC may or may not create it automatically.
**Why it happens:** The official docs say "the path component must point to an existing folder" for path-based remotes (when using a folder ID as base). For `root`-based paths the behavior is less documented.
**How to avoid:** Create the full folder hierarchy `decksray/MTG-Recognition/data/test` in Google Drive manually before first `dvc push`, or verify DVC creates it on first push with the `root`-based URL.
**Warning signs:** `dvc push` fails with a "folder not found" error.

[CITED: doc.dvc.org/user-guide/data-management/remote-storage/google-drive — "path component must point to an existing folder"]
[ASSUMED: auto-creation behavior for `gdrive://root/path` is unclear from docs — manual folder creation is the safe default]

### Pitfall 3: `gdrive_client_secret` Accidentally Committed

**What goes wrong:** `dvc remote modify origin gdrive_client_secret '...'` writes to `.dvc/config` which is committed to git — secret exposed.
**Why it happens:** `dvc remote modify` defaults to the shared config, not local config.
**How to avoid:** Always use `dvc remote modify --local origin gdrive_client_secret '...'` — writes to `.dvc/config.local` which `dvc init` automatically adds to `.dvc/.gitignore`.
**Warning signs:** `git diff --cached .dvc/config` shows `gdrive_client_secret`.

[CITED: doc.dvc.org/user-guide/data-management/remote-storage/google-drive]

### Pitfall 4: `dvc add data/test` Adds Redundant `.gitignore` Entry

**What goes wrong:** `dvc add data/test` tries to add `/data/test` to the root `.gitignore`. Since `data/test/` is already in `.gitignore`, this may result in a duplicate entry or a slightly different pattern (e.g., `/data/test` vs `data/test/`).
**Why it happens:** DVC always writes its own gitignore entry regardless of existing patterns. There is a known DVC issue (#7815) where DVC modifies `.gitignore` even when a directory pattern already exists.
**How to avoid:** After `dvc add`, review `.gitignore` for duplicates and clean them if needed. Functionally harmless but tidy to remove.
**Warning signs:** `.gitignore` contains both `data/test/` and `/data/test`.

[CITED: github.com/iterative/dvc/issues/7815]

### Pitfall 5: `data/test.dvc` Not Committed — Collaborators Cannot Pull

**What goes wrong:** `dvc pull` on a collaborator machine fails with "no data to pull" or "file not found in remote".
**Why it happens:** DVC uses `.dvc` pointer files to know what to fetch. If `data/test.dvc` is not committed to git, the remote mapping is unknown.
**How to avoid:** Always `git add data/test.dvc` and commit before announcing `dvc pull` works.
**Warning signs:** `dvc status` shows data present locally but `dvc pull` on a fresh clone fetches nothing.

[CITED: doc.dvc.org/start — "DVC stores information about the added file in a special .dvc file"]

### Pitfall 6: Moving Only `images/raw/` — Missing `images/ready/` and `ocr/`

**What goes wrong:** CONTEXT.md D-02 and the REQUIREMENTS confirm ALL of `benchmarks/dataset/` must move. Discovery shows three subdirectories: `images/raw/`, `images/ready/`, AND `ocr/`.
**Why it happens:** The requirements mention "images" prominently, easy to overlook `ocr/` since it's currently gitignored.
**How to avoid:** Move `benchmarks/dataset/` as a whole tree, not piecemeal. The `ocr/` directory has 20 files on disk that must also move or they are silently lost.
**Warning signs:** `data/test/6_benchmark/ocr/` is empty after migration.

[VERIFIED: find benchmarks/dataset -type f — local codebase inspection]

---

## Code Examples

### DVC Init Output (files to commit)

```bash
# Source: doc.dvc.org/command-reference/init
$ dvc init
Initialized DVC repository.

You can now commit the changes to git.

+---------------------------------------------------------------------+
| DVC has enabled anonymous aggregate usage analytics.               |
| ...                                                                |
+---------------------------------------------------------------------+

# Files created by dvc init (all must be committed to git):
# .dvc/.gitignore   — ignores /cache and /tmp inside .dvc/
# .dvc/config       — empty initially, populated by dvc remote add
# .dvcignore        — like .gitignore but for DVC operations
```

### DVC Remote Add (Google Drive path-based)

```bash
# Source: doc.dvc.org/user-guide/data-management/remote-storage/google-drive
$ dvc remote add -d origin "gdrive://root/decksray/MTG-Recognition/data/test"
Setting 'origin' as a default remote.

$ dvc remote modify origin gdrive_client_id '<your-oauth-client-id>'
$ dvc remote modify --local origin gdrive_client_secret '<your-oauth-client-secret>'
# --local → writes to .dvc/config.local (gitignored)
```

### DVC Add Directory

```bash
# Source: doc.dvc.org/command-reference/add
$ dvc add data/test
100% Adding...|████████████████████████████████|1/1 [00:15,  0.07file/s]

# Files created:
#   data/test.dvc   ← git-tracked pointer
#   data/.gitignore ← adds /test entry (may already be covered by root .gitignore)
```

### DVC Push (first time — triggers OAuth)

```bash
# Source: doc.dvc.org/command-reference/push
$ dvc push
# On first run: opens browser → Google OAuth consent → paste code back
# Credentials cached at: ~/.cache/pydrive2fs/{client_id}/default.json

# Subsequent runs: uses cached credentials automatically
$ dvc push
```

### Collaborator Setup

```bash
# Source: doc.dvc.org/command-reference/pull
$ git clone git@github.com:decksray/MTG-Recognition.git
$ cd MTG-Recognition
$ uv sync --dev          # installs dvc[gdrive]
# Collaborator must also authenticate (browser OAuth, one-time)
$ dvc pull               # downloads all data/test/ from Google Drive
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DVC default OAuth app | Custom GCP OAuth app required | Late 2023 | All new GDrive setups need custom app; cannot skip |
| `gdrive_user_credentials_file` in `.dvc/tmp/` | Platform cache at `~/.cache/pydrive2fs/` | DVC 3.x (pydrive2 → pydrive2fs) | Credentials no longer in project directory |
| `pip install dvc[gdrive]` | `uv add 'dvc[gdrive]'` or pyproject.toml entry | N/A (uv adoption) | Same package, different tool |

**Deprecated/outdated:**
- `dvc remote add ... gdrive://<folder-id>` using numeric folder IDs: still works but path-based URLs (`gdrive://root/path/`) are also supported for personal use.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | DVC auto-creates `decksray/MTG-Recognition/data/test` folder hierarchy in Google Drive on first `dvc push` with `gdrive://root/path` URL | Common Pitfalls #2, Code Examples | If auto-creation fails, `dvc push` errors; fix: create folders manually in Google Drive first |
| A2 | `benchmarks/pipeline_timings/pipeline_timings.py` should be removed from git when gitignoring `benchmarks/` entirely per D-03 | Runtime State Inventory | If the script is needed in git, use a more targeted gitignore pattern (`benchmarks/dataset/`) instead |

**Non-assumed claims:** All DVC command syntax, version numbers, credential cache paths, and OAuth pitfall verified via official docs or GitHub issues.

---

## Open Questions

1. **Does D-03 mean gitignore all of `benchmarks/` or only `benchmarks/dataset/`?**
   - What we know: D-03 says "gitignore `benchmarks/` entirely." One file `benchmarks/pipeline_timings/pipeline_timings.py` is currently in git and would be removed from tracking.
   - What's unclear: Whether the planner intends to drop this script from git or keep it.
   - Recommendation: Planner should add `benchmarks/` to `.gitignore` (D-03 is explicit), and do `git rm --cached benchmarks/pipeline_timings/pipeline_timings.py` as well. If the script is needed, Phase 2 or 3 could move it to `src/` or `scripts/`.

2. **Should `gdrive_client_id` be committed to `.dvc/config` or kept local?**
   - What we know: `gdrive_client_id` is not a secret (it is public in OAuth flows). DVC docs show it in the shared config. Only `gdrive_client_secret` must be kept local.
   - What's unclear: Team preference for keeping client_id in shared config.
   - Recommendation: Commit `gdrive_client_id` to `.dvc/config` (it's public). Keep `gdrive_client_secret` in `.dvc/config.local` (gitignored). Document this in CONTRIBUTING.md.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| uv | Install DVC, run project | Yes | 0.10.7 | — |
| Python 3.12 | DVC runtime | Yes | 3.12.3 | — |
| git | DVC init, git rm --cached | Yes | (system git) | — |
| dvc[gdrive] | All DVC operations | No — not yet installed | — | Install via `uv sync --dev` after pyproject.toml edit |
| Google Chrome/browser | OAuth flow for Google Drive auth | Yes (assumed on dev machine) | — | Any browser |
| Google Drive API | dvc push/pull | Requires GCP project setup | — | N/A — no alternative remote |
| Internet access | dvc push/pull to Google Drive | Yes (assumed) | — | — |

**Missing dependencies with no fallback:**
- Custom GCP OAuth credentials (`gdrive_client_id` + `gdrive_client_secret`) — must be created before `dvc push` can succeed.

**Missing dependencies with fallback:**
- `dvc[gdrive]` — not installed; installs in Wave 1 via `uv sync --dev` after pyproject.toml edit.

---

## Security Domain

> `security_enforcement` not explicitly set in config.json — treated as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | Yes | Google OAuth via custom GCP app (personal accounts) |
| V3 Session Management | No | Credentials cached by pydrive2fs (not session-based) |
| V4 Access Control | Partial | Google Drive folder shared to specific users only |
| V5 Input Validation | No | No user input in this phase |
| V6 Cryptography | No | OAuth handled by pydrive2fs/Google SDK |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| OAuth client secret leaked to git | Information Disclosure | `dvc remote modify --local` (writes to gitignored `.dvc/config.local`) |
| Google Drive folder shared with "anyone with link" | Information Disclosure | Share to specific users/emails only — "Anyone with a link" not guaranteed to work with DVC |
| GCP project JSON key file committed | Information Disclosure | JSON key files must never be committed — service account not used here, but note for future |

---

## Sources

### Primary (HIGH confidence)
- `doc.dvc.org/command-reference/add` — dvc add behavior, .dvc file format, directory tracking
- `doc.dvc.org/user-guide/data-management/remote-storage/google-drive` — gdrive URL format, OAuth flow, credential cache paths, custom GCP app steps
- `doc.dvc.org/start` — command sequence: init → add → remote add → push → pull
- PyPI `dvc` package — version 3.67.1 [VERIFIED via curl pypi.org/pypi/dvc/json]
- PyPI `dvc-gdrive` package — version 3.0.1 [VERIFIED via curl pypi.org/pypi/dvc-gdrive/json]

### Secondary (MEDIUM confidence)
- `github.com/treeverse/dvc/issues/10516` — "This app is blocked" OAuth issue; custom GCP app required [VERIFIED: issue is open, multiple confirming comments]
- `github.com/iterative/dvc/issues/7815` — dvc add modifies .gitignore even when directory pattern exists [CITED]
- `discuss.dvc.org/t/migration-tutorials/601` — migration sequence: move → git rm --cached → dvc add → push [CITED]

### Tertiary (LOW confidence)
- WebSearch results on DVC gdrive setup — corroborating, not primary

---

## Metadata

**Confidence breakdown:**
- Standard stack (DVC 3.67.1, dvc-gdrive 3.0.1): HIGH — verified via PyPI
- Core command syntax: HIGH — verified via official DVC docs
- OAuth "app blocked" pitfall: HIGH — verified via open GitHub issue with maintainer confirmation
- gdrive://root/path auto-creation behavior: LOW — not explicitly documented; assumed based on general DVC behavior

**Research date:** 2026-05-10
**Valid until:** 2026-08-10 (90 days — DVC is stable; OAuth situation may change if DVC re-verifies app)
