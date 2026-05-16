# Phase 2: CLI & Sync Command - Pattern Map

**Mapped:** 2026-05-16
**Files analyzed:** 4 (2 new, 2 modified)
**Analogs found:** 4 / 4

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/mtg_recognition/cli.py` | utility / CLI orchestrator | request-response + batch | `src/mtg_recognition/engine/pipeline.py` | role-match |
| `src/mtg_recognition/jobs/build_database.py` | service / job | batch + file-I/O | self (refactor only) | exact |
| `pyproject.toml` | config | — | self (add section) | exact |
| `.gitignore` | config | — | self (add entry) | exact |

---

## Pattern Assignments

### `src/mtg_recognition/cli.py` (utility, request-response + batch)

**Primary analog:** `src/mtg_recognition/engine/pipeline.py`
**Secondary analog:** `src/mtg_recognition/engine/segmentation/cli.py`

---

#### Imports pattern

Copy this import block style from `pipeline.py` (lines 1-17) — note `from __future__ import annotations` is used in `segmentation/cli.py` (line 25) but not `pipeline.py`; match the surrounding file style (no `from __future__` in top-level modules):

```python
import json
from pathlib import Path
from multiprocessing import cpu_count
from urllib import request

import click
from rich.console import Console
from rich.panel import Panel

from mtg_recognition.settings import settings
from mtg_recognition.engine.pipeline import main as _recognize_main
from mtg_recognition.engine.segmentation.cli import main as _segment_main
from mtg_recognition.engine.matching.phash_matcher import PHashCardMatcher
from mtg_recognition.jobs import build_database
```

Key observations:
- `rich` is already a project dependency (`rich>=14.3.2`) — `pipeline.py` imports `from rich import print` (line 9). `cli.py` should use `rich.console.Console` for the orchestration layer output so it can use `console.print()` with markup and `Panel`.
- `click` is already a dependency (`click>=8.3.1`) and used in both `pipeline.py` and `segmentation/cli.py`.
- All internal imports use the full `mtg_recognition.*` package path — never relative imports at the top-level module.

---

#### Click group pattern

**Analog:** neither existing file uses `@click.group()` — this is new. However, `add_command()` is the mechanism described in D-11. Pattern to follow:

```python
console = Console()

@click.group()
def cli() -> None:
    """MTG card recognition toolkit."""
    pass

# Register existing commands directly — zero code duplication (D-11)
cli.add_command(_recognize_main, name="recognize")
cli.add_command(_segment_main, name="segment")
```

This pattern mirrors Click's documented `add_command()` usage, which the project decisions locked in (D-11). The `_recognize_main` and `_segment_main` aliases are used so the local name does not shadow the `main` guard at the bottom of `cli.py`.

---

#### Click command option pattern

Copy `@click.option` decorator style from `pipeline.py` (lines 153-184) verbatim — note the conventions:
- Short flag first (`"-i"`), long flag second (`"--input"`), then `"dest_name"` as third positional
- `type=click.Path(..., path_type=Path)` for path arguments
- `show_default=True` on numeric options
- `help=` string on every option

```python
# From pipeline.py lines 153-184 — copy this decorator style for build-index and sync
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=True, readable=True, path_type=Path
    ),
    required=True,
    help="Path to a single image file or a directory of image files.",
)
@click.option(
    "--k-phash",
    type=click.INT,
    default=20,
    show_default=True,
    help="Number of pHash candidates for the coarse retrieval stage.",
)
```

For `build-index`:
```python
@cli.command("build-index")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Rebuild pHash index even if it already exists.",
)
@click.option(
    "--workers",
    type=click.INT,
    default=cpu_count() - 1,
    show_default=True,
    help="Number of parallel worker processes for hash computation.",
)
def build_index(force: bool, workers: int) -> None:
    ...
```

---

#### Error / pre-flight check pattern

Copy error exit pattern from `segmentation/cli.py` (lines 74-76) and `pipeline.py` (lines 204-207):

```python
# segmentation/cli.py lines 74-76
if not img_paths:
    click.echo(f"No image files found in: {input_path}", err=True)
    raise SystemExit(1)

# pipeline.py lines 204-207 — same pattern
if not img_paths:
    click.echo(f"No image files found in: {input_path}", err=True)
    raise SystemExit(1)
```

Apply for `build-index` pre-flight (D-10):

```python
img_dir = settings.IMG_DIR
if not img_dir.exists() or not any(img_dir.iterdir()):
    click.echo(
        "Error: data/reference/images/ is missing or empty. Run 'mtg sync' first.",
        err=True,
    )
    raise SystemExit(1)
```

---

#### Rich output pattern (sync step headers + skip messages)

`pipeline.py` uses `from rich import print` (line 9) for colored terminal output. The sync orchestrator needs richer formatting (D-13, D-14). Use `Console` instance explicitly to avoid overriding `builtins.print` in a module that other code imports:

```python
# Instantiate once at module level
console = Console()

# Step header (D-13)
console.print("\n[bold]--- Step 1: MobileCache ---[/bold]")

# Skip message with checkmark (D-13, specifics section)
console.print(f"[green]✓ MobileCache is up-to-date ({version}) — skipping rebuild[/green]")

# Final summary panel (D-14)
console.print(Panel(summary_text, title="Sync Summary", expand=False))
```

---

#### `if __name__ == "__main__"` guard pattern

Both existing CLIs end with this identical guard — copy it:

```python
# pipeline.py line 324-325
if __name__ == "__main__":
    main()

# segmentation/cli.py line 114-115
if __name__ == "__main__":
    main()
```

For `cli.py` the guard calls `cli()` (the group):

```python
if __name__ == "__main__":
    cli()
```

---

#### State file read/write pattern

No analog exists for JSON state file in this codebase. Use stdlib `json` + `pathlib` as already done in `pipeline.py` (lines 1, 320):

```python
# pipeline.py line 320 — write JSON
output_file.write_text(json.dumps(all_results, indent=2))

# Adapt for state file (D-01, D-02, D-03):
STATE_FILE = settings.REF_DIR / ".mtg-sync-state.json"

def _load_state() -> dict:
    """Return state dict. Missing or corrupt file → all-null (D-03)."""
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"mobilecache_version": None, "images_version": None, "phash_version": None}

def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))
```

The null-before/version-on-success resilience pattern (D-02) requires two writes per step:

```python
# Start of step — mark as in-progress (null)
state["mobilecache_version"] = None
_save_state(state)

# ... do the work ...

# Success — write resolved version
state["mobilecache_version"] = resolved_version
_save_state(state)
```

---

### `src/mtg_recognition/jobs/build_database.py` (service/job, batch + file-I/O)

**Analog:** self — this is a refactor of the existing file. The goal is to extract step functions from `main()` without breaking backward compatibility.

---

#### Current `main()` structure (lines 261-289)

```python
def main():
    print("--- Step 1: Retrieving Data ---")
    if not settings.MTGO_DB_FILE_ZIP.exists():
        print(f"Downloading {settings.MTGJSON_URL}...")
        req = request.Request(settings.MTGJSON_URL, headers=SCRYFALL_HEADERS)
        with (
            request.urlopen(req) as response,
            open(settings.MTGO_DB_FILE_ZIP, "wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)

    print("Extracting...")
    with zipfile.ZipFile(settings.MTGO_DB_FILE_ZIP, "r") as zip_ref:
        zip_ref.extractall(settings.TMP_DIR)

    print("\n--- Step 2: Building Local Database ---")
    source_engine = create_engine(f"sqlite:///{settings.MTGO_DB_FILE}")
    dest_engine = create_engine(f"sqlite:///{settings.LOCAL_DB_FILE}")
    create_database(source_engine, dest_engine, force=True)

    print("\n--- Step 3: Downloading Images ---")
    download_raw_images(dest_engine)
```

---

#### Refactoring approach (Claude's Discretion from CONTEXT.md)

Extract two new named functions from `main()` — keep `main()` intact and calling them:

```python
def build_mobilecache() -> None:
    """Download AllPrintings ZIP + rebuild MobileCache. Step 1+2 of the original main()."""
    print("--- Step 1: Retrieving Data ---")
    if not settings.MTGO_DB_FILE_ZIP.exists():
        print(f"Downloading {settings.MTGJSON_URL}...")
        req = request.Request(settings.MTGJSON_URL, headers=SCRYFALL_HEADERS)
        with (
            request.urlopen(req) as response,
            open(settings.MTGO_DB_FILE_ZIP, "wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)

    print("Extracting...")
    with zipfile.ZipFile(settings.MTGO_DB_FILE_ZIP, "r") as zip_ref:
        zip_ref.extractall(settings.TMP_DIR)

    print("\n--- Step 2: Building Local Database ---")
    source_engine = create_engine(f"sqlite:///{settings.MTGO_DB_FILE}")
    dest_engine = create_engine(f"sqlite:///{settings.LOCAL_DB_FILE}")
    create_database(source_engine, dest_engine, force=True)
    return dest_engine  # caller needs engine for image download step


def download_images(dest_engine=None) -> None:
    """Download reference images. Step 3 of the original main(). (D-05: already incremental)"""
    if dest_engine is None:
        dest_engine = create_engine(f"sqlite:///{settings.LOCAL_DB_FILE}")
    print("\n--- Step 3: Downloading Images ---")
    download_raw_images(dest_engine)


def main():
    """Backward-compatible entry point. Called by 'python -m mtg_recognition.jobs.build_database'."""
    dest_engine = build_mobilecache()
    download_images(dest_engine)
```

Key observations from the existing file:
- `create_database()` already has a `force` parameter (line 45) — `build_mobilecache()` calls it with `force=True` (matching current `main()` behavior).
- `download_raw_images()` (line 251) is already a named function and already incremental (skips existing files per line 149) — it is called directly by `download_images()`.
- `SCRYFALL_HEADERS` (line 16) is already a module-level constant — accessible from the new step functions without changes.
- No decorators on any existing job functions — maintain this plain-function style.

---

#### MTGJSON version fetch pattern (new — no analog in codebase)

The Meta.json fetch in Step 1 of `mtg sync` (D-04) uses the same HTTP pattern already in `main()` (lines 265-270) via `urllib.request`. The version check is lightweight:

```python
# Pattern from build_database.py lines 265-270 — adapt for Meta.json fetch
import json
from urllib import request as urllib_request

META_URL = "https://mtgjson.com/api/v5/Meta.json"

def fetch_mtgjson_version() -> str:
    """Fetch current MTGJSON version via lightweight GET (~200 bytes). (D-04)"""
    req = urllib_request.Request(META_URL, headers=SCRYFALL_HEADERS)
    with urllib_request.urlopen(req) as response:
        data = json.loads(response.read())
    return data["meta"]["version"]
```

This function belongs in `cli.py` (the sync orchestrator), not in `build_database.py`, since the version check is a sync-layer concern.

---

### `pyproject.toml` (config — add `[project.scripts]` section)

**Analog:** the existing `pyproject.toml` — no `[project.scripts]` section exists yet.

**Pattern:** follow the TOML section style already in the file. Insert the new section after `[project]` and before `[dependency-groups]`:

```toml
[project.scripts]
mtg = "mtg_recognition.cli:cli"
```

The `cli` name on the right side refers to the `@click.group()` object named `cli` in `src/mtg_recognition/cli.py`. The `[tool.uv.build-backend]` section (lines 88-90) already declares `module-root = "src"`, so `mtg_recognition.cli` resolves correctly as `src/mtg_recognition/cli.py`.

Existing `[tool.uv]` section (line 62-63) confirms `package = true`, which is required for `[project.scripts]` to work with uv.

---

### `.gitignore` (config — add state file entry)

**Analog:** the existing `.gitignore`, specifically the "Reference cards dataset" block (lines 6-10):

```gitignore
# Reference cards dataset
data/reference/images
data/test/

data/reference/tmp
data/reference/download_errors.log
```

**Pattern:** add the state file entry immediately after the existing `data/reference/` entries, keeping it in the same comment block:

```gitignore
# Reference cards dataset
data/reference/images
data/test/

data/reference/tmp
data/reference/download_errors.log
data/reference/.mtg-sync-state.json
```

Note: `data/reference/` itself is NOT listed in `.gitignore` (only its subdirectories/files are), meaning `MobileCache.sqlite` and the index files are also not tracked but not explicitly listed here. The state file follows this same pattern — one explicit entry per artifact.

---

## Shared Patterns

### `mkdir -p` on first run

**Source:** `segmentation/cli.py` line 78; `pipeline.py` lines 208-209; `phash_matcher.py` lines 155-156

All three files use `Path.mkdir(parents=True, exist_ok=True)` for directory creation:

```python
# segmentation/cli.py line 78
output_dir.mkdir(parents=True, exist_ok=True)

# pipeline.py lines 208-209
if output_file is not None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

# phash_matcher.py line 155
self.index_path.parent.mkdir(parents=True, exist_ok=True)
```

Apply to `mtg sync` for `data/reference/` creation on first run (Claude's Discretion from CONTEXT.md):

```python
settings.REF_DIR.mkdir(parents=True, exist_ok=True)
```

---

### `settings` singleton import pattern

**Source:** `src/mtg_recognition/settings.py` line 91-92; used in `build_database.py` line 12, `segmentation/cli.py` line 34

```python
# settings.py lines 91-92 — the singleton is created at module level
settings = Settings()
settings.ensure_dirs()

# All consumers import the singleton directly
from mtg_recognition.settings import settings
```

Use `settings.REF_DIR` (line 51-52 of `settings.py`) to derive the state file path in `cli.py` — no hardcoded path strings:

```python
STATE_FILE = settings.REF_DIR / ".mtg-sync-state.json"
```

---

### `raise SystemExit(1)` for non-zero exit

**Source:** `segmentation/cli.py` line 76; `pipeline.py` line 207

Both existing CLIs use `raise SystemExit(1)` (not `sys.exit(1)`) for error exits. Apply identically in `cli.py` for all pre-flight failures.

---

### `if __name__ == "__main__"` + `__main__.py` backward-compat

**Source:** `pipeline.py` lines 324-325; `segmentation/cli.py` lines 114-115; `build_database.py` lines 292-293

The `python -m mtg_recognition.jobs.build_database` invocation works because `build_database.py` has the guard. The refactoring must not move or remove this guard (CLI-06 backward-compat). The new step functions simply become the implementation that `main()` delegates to.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `cli.py` — `@click.group()` | utility | — | No existing Click group in the codebase; both existing CLIs use `@click.command()` only |
| `cli.py` — state file JSON read/write | utility | file-I/O | No precedent for a persistent sync state file; stdlib `json` + `pathlib` pattern is used (same as `pipeline.py` output writing) |
| `cli.py` — Meta.json HTTP fetch | utility | request-response | HTTP fetches in `build_database.py` use `urllib.request` (sync, not async) — same pattern applies |
| `cli.py` — `rich.console.Panel` summary | utility | — | `pipeline.py` uses `from rich import print` only; `Panel` is new but follows `rich` library conventions |

---

## Metadata

**Analog search scope:** `src/mtg_recognition/` (all `.py` files), `pyproject.toml`, `.gitignore`
**Files read:** 7 (`pipeline.py`, `segmentation/cli.py`, `build_database.py`, `phash_matcher.py`, `settings.py`, `pyproject.toml`, `.gitignore`)
**Pattern extraction date:** 2026-05-16
