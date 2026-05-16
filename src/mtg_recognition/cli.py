"""Top-level `mtg` CLI group.

Wraps the existing recognition and segmentation pipelines as Click subcommands
so collaborators can use `mtg recognize` and `mtg segment` instead of the long
`uv run python -m mtg_recognition.engine.*` invocations.

Also provides `mtg build-index` (standalone pHash rebuild) and `mtg sync`
(full reference data stack: MobileCache → images → pHash index, with
intelligent version-based skipping).
"""

import json
import sqlite3
from multiprocessing import cpu_count
from urllib import request as urllib_request

import click
from rich.console import Console
from rich.panel import Panel

from mtg_recognition.engine.matching.phash_matcher import PHashCardMatcher
from mtg_recognition.engine.pipeline import main as _recognize_main
from mtg_recognition.engine.segmentation.cli import main as _segment_main
from mtg_recognition.jobs import build_database
from mtg_recognition.settings import settings

# Module-level Console instance — used by build-index and sync step headers
# and summary panel.
console = Console()

# State file path for sync version tracking (D-01)
STATE_FILE = settings.REF_DIR / ".mtg-sync-state.json"

# MTGJSON Meta endpoint for lightweight version check (D-04)
META_URL = "https://mtgjson.com/api/v5/Meta.json"


def _load_state() -> dict:
    """Return the sync state dict from disk.

    Returns all-null on FileNotFoundError or JSON decode error (D-03:
    missing or corrupt state treated the same as first-run).
    """
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "mobilecache_version": None,
            "images_version": None,
            "phash_version": None,
        }


def _save_state(state: dict) -> None:
    """Write state dict to STATE_FILE as pretty-printed JSON."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _fetch_mtgjson_version() -> str:
    """Fetch the current MTGJSON version via a lightweight META_URL GET.

    Uses the same SCRYFALL_HEADERS as build_database.py (D-04).
    Re-raises any exception so the sync orchestrator can exit non-zero (SYNC-04).
    """
    req = urllib_request.Request(META_URL, headers=build_database.SCRYFALL_HEADERS)
    with urllib_request.urlopen(req) as response:
        data = json.loads(response.read())
    return data["meta"]["version"]


@click.group()
def cli() -> None:
    """MTG card recognition toolkit."""
    pass


# Register existing commands directly — zero code duplication (D-11).
# The name= argument gives the user `mtg recognize` and `mtg segment` regardless
# of what the underlying function is named inside its own module.
cli.add_command(_recognize_main, name="recognize")
cli.add_command(_segment_main, name="segment")


@cli.command("build-index")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    show_default=True,
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
    """Rebuild the pHash index from reference images already on disk.

    Requires data/reference/images/ to be populated first (run 'mtg sync'
    to download reference images if you haven't already).
    """
    # Preflight check (D-10): image directory must exist and be non-empty.
    img_dir = settings.IMG_DIR
    if not img_dir.exists() or not any(img_dir.iterdir()):
        click.echo(
            "Error: data/reference/images/ is missing or empty. Run 'mtg sync' first.",
            err=True,
        )
        raise SystemExit(1)

    # Instantiate with auto_build=False so the preflight branch is the only path
    # that initiates a build — avoids a double-build when force=False and files exist.
    matcher = PHashCardMatcher(auto_build=False)
    matcher.build_index(force=force, num_workers=workers)


@cli.command("sync")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    show_default=True,
    help="Bypass all version checks and rebuild MobileCache, images, and pHash index unconditionally.",
)
@click.option(
    "--skip-db-check",
    is_flag=True,
    default=False,
    show_default=True,
    help="Skip the MTGJSON Meta.json HTTP freshness check (Step 1). Use when offline.",
)
def sync(force: bool, skip_db_check: bool) -> None:
    """Rebuild the full reference data stack in one shot.

    Runs three steps in order: MobileCache rebuild → reference image download →
    pHash index build. Skips steps whose local version already matches the remote
    MTGJSON version. Use --force to bypass all version checks.
    """
    # First-run directory creation (Claude's Discretion from CONTEXT.md)
    settings.REF_DIR.mkdir(parents=True, exist_ok=True)

    state = _load_state()
    summary: dict[str, str] = {}

    try:
        # ------------------------------------------------------------------
        # Step 1 — MobileCache (D-04, D-13)
        # ------------------------------------------------------------------
        console.print("\n[bold]--- Step 1: MobileCache ---[/bold]")

        # Determine whether to rebuild and get remote_version when available
        remote_version: str | None = None
        rebuild_mobilecache = force

        if not force:
            if skip_db_check:
                # No network fetch — check local state only (D-08)
                if (
                    state["mobilecache_version"] is not None
                    and settings.LOCAL_DB_FILE.exists()
                ):
                    rebuild_mobilecache = False
                else:
                    rebuild_mobilecache = True
            else:
                remote_version = _fetch_mtgjson_version()
                if (
                    state["mobilecache_version"] == remote_version
                    and settings.LOCAL_DB_FILE.exists()
                ):
                    rebuild_mobilecache = False
                else:
                    rebuild_mobilecache = True

        if not rebuild_mobilecache:
            console.print(
                f"[green]✓ MobileCache is up-to-date ({state['mobilecache_version']}) — skipping rebuild[/green]"
            )
            summary["MobileCache"] = f"skipped ({state['mobilecache_version']})"
            dest_engine = None
        else:
            # Write in-progress null BEFORE the work begins (D-02)
            state["mobilecache_version"] = None
            _save_state(state)

            dest_engine = build_database.build_mobilecache()

            # Resolve the version to persist
            if remote_version is not None:
                resolved = remote_version
            else:
                # skip_db_check=True — read version from the extracted AllPrintings.sqlite
                try:
                    conn = sqlite3.connect(str(settings.MTGO_DB_FILE))
                    row = conn.execute("SELECT version FROM meta").fetchone()
                    conn.close()
                    resolved = row[0] if row else "unknown"
                except Exception:
                    resolved = "unknown"

            state["mobilecache_version"] = resolved
            _save_state(state)
            summary["MobileCache"] = f"rebuilt ({resolved})"

        # ------------------------------------------------------------------
        # Step 2 — Reference images (D-05, D-13)
        # ------------------------------------------------------------------
        console.print("\n[bold]--- Step 2: Reference Images ---[/bold]")

        rebuild_images = force or not (
            state["images_version"] is not None
            and state["images_version"] == state["mobilecache_version"]
        )

        if not rebuild_images:
            console.print(
                f"[green]✓ Reference images are up-to-date ({state['images_version']}) — skipping download[/green]"
            )
            summary["Reference images"] = f"skipped ({state['images_version']})"
        else:
            # Write in-progress null BEFORE the work begins (D-02)
            state["images_version"] = None
            _save_state(state)

            build_database.download_images(dest_engine=dest_engine)

            state["images_version"] = state["mobilecache_version"]
            _save_state(state)
            summary["Reference images"] = f"downloaded ({state['images_version']})"

        # ------------------------------------------------------------------
        # Step 3 — pHash index (D-06, D-13, D-15)
        # ------------------------------------------------------------------
        console.print("\n[bold]--- Step 3: pHash Index ---[/bold]")

        rebuild_phash = force or not (
            state["phash_version"] is not None
            and state["phash_version"] == state["images_version"]
        )

        if not rebuild_phash:
            console.print(
                f"[green]✓ pHash index is up-to-date ({state['phash_version']}) — skipping rebuild[/green]"
            )
            summary["pHash index"] = f"skipped ({state['phash_version']})"
        else:
            # Write in-progress null BEFORE the work begins (D-02)
            state["phash_version"] = None
            _save_state(state)

            # D-15: tqdm progress bar inside build_index() passes through unchanged
            matcher = PHashCardMatcher(auto_build=False)
            matcher.build_index(force=True)

            state["phash_version"] = state["images_version"]
            _save_state(state)
            summary["pHash index"] = f"rebuilt ({state['phash_version']})"

    except Exception as exc:
        # SYNC-04: non-zero exit with descriptive error message
        # Use click.echo for stderr output (rich console does not accept err=True)
        click.echo(
            f"Error during sync: {type(exc).__name__}: {exc}",
            err=True,
        )
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Summary panel (D-14)
    # ------------------------------------------------------------------
    summary_text = "\n".join(
        f"{artifact}: {outcome}" for artifact, outcome in summary.items()
    )
    console.print(Panel(summary_text, title="Sync Summary", expand=False))


if __name__ == "__main__":
    cli()
