"""`mtg app`: start the Streamlit web app.

Usage examples:
    uv run mtg app

    uv run mtg app --server.port 8600
"""

import os
from importlib.resources import files

import click


@click.command(name="app", context_settings={"ignore_unknown_options": True})
@click.argument("streamlit_args", nargs=-1, type=click.UNPROCESSED)
def app(streamlit_args: tuple[str, ...]) -> None:
    """Start the web app; extra options go to `streamlit run`.

    The app is served on http://localhost:8501; set another port with
    --server.port.

    Examples:

    \b
    uv run mtg app
    uv run mtg app --server.port 8600
    """
    # the script's path, not an import: importing the app runs the Streamlit page
    script = str(files("mtg_recognition.app") / "app.py")
    # replace this process so Ctrl+C and Docker's stop signal reach Streamlit
    # directly, and Streamlit's exit code is the command's
    os.execvp("streamlit", ["streamlit", "run", script, *streamlit_args])
