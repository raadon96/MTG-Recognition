# Contributing to MTG Recognition

How to set up the code, run the tests and the benchmark, and send changes. To run the app without changing it, see the [README](README.md).

## Quick Start

You need git, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Docker only to build the image. No dataset key is needed.

```bash
git clone https://github.com/raadon96/MTG-Recognition.git
cd MTG-Recognition
uv sync --dev                  # Python and the locked dependencies
uv run pre-commit install      # run the checks on every commit
uv run pytest -m "not slow"    # the fast tests
uv run mtg app                 # the web app, at http://localhost:8501
docker compose up --build      # or the app from the Docker image, as users run it
```

Every commit runs `ruff check --fix`, `ruff format`, `lint-imports`, `ty check` and the fast tests, and refuses commits to `main`; `uv run pre-commit run --all-files` runs them by hand.

The catalog, `data/reference/catalog.sqlite`, is committed, so recognition works right after cloning.

## Branches and pull requests

- Branch off `dev`, and open pull requests against `dev`. `main` takes releases only.
- Name the branch `<type>/<issue>-<slug>`, with the type one of `feat`, `fix`, `refactor`, `docs`, `test` or `chore`; for example `fix/123-double-faced-cards`.
- A pull request that changes recognition results commits the new baseline (see [The benchmark](#the-benchmark)).

## Commands

Each command describes itself in full with `uv run mtg <command> --help`.

| Command | What it is for |
|---|---|
| `mtg app` | Start the web app |
| `mtg recognize` | Rank the printings of every card in the query images |
| `mtg segment` | Write each card in a query image as a card image |
| `mtg reference status` | Compare the catalog with the latest MTGJSON release |
| `mtg reference update` | Build the reference data from the latest MTGJSON release |
| `mtg reference build` | Rebuild the reference data from the catalog's release |
| `mtg reference download-images` | Download the reference images missing from disk, for debugging matching |
| `mtg benchmark run` | Recognise the benchmark and compare with the baseline |
| `mtg benchmark update-baseline` | Copy the latest results onto the baseline |
| `mtg benchmark status` | Summarise the baseline or the latest results |

## The dataset

The slow tests and the benchmark need the dataset, which DVC keeps on Google Drive ([ADR 0001](docs/adr/0001-benchmark-dataset-on-google-drive-via-dvc.md)).

Ask the maintainer for a read-only key, a JSON file. Keep it outside the repo, for example at `~/.config/mtg-recognition/dataset-reader.json`, and point DVC at it:

```bash
uv run dvc remote modify --local origin gdrive_use_service_account true
uv run dvc remote modify --local origin gdrive_service_account_json_file_path ~/.config/mtg-recognition/dataset-reader.json
uv run dvc pull                # or uv run dvc pull data/benchmark.dvc for the benchmark only
```

`--local` writes the two settings to the gitignored `.dvc/config.local`; never put them in the committed `.dvc/config`.

The pull fills `data/benchmark/` with the labelled query images, and `data/scratch/` with query images that fail recognition, kept for debugging.

To add query images, open an issue and share them with the maintainer, who adds them as described in [Maintaining the benchmark dataset](docs/guides/dataset-maintenance.md#push-new-images). The key is read-only, so you can't push them yourself.

## The benchmark

```bash
uv run pytest -m slow
uv run mtg benchmark run
```

1. `mtg benchmark run` fails on any change in exact printing or card count, better or worse.
2. Look at the failing images with `uv run mtg benchmark status --latest --failures`.
3. If the change is wanted, run `uv run mtg benchmark update-baseline`.
4. Commit `data/benchmark-baseline.json` in the same pull request.

## The benchmark in CI

- The `engine` workflow runs the slow tests and the benchmark on every pull request, except those that change only Markdown files, `docs/` or `.claude/`. Those report no `engine` check at all, so don't make it a required check.
- It uploads the results as the `benchmark-latest` artifact.
- A GitHub runner and a laptop give the same results ([#146](https://github.com/raadon96/MTG-Recognition/issues/146)), so record baselines locally. If CI fails on a baseline that passes locally, results depend on the machine again: open an issue, and commit neither result as the baseline until it is fixed.

## Docker image

The `docker` workflow builds the image on each pull request that changes what it is built from, and checks that it serves the web app and recognises a card. Each `v*` tag publishes the image to `ghcr.io/raadon96/mtg-recognition`, for `linux/amd64` and `linux/arm64`.

## Troubleshooting

| Error | Cause and fix |
|---|---|
| `pre-commit install` refuses because `core.hooksPath` is set | Check where it points with `git config --show-origin core.hooksPath`. If it is the default `.git/hooks`, run `git config --unset-all core.hooksPath`, then install again |
| `dvc pull` says "File not found: <folder-id>" | The key can't see the dataset folder: it is for another account, or the folder isn't shared with it. Ask the maintainer |
| `module 'OpenSSL.crypto' has no attribute 'sign'` | pyOpenSSL 24.3 or later is installed, which the Google Drive sign-in can't use. Run `uv sync --dev` to get the locked versions |
| `dvc pull` says "no data to pull" | `data/benchmark.dvc` and `data/scratch.dvc` are missing from your branch. Run `git pull` |
| `dvc pull` says "file not found in remote" | The maintainer hasn't pushed the data those files point to yet. Ask them to |
| `dvc push` says "Insufficient permissions for the specified parent" | The key is read-only. Share the images with the maintainer instead |
