# PROTOTYPE: CONTRIBUTING outline (throwaway, for "Draft the CONTRIBUTING outline", #258)

Rough outline of the contributors-only CONTRIBUTING.md. Order is the proposal; wording is a first take to react to.
`OPEN:` marks a question to decide.

---

# Contributing to MTG Recognition

One line: how to set up the code, run the tests and the benchmark, and send changes. Users who only want to run the app: README.

## Quick Start

Works without a dataset key: the fast tests and the hooks need no DVC data.

Needs: git, [uv](https://docs.astral.sh/uv/getting-started/installation/), Docker (only to build the image).

```bash
git clone https://github.com/raadon96/MTG-Recognition.git
cd MTG-Recognition
uv sync --dev                     # Python and the locked dependencies
uv run pre-commit install         # the checks run on every commit
uv run pytest -m "not slow"       # the fast tests
uv run streamlit run src/mtg_recognition/app/app.py   # the app, at http://localhost:8501
docker compose up --build         # or the app from the image, as users run it
```

One line on what the hooks run: ruff, `lint-imports`, `ty check`, the fast tests; `uv run pre-commit run --all-files` runs them by hand.
One line: the catalog is committed, so recognition works right after cloning; no `mtg reference` command is needed.

## Branches and pull requests

OPEN: new section; the rules now live only in CLAUDE.md.

- Branch off `dev`; PRs target `dev`. `main` takes releases only.
- Branch name `<type>/<issue>-<slug>` (types: feat, fix, refactor, docs, test, chore).
- A PR that changes recognition results commits the new baseline (see The benchmark).

## Commands

One line: `uv run mtg <command> --help` is the full description; the table only says what each is for.

| Command | What it is for |
|---|---|
| `mtg recognize` | Recognise one query image or a directory |
| `mtg segment` | Write the card images found in a query image |
| `mtg reference status` | Compare the catalog's MTGJSON release with the latest |
| `mtg reference update` | Build the reference data from the latest MTGJSON release (maintainer) |
| `mtg reference build` | Rebuild the catalog's own release after changing the build code (maintainer) |
| `mtg reference download-images` | Download the reference images, to look at them while debugging matching |
| `mtg benchmark run` | Recognise the labelled images and compare with the baseline |
| `mtg benchmark update-baseline` | Adopt the latest results as the baseline |
| `mtg benchmark status` | Summarise the baseline or the latest results |

## The dataset

Needed for the slow tests and the benchmark.

1. Ask the maintainer for a read-only key (a JSON file); keep it outside the repo.
2. Point DVC at it (the two `dvc remote modify --local` lines); one sentence that `--local` keeps them out of the committed config.
3. `uv run dvc pull` (or `uv run dvc pull data/benchmark.dvc` for the benchmark only).

One line on what lands where: `data/benchmark/` (the labelled query images), `data/scratch/` (query images that fail, kept for debugging).
Link to ADR 0001 for why Google Drive and DVC.

To add query images: open an issue and share them with the maintainer, who adds them (link to the guide).

## The benchmark

```bash
uv run pytest -m slow
uv run mtg benchmark run
```

The loop, in four lines: `run` fails on any change in exact printing or card count, better or worse; look at the changed images (`mtg benchmark status --latest --failures`); if the change is wanted, `mtg benchmark update-baseline`; commit `data/benchmark-baseline.json` in the same PR.

## The benchmark in CI

Short form:

- The `engine` workflow runs the slow tests and the benchmark on every PR, except PRs that change only Markdown, `docs/` or `.claude/`; those report no `engine` check.
- It uploads the results as the `benchmark-latest` artifact.
- A runner and a laptop give the same results (#146). Record baselines locally. If CI fails on a baseline that passes locally, that is a bug: open an issue and commit neither result.

OPEN: the cache paragraph (which cache, why pushes to `dev`/`main` start a run, the Monday/Thursday schedule) moves to comments in `.github/workflows/engine.yml`, next to the triggers it explains.

## Docker image

OPEN: one short section, once the release workflow exists: PRs that touch the image build and smoke-test it; `v*` tags publish `ghcr.io/raadon96/mtg-recognition`. Or leave it out until then.

## Troubleshooting

Contributor errors only, as a table (error → cause and fix), like the guide's:

- `pre-commit install` refuses because `core.hooksPath` is set.
- `dvc pull`: "File not found: <folder-id>".
- `module 'OpenSSL.crypto' has no attribute 'sign'`.
- `dvc pull`: "no data to pull".
- `dvc pull`: "file not found in remote".
- `dvc push`: "Insufficient permissions" (the key is read-only; send the images to the maintainer).

---

## Moves to `docs/guides/dataset-maintenance.md`

- Maintainer setup: the OAuth client secret, the consent screen steps, the token cache path.
- Adding query images: steps 1 to 3 (drop under `data/benchmark/` or `data/scratch/`, no byte-identical duplicates, label in `manifest.json`, `dvc add`, `dvc push`, commit the `.dvc` pointers). The guide's "Push new images" now holds these instead of pointing back to CONTRIBUTING.
- Troubleshooting `invalid_grant` (already there; drop from CONTRIBUTING).

## Moves into `--help`

What CONTRIBUTING and the README say that `--help` lacks:

- `mtg segment`: no description at all; add one ("Write each card image found in a query image to the output directory.").
- `mtg benchmark run`: checks the manifest against the images and the catalog; prints the images that changed against the baseline; to accept a change, run `update-baseline` and commit the baseline in the same PR.
- `mtg benchmark status`: what it prints (commit, date, MTGJSON version, totals, runtime; whether the latest results match the baseline). `--failures`: the labels' and recognised printings, the recognition error; labels show as `unknown` until the dataset is pulled.
- `mtg reference status`: the result is up to date, outdated (with the command to run) or unknown when MTGJSON can't be reached.
- `mtg reference update`: AllPrintings kept in `data/reference/mtgjson/`; downloads about 116k images, 12 GB, on a first run; failures go to `data/reference/image_errors.log`.
- `mtg reference build`: downloads AllPrintings only while that release is still MTGJSON's latest, fails otherwise and names `mtg reference update`; `--no-cache` example: Scryfall replaced an image under the same URL.
- `mtg reference download-images`: not needed to recognise cards (matching uses the catalog's pHashes); about 116k images, 12 GB; failures go to `data/reference/download_errors.log`, removed when a run has none; running again downloads only what is missing.

## Cut as repetition

- The five-line `mtg benchmark` command block and its long paragraph (the table and `--help`).
- The second `uv sync --dev` in the dataset setup.
- "Reviewers will `git pull && uv run dvc pull`" (step 3).
- The history of #146 beyond one link.
- The README's development setup (`source .venv/bin/activate`, the `mtg reference` paragraphs): replaced by the Quick Start and `--help`.
- "Decksray" in the title.

## CLAUDE.md

- "setup and adding images: `CONTRIBUTING.md`" becomes: setup in CONTRIBUTING.md, adding images and the remote in `docs/guides/dataset-maintenance.md`.
