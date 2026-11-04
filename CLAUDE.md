## Project

MTG card recognition app: takes a photo of a Magic: The Gathering card, segments it, extracts text via OCR, and matches it against the catalog, a SQLite file of every printing. Served as a Streamlit web app via Docker, and run from the terminal through the `mtg` CLI.

## Structure

```
src/mtg_recognition/
├── engine/        recognises one query image: segmentation, then matching
│   ├── pipeline.py      runs both stages
│   ├── segmentation/    turns each card in a query image into a card image
│   ├── matching/        shortlists candidates by pHash similarity
│   ├── ocr/             reads each card image and scores candidates against the reading
│   ├── image.py         loads and resizes images for the engine
│   └── ranking.py       ranked candidates and their field scores
├── app/           Streamlit UI over the engine
├── jobs/          builds the reference data from MTGJSON releases, as plain functions returning results
├── benchmark/     validates the manifest, scores each query image against its label, compares results with the baseline
├── reference/     the reference data, read by the engine and written by jobs
│   ├── layout.py        on-disk layout
│   ├── catalog.py       the catalog's schema, and reads printings, pHashes and the MTGJSON version out of it
│   └── sqlite.py        opens the SQLite files read-only
├── settings.py    configuration from the environment and `.env`
└── cli/           the `mtg` group and its commands; the only package that imports Click
    ├── recognize.py     `mtg recognize`
    ├── segment.py       `mtg segment`
    ├── reference.py     `mtg reference status`, `mtg reference update`, `mtg reference build`, `mtg reference download-images`
    └── benchmark.py     `mtg benchmark run`, `mtg benchmark update-baseline`, `mtg benchmark status`
data/reference/    the reference data
models/            model weights: {segmentation,ocr}/prod is tracked, {segmentation,ocr}/test is gitignored
tests/             unit and regression tests
```

## Code style

Before writing, refactoring or reviewing Python, call the Skill tool with `direct-python`. See `CODING_STANDARDS.md`.

## Plain language

In code, comments, issues, PR bodies and commit messages, write the literal word. These metaphors recur in this repo; use the literal word beside each:

- pin → set, fix to a value
- land → merge, ship
- settle → decide
- guard → check
- carry → hold, keep
- sharp → precise, exact
- load-bearing → required, depended on

Name domain things with the terms in `GLOSSARY.md`; when a term exists, use it rather than a new name.

- Before: "Pin the candidate count so the top match lands in the shortlist; the OCR crop is load-bearing."
- After: "Set the candidate count so the best printing is among the candidates; scoring needs the collector information crop."

When passing on review findings, state each in plain words with a short before/after example, rather than citing the rule it breaks.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues on `raadon96/MTG-Recognition` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Git

Only the user opens, merges or releases PRs, by running `/ship`. Agents commit and push, then stop.

### Workflow

#### Branches

- `main`: release branch and GitHub default.
- `dev`: integration branch.
- Never commit to `main` directly; fixes go through `dev` and a release.

#### Feature work

- Branch off `dev`, never `main`.
- Worktrees default to `main`, so base them on `dev` explicitly.
- For work on a GitHub issue, create the branch with `gh issue develop <issue> --base dev --name <branch> --checkout`. This links the branch to the issue, in its Development section. For a worktree, drop `--checkout`, then `git fetch` and base the worktree on the new branch.
- For a branch with no issue (see Branch naming), use `git switch -c <branch> dev`.
- The pre-commit hook runs the checks on `git commit`; if a commit prints no hook output, run `uv run pre-commit install`.

### Branch naming

Format: `<type>/<issue-number>-<short-slug>`

Types (use only these):
- `feat` – new feature or capability
- `fix` – bug fix
- `refactor` – code restructuring without behavior change
- `docs` – documentation only
- `test` – adding or fixing tests
- `chore` – maintenance (config, dependencies, `.gitignore`, cleanup)

Rules:
- Always include the GitHub issue number. If there is no issue, ask before creating the branch; once approved, name it `<type>/<short-slug>`.
- If a change spans several types, pick the main one.

### Attribution

- Never credit Claude or Claude Code as a contributor: no `Co-Authored-By` trailer in commits, no "Generated with Claude Code" line in PR bodies, issues or comments.


## Commands

```bash
uv sync --dev                        # install dependencies (includes DVC)
uv run pre-commit run --all-files    # run the commit hooks by hand
uv run dvc pull                      # test images; required before running tests
uv run pytest
uv run lint-imports                  # contracts: the layers, the engine stages independent, only the CLI imports Click
uv run mtg --help                    # recognize, segment, reference, benchmark
uv run mtg reference status          # the catalog's MTGJSON release against the latest one
uv run mtg reference update          # build the reference data from the latest MTGJSON release; nothing when the catalog already is it
uv run mtg reference build           # rebuild the catalog's own release, reusing AllPrintings and unchanged pHashes (--no-cache: reuse nothing)
uv run mtg reference download-images # download the reference images missing for the committed catalog; changes nothing else
uv run mtg benchmark status          # summarise the baseline (--latest: the latest results; --failures: list failing images)
uv run mtg benchmark run             # recognise the labelled images; non-zero on any change from the baseline
uv run mtg benchmark update-baseline # copy output/benchmark/latest.json onto data/benchmark-baseline.json
```

Test images are DVC-tracked on Google Drive; setup and adding images: `CONTRIBUTING.md`.

`data/reference/catalog.sqlite` (each printing's details and pHash, and the MTGJSON version) is committed; change it only through `mtg reference update` or `mtg reference build`. `data/benchmark-baseline.json` is committed; change it only through `mtg benchmark update-baseline`.
