## Project

MTG card recognition app: takes a photo of a Magic: The Gathering card, segments it, extracts text via OCR, and matches it against a ~1M-card SQLite database. Served as a Streamlit web app via Docker.

## Structure

```
src/mtg_recognition/
├── engine/        recognises one query image: segmentation, then matching
│   ├── pipeline.py      runs both stages
│   ├── segmentation/    turns each card in a query image into a card image
│   ├── matching/        shortlists candidates by pHash similarity
│   └── ocr/             reads each card image and scores candidates against the reading
├── app/           Streamlit UI over the engine
├── jobs/          builds the reference data from MTGJSON releases (`mtg update`, `mtg build-index`)
├── core/          loads, resizes and lists images for the engine
├── catalog.py     reads printings out of the catalog
├── reference.py   on-disk layout and sync state of the reference data
├── settings.py    configuration from the environment and `.env`
└── cli.py         the `mtg` group; each command is defined in its domain package
data/reference/    the reference data
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

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues on `decksray/MTG-Recognition` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Git

### Workflow

- Branches:
  - `main`: release branch and GitHub default.
  - `dev`: integration branch.
- Feature work:
  - Branch off `dev`, never `main`.
  - Worktrees default to `main`, so base them on `dev` explicitly.
  - For work on a GitHub issue, create the branch with `gh issue develop <issue> --base dev --name <branch> --checkout`. This links the branch to the issue, in its Development section. For a worktree, drop `--checkout`, then `git fetch` and base the worktree on the new branch.
  - For a branch with no issue (see Branch naming), use `git switch -c <branch> dev`.
  - Before committing, run `uv run pre-commit run --all-files`; the hooks may not be installed in this clone.
  - Open PRs against `dev`: always pass `gh pr create --base dev`.
  - Feature PRs are squash-merged into `dev`.
- Releases:
  - Open a PR from `dev` to `main`.
  - Merge it with a merge commit, never squash.
  - Tag the release afterwards (e.g. `v0.2.0`).
- Never commit to `main` directly; fixes go through `dev` and a release.

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
uv run ruff check --fix src tests && uv run ruff format src tests
uv run lint-imports                  # engine.matching, engine.ocr and engine.segmentation must not import each other
uv run mtg --help                    # recognize, segment, status, update, build-index
```

Test images are DVC-tracked on Google Drive; setup and adding images: `CONTRIBUTING.md`.

`data/reference/` (catalog, sync state, pHash index) is committed; change it only through `mtg update` and `mtg build-index`.
