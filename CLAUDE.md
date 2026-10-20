## Project

MTG card recognition app: takes a photo of a Magic: The Gathering card, segments it, extracts text via OCR, and matches it against a ~1M-card SQLite database. Served as a Streamlit web app via Docker.

The engine (`engine/`) recognises one image at a time in two stages: segmentation (find and warp each card) then matching (pHash shortlists candidates, OCR reranks them against the card catalog). `app/` is the Streamlit UI over the engine; `jobs/` builds the reference data (`mtg update`, `mtg build-index`); `cli.py` registers the `mtg` subcommands defined in their domain modules.

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
- Always include the GitHub issue number. If there is no issue, ask before creating the branch.
- If a change spans several types, pick the main one.

### Attribution

- Never credit Claude or Claude Code as a contributor: no `Co-Authored-By` trailer in commits, no "Generated with Claude Code" line in PR bodies, issues or comments.


## Commands

```bash
uv sync --dev                        # install dependencies (includes DVC)
uv run dvc pull                      # test images; required before running tests
uv run pytest
uv run ruff check --fix src tests && uv run ruff format src tests
uv run lint-imports                  # engine.matching, engine.ocr and engine.segmentation must not import each other
uv run mtg --help                    # recognize, segment, status, update, build-index
```

Test images are DVC-tracked on Google Drive; setup and adding images: `CONTRIBUTING.md`.

`data/reference/` (database, sync state, pHash index) is committed; change it only through `mtg update` and `mtg build-index`.
