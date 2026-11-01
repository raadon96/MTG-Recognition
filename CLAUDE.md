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
│   ├── layout.py        on-disk layout and sync state
│   ├── catalog.py       the catalog table schema, and reads printings out of the catalog
│   └── sqlite.py        opens the SQLite files read-only
├── settings.py    configuration from the environment and `.env`
└── cli/           the `mtg` group and its commands; the only package that imports Click
    ├── recognize.py     `mtg recognize`
    ├── segment.py       `mtg segment`
    ├── reference.py     `mtg status`, `mtg update`, `mtg build-index`
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

#### Pull requests

Feature PRs go into `dev` and are squash-merged, with GitHub's default commit message: the PR title followed by `(#<pr>)`.

When the user asks to open a PR (without saying "main", which starts a release), start at step 1. When the user says "merge" on an open feature PR, start at step 2.

1. Open the PR against `dev`: always pass `gh pr create --base dev`. Title it as a conventional commit, ending with `(#<issue>)` when there is an issue, e.g. `fix: read the collector number from the right crop (#42)`.
2. Wait for CI: `gh pr checks <pr> --watch`. This can take several minutes; if no checks are reported yet, wait a few seconds and retry. If a check fails, stop and report the check and the relevant log lines; don't merge and don't start a fix. If CI passes and the user hasn't already said "merge", ask whether to merge, and stop unless they say yes.
3. `gh pr merge <pr> --squash`
4. `git switch dev && git pull`
5. Delete the branch: `git push origin --delete <branch>`, then `git branch -D <branch>` (`-D` because git doesn't recognize a squash-merged branch as merged). If the branch is checked out in a worktree, run `git worktree remove <path>` first.
6. If the branch name has an issue number, close the issue: `gh issue close <issue> --comment "Done in #<pr>."` GitHub only closes issues for merges into `main`. If the PR body says the work is partial, leave the issue open and tell the user.

#### Releases

When the user says "open a PR against main" or "release":

1. `git switch dev && git pull`. Stop if the working tree has uncommitted changes, or if `git log --oneline main..dev` is empty.
2. Choose the version from `git log --oneline main..dev`: minor if there is a `feat` commit or a breaking change, otherwise patch. Never major; `1.0.0` is the user's decision. Tell the user the version and why.
3. `uv version --bump <minor|patch>`, then `git add pyproject.toml uv.lock && git commit -m "chore: release v<version>" && git push`
4. `gh pr create --base main --head dev --title "Release v<version>"`, with the output of `git log --oneline main..dev` as the body.
5. Wait for CI as for feature PRs, then give the user the version and the PR link and ask whether to merge.

When the user says "merge" on the release PR:

1. `gh pr merge <pr> --merge`: a merge commit, never squash. Don't delete `dev`.
2. Tag the merge commit: `git switch main && git pull && git tag -a v<version> -m v<version> && git push origin v<version>`. A `v*` tag can't be moved or deleted once pushed.
3. `gh release create v<version> --generate-notes`
4. Bring `dev` up to `main`: `git switch dev && git merge --ff-only main && git push`

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
uv run mtg --help                    # recognize, segment, status, update, build-index, benchmark
uv run mtg benchmark run             # recognise the labelled images; non-zero on any change from the baseline
uv run mtg benchmark update-baseline # copy output/benchmark/latest.json onto data/benchmark-baseline.json
uv run mtg benchmark status          # summarise the baseline (--latest: the latest results; --failures: list failing images)
```

Test images are DVC-tracked on Google Drive; setup and adding images: `CONTRIBUTING.md`.

`data/reference/` (catalog, sync state, pHash index) is committed; change it only through `mtg update` and `mtg build-index`. `data/benchmark-baseline.json` is committed; change it only through `mtg benchmark update-baseline`.
