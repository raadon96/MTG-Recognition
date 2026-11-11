---
name: ship
description: Open, merge or release a PR, working out what to do from the current branch and any open PR. `/ship merge` also merges once CI passes, without asking.
disable-model-invocation: true
---

# Ship

## Starting point

Work out what to do from the current branch (`git branch --show-current`, empty on a detached HEAD) and any open PR (`gh pr list --head <branch> --state open`):

| Current branch | State | What `/ship` does |
|---|---|---|
| Feature branch | No PR yet | Opens the PR, waits for CI, asks whether to merge |
| Feature branch | PR open | Waits for CI, asks whether to merge |
| `dev` | No release PR yet | Runs release steps 1-2, asks the user to confirm the version, then continues through CI and asks whether to merge |
| `dev` | Release PR open | Waits for CI, asks whether to merge |
| `main` or detached HEAD | Any | Stops and says why |

On a feature branch, follow Pull requests: from step 1 with no PR yet, from step 2 with a PR open. On `dev`, follow Releases: from step 1 with no release PR yet, from step 5 with a release PR open.

## `/ship merge`

The `merge` argument counts as the user having already said "merge": don't ask again once CI passes, and go on to merge. On a release, this includes tagging and publishing. It never skips the version confirmation. If CI fails, stop and report it, as in step 2 of Pull requests.

## Pull requests

Feature PRs go into `dev` and are squash-merged, with GitHub's default commit message: the PR title followed by `(#<pr>)`.

1. Write the PR body: if `mattpocock-skills:pr` is in the available skills, call the Skill tool with it first and follow it. Then open the PR against `dev`: always pass `gh pr create --base dev`. Title it as a conventional commit, e.g. `fix: read the collector number from the right crop (#42)`. When there is an issue, end the title with `(#<issue>)` and start the body with `Closes #<issue>` and a blank line.
2. Wait for CI: `gh pr checks <pr> --watch`. This can take several minutes; if no checks are reported yet, wait a few seconds and retry. If a check fails, stop and report the check and the relevant log lines; don't merge and don't start a fix. If CI passes and the user hasn't already said "merge", ask whether to merge, and stop unless they say yes.
3. `gh pr merge <pr> --squash`
4. `git switch dev && git pull`
5. Delete the branch: `git push origin --delete <branch>`, then `git branch -D <branch>` (`-D` because git doesn't recognize a squash-merged branch as merged). If the branch is checked out in a worktree, run `git worktree remove <path>` first.
6. If the branch name has an issue number, close the issue: `gh issue close <issue> --comment "Done in #<pr>."` GitHub only closes issues for merges into `main`. If the PR body says the work is partial, leave the issue open and tell the user.

## Releases

To open the release PR:

1. `git switch dev && git pull`. Stop if the working tree has uncommitted changes, or if `git log --oneline main..dev` is empty.
2. Choose the version from `git log --oneline main..dev`: minor if there is a `feat` commit or a breaking change, otherwise patch. Never major; `1.0.0` is the user's decision. Tell the user the version and why, and ask them to confirm, for example: "This will release v0.6.0 because of a `feat` commit. Proceed?" Stop unless they say yes: nothing is changed until then, so the version bump in step 3 runs only after it.
3. `uv version --bump <minor|patch>`, then `git add pyproject.toml uv.lock && git commit -m "chore: release v<version>" && git push`
4. `gh pr create --base main --head dev --title "Release v<version>"`, with the output of `git log --oneline main..dev` as the body.
5. Wait for CI as for feature PRs, then give the user the version and the PR link and ask whether to merge.

When the user says yes to merging the release PR, or ran `/ship merge`:

1. `gh pr merge <pr> --merge`: a merge commit, never squash. Don't delete `dev`.
2. Tag the merge commit: `git switch main && git pull && git tag -a v<version> -m v<version> && git push origin v<version>`. A `v*` tag can't be moved or deleted once pushed.
3. `gh release create v<version> --generate-notes`
4. Bring `dev` up to `main`: `git switch dev && git merge --ff-only main && git push`
