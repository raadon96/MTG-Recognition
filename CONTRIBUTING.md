# Contributing to Decksray MTG Recognition

This guide covers everything a new collaborator or contributor needs to set up the development environment, pull the test/benchmark dataset, and share new test images.

## Development Setup

Install the dependencies and the git hooks:

```bash
uv sync --dev
uv run pre-commit install
```

Every commit then runs `ruff check --fix`, `ruff format`, `lint-imports`, `ty check` and the fast tests (`pytest -m "not slow"`), and commits to `main` are refused. Run the hooks by hand with `uv run pre-commit run --all-files`.

If `pre-commit install` refuses because `core.hooksPath` is set, check where it points (`git config --show-origin core.hooksPath`). If it is the default `.git/hooks`, run `git config --unset-all core.hooksPath` first.

## Dataset Setup

The test and benchmark dataset is versioned with [DVC](https://dvc.org/) and hosted on a shared Google Drive folder. To pull the dataset, you need a one-time Google Cloud Platform (GCP) OAuth setup, then a single `dvc pull` command.

### One-time setup (per developer)

DVC's default Google OAuth app is blocked by Google. Each contributor must create a personal GCP OAuth Desktop app:

1. Go to <https://console.cloud.google.com/> and create a new project (or pick an existing one).
2. Enable the **Google Drive API** under APIs & Services -> Library.
3. Configure the **OAuth consent screen**: User Type = External; add your own email as a Test user.
4. Create credentials: APIs & Services -> Credentials -> Create Credentials -> OAuth client ID -> Application type = **Desktop app**. Note the `client_id` and `client_secret`.
5. Ask a project maintainer to share the Google Drive folder `decksray/MTG-Recognition/data/test` with your Google account (read access for pull, write access for push).

Then configure your local DVC with the secret (note the `--local` flag — this writes to `.dvc/config.local`, which is gitignored, NOT to the committed `.dvc/config`):

```bash
uv sync --dev
uv run dvc remote modify --local origin gdrive_client_secret '<your-client-secret>'
```

The committed `.dvc/config` already contains the public `gdrive_client_id`, the remote URL (`gdrive://root/decksray/MTG-Recognition/data/test`), and the remote name (`origin`).

### Pull the dataset

```bash
uv run dvc pull
```

On first run, your browser will open Google's OAuth consent screen. Sign in with the Google account that was added as a Test user in step 3. After grant, DVC caches your token at `~/.cache/pydrive2fs/<client_id>/default.json` and subsequent pulls run non-interactively.

The pull will populate `data/test/` with all 7 image categories plus the migrated `data/test/6_benchmark/`.

## Adding New Test Images

Anyone with write access to the Google Drive folder can contribute new test images. The workflow:

### 1. Drop the new images under `data/test/`

Place new images under the appropriate category subdirectory (e.g. `data/test/1_photos_single_card/`). Filenames are typically the card UUID — see existing files for examples.

### 2. Re-track and push

```bash
uv run dvc add data/test
uv run dvc push
git add data/test.dvc
git commit -m "data: add new test images for <description>"
```

`dvc add` re-hashes `data/test/`, updating `data/test.dvc` with the new content hash. `dvc push` uploads only the new blobs (existing blobs are deduplicated by MD5). The `git commit` of the updated `data/test.dvc` is what tells future `dvc pull`s to fetch the new images.

### 3. Open a PR

Push your branch and open a pull request. Reviewers will `git pull && uv run dvc pull` to fetch your additions before reviewing.

## Troubleshooting

### "This app is blocked"
Your `gdrive_client_id`/`gdrive_client_secret` are missing or stale. Re-run the One-time setup above.

### `dvc pull` says "no data to pull"
The `data/test.dvc` file is missing from your branch. Run `git pull` to fetch the latest pointer file.

### `dvc pull` says "file not found in remote"
The contributor who added the data forgot to `dvc push` (or pushed to the wrong remote). Ask them to re-push.
