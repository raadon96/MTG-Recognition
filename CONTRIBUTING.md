# Contributing to Decksray MTG Recognition

This guide covers everything a new collaborator or contributor needs to set up the development environment, pull the benchmark and scratch datasets, and share new benchmark images.

## Development Setup

Install the dependencies and the git hooks:

```bash
uv sync --dev
uv run pre-commit install
```

Every commit then runs `ruff check --fix`, `ruff format`, `lint-imports`, `ty check` and the fast tests (`pytest -m "not slow"`), and commits to `main` are refused. Run the hooks by hand with `uv run pre-commit run --all-files`.

If `pre-commit install` refuses because `core.hooksPath` is set, check where it points (`git config --show-origin core.hooksPath`). If it is the default `.git/hooks`, run `git config --unset-all core.hooksPath` first.

## Dataset Setup

The test and benchmark dataset is versioned with [DVC](https://dvc.org/) and hosted in a Google Drive folder owned by a Google account that holds nothing but the dataset ([ADR 0001](docs/adr/0001-benchmark-dataset-on-google-drive-via-dvc.md)). Contributors and CI pull with a read-only key; only the maintainer pushes.

### One-time setup (per developer)

Ask the maintainer for a read-only key: a JSON key file for the `dataset-reader` service account, which can view the dataset folder and nothing else. Keep it outside the repo, e.g. at `~/.config/mtg-recognition/dataset-reader.json`, then point your local DVC at it:

```bash
uv sync --dev
uv run dvc remote modify --local origin gdrive_use_service_account true
uv run dvc remote modify --local origin gdrive_service_account_json_file_path ~/.config/mtg-recognition/dataset-reader.json
```

The `--local` flag writes to `.dvc/config.local`, which is gitignored. Never put these two settings in the committed `.dvc/config`: the maintainer pushes through OAuth, not the service account. The committed `.dvc/config` already holds the remote name (`origin`), its URL (`gdrive://<folder-id>`) and the public `gdrive_client_id`.

### Pull the dataset

```bash
uv run dvc pull
```

The pull fills `data/benchmark/` with the 7 image categories, and `data/scratch/` with query images that fail recognition, kept for debugging (`error/`, `failure_cases/`, `uncommon_layout/`). To fetch only the benchmark, run `uv run dvc pull data/benchmark.dvc`. The key is read-only: `dvc push` with it fails with "Insufficient permissions for the specified parent".

### Run the benchmark

```bash
uv run mtg benchmark run               # recognise every labelled image, compare with the baseline
uv run mtg benchmark update-baseline   # adopt the latest results as the baseline
```

`run` checks `data/benchmark/manifest.json` against the images and the catalog, recognises each image, writes the results to `output/benchmark/latest.json` and prints the images that flipped against the baseline, `data/benchmark-baseline.json`. It exits non-zero on any change in exact printing or card count, better or worse. To accept a change, run `update-baseline` and commit the baseline in the same PR.

## Adding New Benchmark Images

Only the maintainer can push to the dataset folder. To contribute images, open an issue or PR describing them and share them with the maintainer, who adds them as follows.

### Maintainer setup

The maintainer pushes through the project's OAuth client, signed in as the dataset account. Put the client secret in the local config:

```bash
uv run dvc remote modify --local origin gdrive_client_secret '<client-secret>'
```

The first push opens Google's consent screen in the browser: pick the dataset account (not a personal one), click Advanced → Go to mtg-recognition (the app is unverified on purpose, see ADR 0001), and tick both Drive permissions. DVC caches the token at `~/.cache/pydrive2fs/<client_id>/default.json`, so later pushes run without a browser.

### 1. Drop the new images under `data/benchmark/` or `data/scratch/`

Place new images in the appropriate category folder (e.g. `data/benchmark/1_photos_single_card/`). Query images that fail recognition and aren't part of the benchmark go in a `data/scratch/` folder instead. Don't add an image that is byte-identical to one already in either dataset. Filenames are typically the card UUID — see existing files for examples. Label each new image in `data/benchmark/manifest.json` with the UUIDs of the cards in it (`[]` for none): `mtg benchmark run` refuses to run on an unlabelled image.

### 2. Re-track and push

```bash
uv run dvc add data/benchmark data/scratch
uv run dvc push
git add data/benchmark.dvc data/scratch.dvc
git commit -m "data: add benchmark images for <description>"
```

`dvc add` re-hashes each folder, updating its pointer (`data/benchmark.dvc`, `data/scratch.dvc`) with the new content hash. `dvc push` uploads only the new blobs (existing blobs are deduplicated by MD5). The `git commit` of the updated pointers is what tells future `dvc pull`s to fetch the new images.

### 3. Open a PR

Push your branch and open a pull request. Reviewers will `git pull && uv run dvc pull` to fetch your additions before reviewing.

## Troubleshooting

### `dvc pull` says "File not found: <folder-id>"
The service account can't see the dataset folder: the key is for another account, or the folder isn't shared with `dataset-reader` as Viewer. The share must be on the folder the remote URL points at, not on the `files` folder inside it. Ask the maintainer.

### `module 'OpenSSL.crypto' has no attribute 'sign'`
Your environment has pyOpenSSL 24.3 or later, which PyDrive2's service-account sign-in can't use. PyDrive2 1.21.3 requires an older one; run `uv sync --dev` to get the locked versions.

### `invalid_grant: Token has been expired or revoked` (maintainer)
The cached OAuth token is no longer valid. Delete `~/.cache/pydrive2fs/<client_id>/default.json` and push again, signing in as the dataset account.

### `dvc pull` says "no data to pull"
The `data/benchmark.dvc` and `data/scratch.dvc` files are missing from your branch. Run `git pull` to fetch the latest pointer files.

### `dvc pull` says "file not found in remote"
The maintainer hasn't pushed the data that `data/benchmark.dvc` or `data/scratch.dvc` points to yet. Ask them to push it.
