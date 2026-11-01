# Reference data is committed to git; reference images are a local cache

The catalog (`catalog.sqlite`) and the pHash index are committed to git, and only `mtg update` and `mtg build-index` change them. Reference images (~11 GB) are a local cache, used only by `mtg build-index`, and never versioned. This way every contributor, CI and the benchmark match against the same reference data. We use git rather than DVC because the files are small, need no Drive credentials, and Docker can `COPY` them. Each update adds up to about 22 MB to the history once git compresses it (the catalog about 15 MB and the pHash index about 7 MB; 37 MB and 34 MB uncompressed), which is acceptable at a few updates a year.

## Considered Options

- **Rebuilding the catalog and pHash index from MTGJSON on every machine**: rejected, because machines would match against different reference data depending on when they last built it.
- **DVC for `data/reference/`**: rejected, because it would need Drive credentials wherever the engine runs, including Docker builds.

## Consequences

- The benchmark baseline records the MTGJSON version, and the `engine` workflow runs on every PR that changes `data/reference/`. A reference data update changes benchmark results, so its PR also updates the baseline through `mtg benchmark update-baseline`.
- The committed sync state never describes the local image folder, so image completeness is always checked on disk. When the sync state recorded the images' version, a `git pull` that brought a new catalog left the old image folder looking up to date, so the new printings' images were never downloaded and the pHash index missed them.
