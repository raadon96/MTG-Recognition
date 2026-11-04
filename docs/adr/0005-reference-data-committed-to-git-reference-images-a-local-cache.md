# Reference data is committed to git; reference images are a local cache

The catalog (`catalog.sqlite`), which holds each printing's pHash (ADR 0006), is committed to git, and only `mtg reference update` and `mtg reference build` change it (ADR 0006). Reference images (~11 GB) are a local cache, used only to compute the pHashes, and never versioned. This way every contributor, CI and the benchmark match against the same reference data. We use git rather than DVC because the files are small, need no Drive credentials, and Docker can `COPY` them. Each new catalog, from an update or from a build after the code that builds the catalog changed, adds up to about 19 MB to the history once git compresses it (the catalog details about 15 MB and the pHashes about 3.5 MB, which don't compress), which is acceptable at a few a year. Before ADR 0006 the pHash index was separate files stored one byte per bit, and a new catalog added about 22 MB.

## Considered Options

- **Rebuilding the catalog and pHash index from MTGJSON on every machine**: rejected, because machines would match against different reference data depending on when they last built it.
- **DVC for `data/reference/`**: rejected, because it would need Drive credentials wherever the engine runs, including Docker builds.

## Consequences

- The benchmark baseline records the MTGJSON version, and the `engine` workflow runs on every PR that changes `data/reference/`. A new catalog, from either command, can change benchmark results, so its PR also updates the baseline through `mtg benchmark update-baseline`.
- The committed sync state never describes the local image folder, so image completeness is always checked on disk. When the sync state recorded the images' version, a `git pull` that brought a new catalog left the old image folder looking up to date, so the new printings' images were never downloaded and the pHash index missed them.
