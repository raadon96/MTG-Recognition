# pHashes are stored in the catalog and searched in NumPy

Each printing's pHash is a 32-byte BLOB column of the catalog table, next to its other details, and the MTGJSON version the catalog was built from is stored in the same file. The catalog is the only reference data file. Before, the pHashes lived in `hashes.npy`, matched by row position to the UUIDs in `index_to_uuid.json`, and the versions in `.mtg-sync-state.json`, so three files had to agree. The engine reads every pHash once at startup (about 50 ms) and compares each card image's pHash with all of them in NumPy, as before (about 2.5 ms per card for 115,875 printings). The database only stores the pHashes; it doesn't search them.

## Considered Options

Search times per card were measured on 2026-11-03 on the same 115,875 pHashes:

- **sqlite-vec** (0.1.9; 0.1.10 still alpha): 5.7–8.6 ms. Rejected: still before 1.0 and may break, a loadable extension that some Python builds disable, and slower than NumPy.
- **Turso Database** (pyturso 0.8.3, still beta): 20 ms, with Hamming distance only through `vector_distance_cos` on `vector1bit` values. Rejected for now: beta, and it replaces the database driver. Turso reads SQLite files, so it can be reconsidered for the in-browser port (ADR 0003), along with Turso Cloud sync as a way to distribute the catalog instead of git (ADR 0005).
- **DuckDB** (2.0): 4.9 ms with `bit_count(xor(...))` on four `UBIGINT` columns. Rejected: it means moving the catalog off SQLite and SQLModel, and duckdb-wasm is a much larger download for the browser port.
- **Plain SQLite with a Python Hamming function**: 28 ms. Rejected: slower than NumPy for no gain.

Searching inside the database would only pay off for filtering in the same query (for example by set code), which the engine doesn't do.

## Consequences

- `mtg update` is the only command that builds the reference data: MTGJSON, then the catalog, then the missing reference images, then the pHashes. There is no state where the catalog holds printings that haven't been hashed yet. It reuses the pHash of every printing whose UUID was already in the previous catalog. `--force` downloads every image again and recomputes every pHash, for example after Scryfall replaces a preview image under the same UUID, or if MTGJSON ever gives an existing UUID to a different card (not seen in this repo's catalogs so far).
- `mtg download-images` fills the local image cache for the committed catalog without changing `data/reference/`.
- A printing without a pHash (no image URL, or its image failed to download or hash) is never a candidate.
