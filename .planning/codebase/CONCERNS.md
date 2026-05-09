# Codebase Concerns

**Analysis Date:** 2026-05-09

## Tech Debt

**Dead module: `deep_matcher.py`:**
- Issue: `DeepCardMatcher` (MobileNetV3-based embedding search) is fully implemented but
  completely unused in production. `pipeline.py` has its call sites commented out with
  "Commented - keeping for future use" notes. The module is only referenced in
  `benchmarks/benchmark_matching.py` and its own `if __name__ == "__main__"` block.
- Files: `src/mtg_recognition/engine/matching/deep_matcher.py`,
  `src/mtg_recognition/engine/pipeline.py` (lines 134, 139)
- Impact: ~343 lines of torch/torchvision/numpydantic imports are loaded transitively in
  benchmarks. The module also duplicates a SQLAlchemy engine (`dest_engine`) at module level
  that will attempt to connect to a CWD-relative path on import.
- Fix approach: Either delete the file entirely (simplest) or promote it to a named
  experimental branch / separate package if deep-learning matching is planned. Remove
  commented-out call sites from `Pipeline._load_index` and `Pipeline.build_index`.

**CWD-relative data paths in multiple modules:**
- Issue: `phash_matcher.py` (lines 17-19), `deep_matcher.py` (lines 22-28), `ocr.py`
  (lines 145, 148), and `settings.py` (lines 40, 46) all use bare string literals like
  `"data/reference/images"` or `"src/mtg_recognition/engine/..."` that resolve relative
  to the process working directory. The app and Docker container work because they are
  always launched from the project root, but tests or library usage from any other
  directory will silently fail to find models or indexes.
- Files: `src/mtg_recognition/engine/matching/phash_matcher.py`,
  `src/mtg_recognition/engine/matching/deep_matcher.py`,
  `src/mtg_recognition/engine/ocr/ocr.py`,
  `src/mtg_recognition/settings.py`,
  `src/mtg_recognition/database.py`
- Impact: Breaks if the package is imported outside the repo root; no clear error — index
  simply reports "not found" and attempts `auto_build` with a non-existent source directory.
- Fix approach: Anchor paths to a well-known absolute location. `settings.py` already
  provides `Settings.DATA_DIR` and `Settings.YOLO_SEG_MODEL_PATH` as `Path` properties;
  thread `settings` through `PHashCardMatcher` and `CardOCR` instead of module-level
  string constants.

**Duplicate `debug` and `debug_info` keys in CLI JSON output:**
- Issue: In `pipeline.py` `main()`, the JSON segment dict is built with a `"debug"` key
  (lines 299-303) and then unconditionally (because `include_debug_info = True` is
  hardcoded) overwritten by a second key `"debug_info"` containing identical data (lines
  305-311). Both keys appear in the output file. `include_debug_info` is never `False` and
  is not a parameter.
- Files: `src/mtg_recognition/engine/pipeline.py` (lines 282-311)
- Impact: Output JSON is larger than necessary; consumers that read `"debug"` get the same
  data as `"debug_info"` with no explanation. The `CLAUDE.md` contract documents only
  `"debug"`, making `"debug_info"` undocumented noise.
- Fix approach: Remove the `include_debug_info` variable and the duplicate `out.update()`
  block. Keep only `"debug"` in the output dict (matches CLAUDE.md spec).

**Dead first assignment in `extract_card_name`:**
- Issue: Line 377 builds `filtered_list` from stripped texts, then line 378 immediately
  overwrites it with a different comprehension using the unstripped `texts`. The first
  assignment is never read.
- Files: `src/mtg_recognition/engine/ocr/ocr_extraction.py` (lines 374-378)
- Impact: Cosmetic dead code; the intended strip-then-filter logic is silently discarded,
  meaning single-character tokens are not stripped before the length check.
- Fix approach: Remove line 377. Ensure the filter on line 378 uses stripped texts:
  `list(filter(lambda text: 2 <= len(text.strip()), texts))`.

**`Pipeline.__init__` default argument objects instantiated at definition time:**
- Issue: `Pipeline.__init__` declares `phash_matcher: PHashCardMatcher = PHashCardMatcher()`
  and `ocr: CardOCR = CardOCR(use_preprocessing=True)` as default parameter values (lines
  34-35). Python evaluates default arguments once at class-definition time, meaning a
  single `PHashCardMatcher` (which immediately loads the 1M-hash index) and a single
  `CardOCR` (which loads PaddleOCR models) are instantiated on module import, regardless
  of whether `Pipeline` is ever constructed.
- Files: `src/mtg_recognition/engine/pipeline.py` (lines 34-35)
- Impact: Any module that `import`s `pipeline` pays the full model-loading cost even if it
  only inspects types or calls unrelated helpers. Makes testing without model files
  impossible.
- Fix approach: Replace with `None` defaults and instantiate inside `__init__` when the
  argument is `None`:
  ```python
  def __init__(self, phash_matcher=None, ocr=None, ...):
      self.phash_matcher = phash_matcher or PHashCardMatcher()
      self.ocr = ocr or CardOCR(use_preprocessing=True)
  ```

**Module-level side effects on import:**
- Issue: `settings.py` calls `settings = Settings()` and `settings.ensure_dirs()` at module
  scope (lines 91-92). `ensure_dirs()` creates `data/reference/images/` and
  `data/reference/tmp/` on disk as a side effect of importing the settings module. The
  `_apply_third_party_env_vars` validator also mutates `os.environ` on every instantiation.
  `database.py` (line 11) creates a SQLAlchemy engine (opens the SQLite file) at module
  scope. `deep_matcher.py` (line 31) creates a second duplicate engine at module scope.
- Files: `src/mtg_recognition/settings.py`,
  `src/mtg_recognition/database.py`,
  `src/mtg_recognition/engine/matching/deep_matcher.py`
- Impact: Importing any module that transitively reaches `settings` or `database` creates
  directories and opens file handles; breaks unit tests that run without the data directory
  present (like in CI without the reference data mounted).
- Fix approach: Wrap `settings = Settings()` in a lazy accessor or move it to an
  application entrypoint. Defer `ensure_dirs()` to the startup path. Consider using a
  `get_engine()` factory instead of a module-level `engine` singleton.

## Known Bugs

**Mutable default argument `candidates: list[Row] = []` in `PHashCardMatcher.find_best_matches`:**
- Symptoms: The default `[]` list is shared across all calls that omit `candidates`. While
  the method only reads the list (never appends to it), this is a latent correctness hazard
  and violates Python convention.
- Files: `src/mtg_recognition/engine/matching/phash_matcher.py` (line 271)
- Same pattern exists in `DeepCardMatcher.find_best_matches` (line 271 of deep_matcher.py).
- Workaround: Always pass `candidates=[]` explicitly (current call sites do this).
- Fix: Use `candidates: list[Row] | None = None` and default to `[]` inside the function
  body.

## Security Considerations

**No issues identified:** Database is opened read-only (`mode=ro&nolock=1`), no user input
is executed as SQL (ORM `select` with `IN` clause), no secrets appear in source.

## Performance Bottlenecks

**`rembg` fallback loads on first use for every new `YoloCardSegmentor` instance:**
- Problem: `RembgCardSegmentor` is lazy-loaded per `YoloCardSegmentor` instance. In
  `batch_processor.py` the `Pipeline` (and therefore `YoloCardSegmentor`) is created once
  and reused, so this is currently fine. But if multiple `Pipeline` instances are created
  (e.g. in tests or future multi-worker scenarios), each will re-load the ~170 MB u2net
  model independently.
- Files: `src/mtg_recognition/engine/segmentation/yolo.py` (line 131, `_get_rembg_segmentor`)
- Improvement path: Consider a module-level or application-level singleton for the rembg
  session, or document that `Pipeline` must be a singleton.

**`_get_indices_from_uuids` uses `list.index()` O(n) lookup in `DeepCardMatcher`:**
- Problem: `_get_idxs_from_uuids` (line 241 of `deep_matcher.py`) calls
  `self.uuid_mapping.index(uuid)` which is O(n) per UUID. For a 1M-card index this is
  ~1M iterations per lookup. `PHashCardMatcher` correctly builds a `uuid_to_idx` dict for
  O(1) lookup; `DeepCardMatcher` does not.
- Files: `src/mtg_recognition/engine/matching/deep_matcher.py` (lines 240-242)
- Impact: Currently dormant (module unused in production). Would be a critical performance
  bug if `DeepCardMatcher` is reactivated with subset-UUID filtering.
- Improvement path: Add a `uuid_to_idx` dict mirror to `DeepCardMatcher` mirroring
  `PHashCardMatcher`.

## Fragile Areas

**Temp PNG files accumulate across Streamlit sessions:**
- Files: `src/mtg_recognition/app/batch_processor.py` (lines 103-107, 191-195),
  `src/mtg_recognition/app/app.py`
- Why fragile: Both the original and warped images are written to `tempfile.NamedTemporaryFile`
  with `delete=False`. The warped path is only cleaned up on exception (line 219-220 in
  `batch_processor.py`). On the success path, `warped_image_temp_path` and
  `original_image_temp_path` are stored in `BatchResult` and never explicitly unlinked.
  Export temp files (CSV, XLSX, JSON) in `app.py` (lines 360-374) are similarly never
  cleaned up. On a long-running Streamlit server, every batch upload accumulates PNG/CSV/
  XLSX/JSON files in the OS temp directory until OS reboot or manual cleanup.
- Safe modification: Wrap the image-display section in `app.py` in a try/finally that
  calls `Path(path).unlink(missing_ok=True)` after the image is read into memory.
  Use `tempfile.TemporaryDirectory` as a context manager in `_match_segments` so all per-
  batch temp files are removed atomically when the batch completes.
- Test coverage: No tests cover temp-file lifecycle.

**Hardcoded year ceiling `2030` in `extract_set_release_year`:**
- Files: `src/mtg_recognition/engine/ocr/ocr_extraction.py` (lines 480, 496)
- Why fragile: MTG cards printed after 2030 will have their copyright years silently
  discarded by the year-range filter, causing `set_release_year` to return `None` and
  degrading OCR match scores for those cards.
- The standalone-number extractor (`_extract_standalone_card_number`, line 242) uses
  `datetime.date.today().year + 1` correctly for its year ceiling — the same dynamic
  approach should be applied in `extract_set_release_year`.
- Safe modification: Replace the `<= 2030` literal with `<= datetime.date.today().year + 1`
  in both occurrences. Add a regression test with a year value of `current_year + 1`.

**`assert` used as a runtime guard in production paths:**
- Files: `src/mtg_recognition/engine/pipeline.py` (line 75),
  `src/mtg_recognition/engine/matching/phash_matcher.py` (lines 124, 156)
- Why fragile: Python's `-O` flag strips `assert` statements entirely. The `assert
  k_final_results <= k_phash_candidates` guard in `Pipeline.find_best_matches` is the
  only contract enforcement; if optimized away, the pipeline silently produces fewer
  results than expected.
- Safe modification: Replace with explicit `if not condition: raise ValueError(...)`.

## Scaling Limits

**Single-process, synchronous batch processor:**
- Current capacity: All segmentation and matching runs in the main Streamlit thread.
- Limit: Large uploads (100+ high-resolution images) block the UI thread for minutes.
  Streamlit's `st.progress` updates are the only feedback.
- Scaling path: Move `BatchProcessor.process_batch` into a background thread or process
  pool using `concurrent.futures.ProcessPoolExecutor`. YOLO's `segment_batch` already
  accepts a `batch_size` parameter to control memory use per call.

## Test Coverage Gaps

**No unit tests for `PHashCardMatcher` or `CardOCR` (integration classes):**
- What's not tested: `PHashCardMatcher.find_best_matches`, `PHashCardMatcher._compute_hash`,
  `CardOCR.extract_properties`, `CardOCR._compute_scores`, `CardOCR.find_best_matches`.
- Files: `src/mtg_recognition/engine/matching/phash_matcher.py`,
  `src/mtg_recognition/engine/ocr/ocr.py`
- Risk: Scoring-weight changes, ROI constant changes, or PaddleOCR API changes break
  silently. The `except RuntimeError` fallback path in `ocr.py` (line 224) has never been
  exercised by a test.
- Priority: High — these are the core matching paths.

**No tests for `Pipeline` or `BatchProcessor`:**
- What's not tested: End-to-end `Pipeline.find_best_matches`, `_match_one` branching
  (single match vs. multi-match), `BatchProcessor.process_batch` mini-batch logic,
  temp-file creation/cleanup, error path (bad image, load failure).
- Files: `src/mtg_recognition/engine/pipeline.py`,
  `src/mtg_recognition/app/batch_processor.py`
- Risk: Refactoring either class risks silent regressions with no test signal.
- Priority: High.

**No tests for `enhance_contrast` (CLAHE + gamma paths):**
- What's not tested: The four brightness-band branches in `enhance_contrast` (very dark,
  dark, very bright, normal).
- Files: `src/mtg_recognition/engine/segmentation/yolo.py` (lines 304-342)
- Risk: Brightness threshold constants can be changed without any test breaking.
- Priority: Medium.

**Regression tests require reference data not in git:**
- What's not tested: `tests/regression/segmentation/test_yolo_reg.py` and
  `test_rembg_reg.py` depend on `benchmarks/dataset/images/` and pre-generated fixture
  files produced by `generate_fixtures.py`. These are absent from the repository.
- Files: `tests/regression/segmentation/`
- Risk: Regression suite cannot run in CI without the reference dataset mounted.
- Priority: Medium — acceptable for a dataset-heavy ML project, but should be documented
  and a CI skip condition added (e.g. `pytest.mark.skipif(not dataset_exists, ...)`).

---

*Concerns audit: 2026-05-09*
