# AGENTS.md

This document provides guidelines for agentic coding agents operating in this repository.

## Project Overview

This is a Magic: The Gathering (MTG) card recognition system. It uses:
- **YOLO models** for card segmentation/detection in images
- **Deep learning embeddings** (MobileNetV3) for card matching
- **Perceptual hashing** for fast card lookup
- **OpenCV** for image processing and perspective transforms

## Build/Run Commands

### Setup
```bash
uv sync                    # Install dependencies
source .venv/bin/activate  # Activate virtual environment
```

### Running the Application
```bash
uv run main.py             # Run main camera app
uv run python -m mtg_recognition.engine.segmentation.segment  # Test segmentation
uv run python -m mtg_recognition.engine.matching.deep_matcher  # Test deep matcher
uv run python -m mtg_recognition.engine.matching.phash_matcher  # Test phash matcher
uv run python -m mtg_recognition.engine.matching.hybrid_matcher  # Test hybrid matcher
uv run python -m mtg_recognition.jobs.build_database  # Build reference database
```

### Testing
```bash
uv run pytest                                    # Run all tests
uv run pytest tests/unit/                       # Run unit tests only
uv run pytest tests/integration/                # Run integration tests only
uv run pytest -v                                # Verbose output
uv run pytest -k "test_name"                   # Run specific test by name pattern
uv run pytest tests/unit/test_segmentation.py   # Run specific test file
uv run pytest tests/ -v --tb=short              # Short traceback on failure
```

### Linting & Type Checking
```bash
uv run ruff check src/ tests/                   # Run ruff linter
uv run ruff check --fix src/ tests/             # Auto-fix linting issues
uv run ruff format src/ tests/                  # Format code with ruff
uv run mypy src/                                # Type checking with mypy (if installed)
```

## Code Style Guidelines

### Imports
- Use absolute imports from package root: `from mtg_recognition.engine.matching import CardMatcher`
- Use relative imports within same package: `from .geometry import perspective_transform`
- Group imports: stdlib → third-party → local
- Sort imports alphabetically within groups

### Naming Conventions
- **Classes**: PascalCase (e.g., `CardSegmentor`, `DeepCardMatcher`)
- **Functions/variables**: snake_case (e.g., `find_corners_from_mask`, `img_warped`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `INDEX_PATH`, `BATCH_SIZE`)
- **Private members**: leading underscore (e.g., `_model`, `_load_index`)
- **Type variables**: PascalCase (e.g., `T`, `U`)

### Type Hints
- Use Python 3.12+ syntax with `|` for unions: `np.ndarray | None`
- Use `typing` module for complex types: `list[str]`, `dict[str, int]`
- Use `Path` from `pathlib` instead of strings for file paths
- Use `NDArray` from `numpydantic` for numpy array types with shape constraints

### Error Handling
- Use specific exceptions: `ValueError`, `FileNotFoundError`, `NotImplementedError`
- Use `raise ValueError(f"message {detail}")` for validation errors
- Use `try/except` blocks with specific exception types
- Log errors with `print(f"Error: {e}")` rather than silent failures
- Use `assert` for invariants and internal consistency checks

### Comments
- Use comments sparingly; let code be self-documenting
- Add comments for complex algorithms or non-obvious logic
- Use TODO comments for future work: `TODO(username): description`
- Never leave commented-out code in the codebase

### Docstrings
- Use Google-style docstrings for public functions/classes
- Include Args, Returns, and Raises sections for complex functions
- Keep one-line docstrings for simple functions
- Document public classes; private methods optional

### File Organization
- **src/mtg_recognition/** - Main package (required for uv build)
- **src/mtg_recognition/engine/** - Core CV/ML logic
- **src/mtg_recognition/engine/segmentation/** - Stage 1: card detection
- **src/mtg_recognition/engine/matching/** - Stage 2: card identification
- **src/mtg_recognition/jobs/** - Database/build scripts
- **src/mtg_recognition/app/** - Web API layer (if present)
- **tests/unit/** - Fast tests with mocked I/O
- **tests/integration/** - Slow tests with real I/O
- **tests/fixtures/** - Test images and mock data

### Class Design
- Use `ABC` (abstract base class) for interfaces: `class CardMatcher(ABC)`
- Use Pydantic `BaseModel` for data models with validation
- Use `@property` for computed attributes
- Use `@abstractmethod` for interface methods

### Image Processing Conventions
- Use `np.ndarray` for images (OpenCV format: BGR)
- Use `Path` for file paths; `cv2.imread()`/`cv2.imwrite()` for I/O
- Use `cv2.resize()` with `INTER_LANCZOS4` for high-quality resizing
- Use `cv2.waitKey()` and `cv2.destroyAllWindows()` for visualization cleanup

### Performance
- Use vectorized numpy operations over loops
- Use `torch.no_grad()` for inference
- Use `DataLoader` with multiple workers for batch processing
- Use `@tqdm` progress bars for long-running operations

### Database Access
- Use `sqlmodel` for ORM with SQLAlchemy + Pydantic
- Use `Session` context managers: `with Session(engine) as session:`
- Use `select()` for queries: `session.exec(select(Table).where(...)).first()`

### Key Patterns
- **Pipeline pattern**: `segment() → match()` for recognition flow
- **Strategy pattern**: Different `CardMatcher` implementations (deep/phash/hybrid)
- **Builder pattern**: `build_index()` for database construction
- **Factory pattern**: `YOLOModel` for model loading

## Module Entry Points

Run any module as a script:
```bash
uv run python -m mtg_recognition.engine.segmentation.segment
uv run python -m mtg_recognition.engine.matching.deep_matcher
uv run python -m mtg_recognition.jobs.build_database
```

## Data Paths (Gitignored)
- `data/reference/` - Reference images and database
- `data/reference/index/` - Precomputed embeddings and hashes
- `benchmarks/results/` - Test results and plots
- `scratch/` - Temporary/experimental scripts
