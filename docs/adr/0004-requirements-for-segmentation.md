# Requirements for segmentation

Any segmentation design must meet these requirements, because the engine is the reference implementation for the in-browser port (ADR 0003). How a design meets them is left to the code.

- It finds several cards in one query image.
- It runs locally, and the in-browser port (WASM/ONNX) can run it too.
- It works on natural photos: the app encourages a plain, bright background but doesn't require one.
- A query image that the scan bypass accepts counts as exactly one card.
- Changes are judged by the benchmark.
