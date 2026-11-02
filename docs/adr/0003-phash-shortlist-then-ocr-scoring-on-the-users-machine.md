# Recognition is a pHash shortlist then OCR scoring, on the user's machine

The engine is the reference implementation for a later in-browser (WASM) port. Matching shortlists candidates by pHash similarity, then scores each against the card's reading from OCR; there are no learned embeddings and no cloud inference. This keeps running costs low and recognition local, unlike competitors that run inference in the cloud. pHash and OCR were accurate enough without image embeddings from a learned model run through torch (such as MobileNetV3), and pHash ports easily to the browser.

## Consequences

- A Raspberry Pi is no longer a target; the in-browser port is.
- Running on CPU is today's setup, not a rule: a GPU path (CUDA here, WebGPU in the browser through ONNX Runtime Web) may be added as an option.
- Thresholds and weights come from early manual tests and may be changed as long as the benchmark holds or improves.
