# Licensed under AGPL-3.0-only

The repository and the published Docker image are licensed under AGPL-3.0-only. Segmentation uses YOLOE weights exported from Ultralytics, and Ultralytics licenses its code and weights, trained models included, under AGPL-3.0 (#191). Publishing the image distributes those weights, so the whole app has to be AGPL-3.0. The only other way out is buying an Ultralytics Enterprise licence. Our own pre- and post-processing, and running the weights as ONNX without `ultralytics` (ADR 0007), don't change that.

The licence is declared in a root `LICENSE`, `pyproject.toml` and the image's OCI label, with no per-file headers. AGPL §13 requires a modified copy served over a network to offer its source, so the app shows a footer, "Source code (AGPL-3.0)", linking to the repository. Forks that change the app must point that link at their own source.

## Considered Options

- **AGPL-3.0-or-later**: rejected. It would commit the project to GPL versions not yet written, and "only" matches the Ultralytics terms exactly.
- **A permissive licence (MIT, Apache-2.0) for the code, AGPL only for the image**: rejected. Two licences for one app are confusing, and the image, which is what users run, would be AGPL anyway.
- **A segmentation model under a permissive licence**: not looked for. Every candidate compared in #195 was an Ultralytics YOLOE model, and YOLOE-11s is what passes the benchmark today. If a permissive model is adopted later, moving off AGPL needs the agreement of everyone who has contributed under it, which makes the licence hard to undo.

## Consequences

- AGPL allows commercial use of the code, but card names, text and images are Wizards of the Coast's, used under the Fan Content Policy, which requires the app to stay free. The README carries the policy's standard notice word for word.
- The shipped OCR and silueta models are Apache-2.0. Their licence text goes in `THIRD_PARTY_NOTICES.md`, which is also copied into the image.
- MobileCLIP (research-only terms), `ultralytics` and `torch` are used only to export the segmentation model (the `export` dependency group). They are neither in the repository nor in the image, so they need no notice.
