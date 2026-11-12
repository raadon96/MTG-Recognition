"""Write the segmentation model, YOLOE-11s with our prompts, as an ONNX export.

Run from the repository root:

    uv run --group export python scripts/export_segmentation.py

Run it again when the prompts, the model or the ultralytics version change, then
commit its output, models/segmentation/prod/yoloe-11s-seg.onnx, with a baseline
updated through `mtg benchmark update-baseline` if the benchmark results change.
The engine runs the export on onnxruntime with its own pre- and post-processing
(docs/adr/0007); only this script needs ultralytics and torch.

The prompts are fused into the exported weights, so the text encoder isn't
needed at runtime. The export is dynamic: a square (static) input pads query
images differently from ultralytics and changes results (#195).
"""

import os
import shutil
from pathlib import Path

# ultralytics would otherwise pip-install what it finds missing
os.environ["YOLO_AUTOINSTALL"] = "false"

# only the export group installs ultralytics, so ty can't find it in the dev venv
from ultralytics import YOLO  # noqa: E402  # ty: ignore[unresolved-import]

PROMPTS = ["book cover", "art work"]
WEIGHTS = "yoloe-11s-seg.pt"
OUTPUT = Path("models/segmentation/prod/yoloe-11s-seg.onnx").resolve()
# ultralytics downloads the weights and the text encoder (mobileclip_blt.ts,
# 600 MB) into the working directory, so work in a gitignored folder that keeps
# them for the next run
WORK_DIR = Path(".scratch/export-segmentation").resolve()


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK_DIR)

    model = YOLO(WEIGHTS)
    model.set_classes(PROMPTS)
    exported = model.export(
        format="onnx",
        dynamic=True,
        agnostic_nms=True,
        imgsz=640,
        opset=20,
        max_det=50,
    )

    shutil.copyfile(exported, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
