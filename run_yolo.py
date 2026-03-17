from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import click  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def get_task_from_yolo_model_name(model_name: str) -> str | None:
    end = model_name.split("-")[-1]
    match end:
        case "seg":
            return "segment"
        case "pose":
            return "pose"
        case "obb":
            return "obb"
        case _:
            return "detect"


def load_pytorch_model(model_name: str, classes: list[str] | None = None):
    task = get_task_from_yolo_model_name(model_name)
    pt_model_path = (
        Path("src/mtg_recognition/engine/segmentation/models/test") / f"{model_name}.pt"
    )
    pytorch_model = YOLO(pt_model_path, task=task)
    if classes:
        pytorch_model.set_classes(classes)
    pytorch_model.info()
    return pytorch_model


def load_model(
    model_name: str,
    onnx_format: bool = True,
    classes: list[str] | None = None,
    reexport: bool = False,
):
    task = get_task_from_yolo_model_name(model_name)
    if onnx_format:
        onnx_model_path = (
            Path("src/mtg_recognition/engine/segmentation/models/test")
            / f"{model_name}.onnx"
        )
        # try loading onnx model if found
        if not onnx_model_path.exists() or reexport:
            # download/load pytorch model and export to onnx
            pytorch_model = load_pytorch_model(model_name, classes)
            pytorch_model.export(format="onnx")
        return YOLO(onnx_model_path, task=task)
    return load_pytorch_model(model_name, classes)


@click.command()
@click.option("-i", "--device-id", type=click.INT, default=4)
@click.option(
    "-m",
    "--model-name",
    type=click.Choice(
        [
            "yoloe-11s-seg",
            "yolov8s-worldv2",
            "yolov8m-worldv2",
            "yoloe-26n-seg",
        ]
    ),
    default="yoloe-11s-seg",
    help="Name of the YOLO model to load (default: yoloe-11s-seg).",
)
@click.option(
    "--onnx-format",
    type=bool,
    is_flag=True,
    help="Whether to load the model in ONNX format (default: True). If False, loads the PyTorch model.",
)
@click.option("-c", "--classes", multiple=True, type=click.STRING, default="book")
@click.option("--conf", type=click.FloatRange(0.0, 1.0), default=0.2)
@click.option(
    "--max-det", type=click.INT, default=1, help="Maximum number of detections."
)
def main(
    device_id: int,
    model_name: str,
    onnx_format: bool,
    classes: list[str],
    conf: float,
    max_det: int,
):
    model = load_model(model_name, onnx_format, classes, reexport=True)
    results = model(
        source=device_id,
        device="cpu",
        imgsz=640,
        show=True,
        conf=conf,
        show_labels=True,
        max_det=max_det,
    )[0]


if __name__ == "__main__":
    main()


# uv run run_yolo.py -i 4 -m yoloe-11s-seg -c "book" --conf 0.1 --max-det=1
# uv run run_yolo.py -i 4 -m yoloe-26n-seg -c "book" --conf 0.1 --max-det=1

# uv run run_yolo.py -i 4 -m yoloe-26n-seg -c "game card" --conf 0.01 --max-det=1 # shitty
# uv run run_yolo.py -i 4 -m yoloe-26n-seg -c "card" --conf 0.01 --max-det=1 # shitty


# uv run run_yolo.py -i 4 -m yolov8s-worldv2 -c "card" --conf 0.2 --max-det=1 # best
# uv run run_yolo.py -i 4 -m yolov8s-worldv2 -c "game card" --conf 0.005 --max-det=1 # shitty

# uv run run_yolo.py -i 4 -m yolov8m-worldv2 -c "card" --conf 0.2 --max-det=1


# uv run run_yolo.py -i 4 -m yolov8s-worldv2 -c "MTG card" --conf 0.01 --max-det=1
# uv run run_yolo.py -i 4 -m yolov8s-worldv2 -c "Magic: The Gathering Card" --conf 0.01 --max-det=1
