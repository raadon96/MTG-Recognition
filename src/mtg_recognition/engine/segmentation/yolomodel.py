from json import load
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PARAM_PATH_FILE = Path(__file__).parent / "parameters.json"


class YOLOModel:
    """
    A wrapper class for the Ultralytics YOLO model used for image segmentation or detection.

    This class handles model initialization, parameter loading from a JSON configuration file,
    and provides an interface for making predictions on image data.

    Attributes:
        model_name (str): The name of the YOLO model to use.
        parameter_set_idx (int): The index of the parameter set from the configuration file.
        model (YOLO): The underlying Ultralytics YOLO model instance.
        parameters (dict): Configuration parameters loaded for the specific model and index.
    """

    def __init__(self, model_name="yoloe-11s-seg", parameter_set_idx=0):
        self.model_name: str = model_name
        self.parameter_set_idx: int = parameter_set_idx
        self.model = self._load_model()

    def _load_model(self):
        with open(PARAM_PATH_FILE) as file:
            self.parameters = load(file)[self.model_name][str(self.parameter_set_idx)]

        model_path = (
            Path("src/mtg_recognition/engine/segmentation/models/prod")
            / f"{self.model_name}.pt"
        )
        task = self.parameters["task"]
        pytorch_model = YOLO(model_path, task=task)
        if self.parameters["category_allowlist"]:
            pytorch_model.set_classes(self.parameters["category_allowlist"])
            pytorch_model.info()
        return pytorch_model

    @property
    def task(self):
        return self.parameters["task"]

    def predict(self, img: np.ndarray) -> np.ndarray | None:
        result: list = self.model(
            source=img,
            conf=self.parameters["conf"],
            device="cpu",
            imgsz=640,
            max_det=1,
            iou=self.parameters["iou"],
            show=False,
            vid_stride=False,
            verbose=False,
        )[0]

        if self.task == "segment":
            if result.masks is None:
                return None
            # shape = (N, 1, 2)
            return result.masks.xy[0][:, np.newaxis, :].astype(np.int32)
        else:
            if self.task == "detect":
                return None
            # shape = (4,)
            return result.boxes.xyxy[0].numpy().ravel().astype(np.int64)
