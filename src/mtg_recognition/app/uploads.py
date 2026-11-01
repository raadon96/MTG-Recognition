"""Results of recognizing uploaded query images, and their export, for the Streamlit
app.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np
import pandas as pd

from mtg_recognition.engine.ocr.reading import ReadingDebug
from mtg_recognition.engine.pipeline import RecognitionResult
from mtg_recognition.engine.ranking import RankedPrinting

# Quality of the card image's display copy; recognition never sees it.
_DISPLAY_JPEG_QUALITY = 90


@dataclass(frozen=True)
class Upload:
    """One uploaded query image file: its name and its encoded bytes, as uploaded."""

    name: str
    data: bytes


class UploadStatus(StrEnum):
    SUCCESS = "success"
    NOT_RECOGNIZED = "not_recognized"
    ERROR = "error"


@dataclass
class UploadResult:
    """Result for one card found in an uploaded query image, or for a query image with
    no card or an error.
    """

    uploaded_filename: str
    query_image: bytes  # as uploaded
    status: UploadStatus
    ranked: list[RankedPrinting] = field(default_factory=list)  # best first
    error_message: str | None = None
    card_image_jpeg: bytes | None = None  # for display only
    reading_debug: ReadingDebug | None = None

    @property
    def best(self) -> RankedPrinting | None:
        """The best printing's entry, or None when the ranking is empty."""
        return self.ranked[0] if self.ranked else None


# Exported fields of the best printing: (JSON key, CSV/Excel column, value).
_BEST_PRINTING_FIELDS: list[tuple[str, str, Callable[[RankedPrinting], object]]] = [
    ("card_name", "Card Name", lambda b: b.printing.name),
    ("uuid", "UUID", lambda b: b.printing.uuid),
    ("score", "Score", lambda b: b.score),
    ("set_code", "Set Code", lambda b: b.printing.set_code),
    ("collector_number", "Collector Number", lambda b: b.printing.collector_number),
    ("base_set_size", "Base Set Size", lambda b: b.printing.base_set_size),
    ("artist", "Artist", lambda b: b.printing.artist),
    ("scryfall_url", "Scryfall URL", lambda b: b.printing.image_url),
]


def _best_printing_fields(result: UploadResult, as_columns: bool) -> dict[str, object]:
    """The exported fields of the result's best printing (all None without one).

    Keyed by JSON key or column name.
    """
    best = result.best
    return {
        (column if as_columns else key): (get(best) if best is not None else None)
        for key, column, get in _BEST_PRINTING_FIELDS
    }


def upload_results(
    upload: Upload, recognition: RecognitionResult
) -> list[UploadResult]:
    """The results of recognizing one upload: one per card found, or a single one for
    a query image with no card or an error.
    """
    name, data = upload.name, upload.data

    if recognition.error is not None:
        return [
            UploadResult(
                name, data, UploadStatus.ERROR, error_message=recognition.error
            )
        ]
    if not recognition.cards:
        return [UploadResult(name, data, UploadStatus.NOT_RECOGNIZED)]

    return [
        UploadResult(
            name,
            data,
            UploadStatus.SUCCESS if card.ranked else UploadStatus.NOT_RECOGNIZED,
            ranked=card.ranked,
            card_image_jpeg=_encode_jpeg(card.card_image),
            reading_debug=(card.reading.debug if card.reading is not None else None),
        )
        for card in recognition.cards
    ]


def results_table(results: list[UploadResult]) -> pd.DataFrame:
    """One row per result: its upload, status and best printing (CSV/Excel export)."""
    return pd.DataFrame(
        [
            {
                "Uploaded Filename": result.uploaded_filename,
                "Status": result.status.value,
                **_best_printing_fields(result, as_columns=True),
            }
            for result in results
        ]
    )


def results_json(results: list[UploadResult]) -> str:
    """One item per result: its upload, status, best printing and error (JSON
    export).
    """
    return json.dumps(
        [
            {
                "uploaded_filename": result.uploaded_filename,
                "status": result.status.value,
                **_best_printing_fields(result, as_columns=False),
                "error_message": result.error_message,
            }
            for result in results
        ],
        indent=2,
    )


def _encode_jpeg(img: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _DISPLAY_JPEG_QUALITY]
    )
    if not ok:
        raise ValueError("Failed to encode the card image as JPEG")
    return buffer.tobytes()
