"""Results of recognizing uploaded query images, and their export, for the Streamlit
app.
"""

import io
import json
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

EXPORT_FORMATS = {  # format -> (MIME type, file name)
    "CSV": ("text/csv", "results.csv"),
    "Excel": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "results.xlsx",
    ),
    "JSON": ("application/json", "results.json"),
}

# Exported fields in file order: JSON key -> CSV/Excel column.
_EXPORT_COLUMNS = {
    "uploaded_filename": "Uploaded Filename",
    "status": "Status",
    "card_name": "Card Name",
    "uuid": "UUID",
    "score": "Score",
    "set_code": "Set Code",
    "collector_number": "Collector Number",
    "base_set_size": "Base Set Size",
    "artist": "Artist",
    "scryfall_url": "Scryfall URL",
}


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


def export_bytes(results: list[UploadResult], export_format: str) -> bytes:
    """The export file's contents, built when the user clicks Download.

    Streamlit ignores st.* calls made here, so a failure is raised with an
    "Export failed: ..." message, which Streamlit shows on the download button.
    """
    try:
        # One row per result; a result with no best printing leaves its fields None.
        rows = []
        for result in results:
            row: dict[str, object] = dict.fromkeys(_EXPORT_COLUMNS)
            row["uploaded_filename"] = result.uploaded_filename
            row["status"] = result.status.value
            best = result.best
            if best is not None:
                row["card_name"] = best.printing.name
                row["uuid"] = best.printing.uuid
                row["score"] = best.score
                row["set_code"] = best.printing.set_code
                row["collector_number"] = best.printing.collector_number
                row["base_set_size"] = best.printing.base_set_size
                row["artist"] = best.printing.artist
                row["scryfall_url"] = best.printing.image_url
            row["error_message"] = result.error_message
            rows.append(row)

        if export_format == "JSON":
            return json.dumps(rows, indent=2).encode()

        table = (
            pd.DataFrame(rows)
            .drop(columns="error_message")
            .rename(columns=_EXPORT_COLUMNS)
        )
        match export_format:
            case "CSV":
                return table.to_csv(index=False).encode()
            case "Excel":
                buffer = io.BytesIO()
                # pandas' stub types reject BytesIO, which to_excel accepts.
                table.to_excel(
                    buffer,  # ty: ignore[invalid-argument-type]
                    index=False,
                    sheet_name="Results",
                    engine="openpyxl",
                )
                return buffer.getvalue()
            case _:
                raise ValueError(f"unknown format {export_format!r}")
    except Exception as e:
        raise RuntimeError(f"Export failed: {e}") from e


def _encode_jpeg(img: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _DISPLAY_JPEG_QUALITY]
    )
    if not ok:
        raise ValueError("Failed to encode the card image as JPEG")
    return buffer.tobytes()
