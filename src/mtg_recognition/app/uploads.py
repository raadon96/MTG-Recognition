"""Results of recognizing uploaded query images, and their export, for the Streamlit
app.
"""

import io
import json
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from mtg_recognition.engine.ocr.reading import Reading
from mtg_recognition.engine.pipeline import RecognitionResult
from mtg_recognition.engine.ranking import RankedPrinting

# Quality of the display copies of the card image and query image; recognition
# never sees them.
_DISPLAY_JPEG_QUALITY = 90

# In pixels, the longest side of the query image's display copy: twice the width
# the app shows it at, so it stays crisp on high-density screens. A phone photo
# as uploaded is several MB, sent again on every script run that shows it.
_QUERY_IMAGE_MAX_DIMENSION = 800

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


@dataclass(slots=True)
class Recognized:
    """A card found in an uploaded query image, with its ranking."""

    uploaded_filename: str
    query_image_jpeg: bytes  # for display only
    ranked: list[RankedPrinting]  # best first, never empty
    card_image_jpeg: bytes  # for display only
    reading: Reading | None  # None when OCR failed


@dataclass(slots=True)
class NotRecognized:
    """A query image with no card, or a card with no candidates."""

    uploaded_filename: str
    query_image_jpeg: bytes  # for display only


@dataclass(slots=True)
class Failed:
    """A query image that recognition could not process."""

    uploaded_filename: str
    query_image_jpeg: bytes | None  # for display only; None when it can't be decoded
    error_message: str


UploadResult = Recognized | NotRecognized | Failed


def upload_results(
    name: str, data: bytes, recognition: RecognitionResult
) -> list[UploadResult]:
    """The results of recognizing one uploaded query image: one per card found, or a
    single one for a query image with no card or an error.
    """
    if recognition.error is not None:
        # an unreadable or oversized query image is one reason recognition fails
        try:
            query_image_jpeg = _encode_query_image(data)
        except Exception:
            query_image_jpeg = None
        return [Failed(name, query_image_jpeg, recognition.error)]

    query_image_jpeg = _encode_query_image(data)
    if not recognition.cards:
        return [NotRecognized(name, query_image_jpeg)]

    results: list[UploadResult] = []
    for card in recognition.cards:
        if card.ranked:
            results.append(
                Recognized(
                    name,
                    query_image_jpeg,
                    ranked=card.ranked,
                    card_image_jpeg=_encode_jpeg(card.card_image),
                    reading=card.reading,
                )
            )
        else:
            results.append(NotRecognized(name, query_image_jpeg))
    return results


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
            row["error_message"] = None
            match result:
                case Recognized():
                    best = result.ranked[0]
                    row["status"] = "success"
                    row["card_name"] = best.printing.name
                    row["uuid"] = best.printing.uuid
                    row["score"] = best.score
                    row["set_code"] = best.printing.set_code
                    row["collector_number"] = best.printing.collector_number
                    row["base_set_size"] = best.printing.base_set_size
                    row["artist"] = best.printing.artist
                    row["scryfall_url"] = best.printing.image_url
                case NotRecognized():
                    row["status"] = "not_recognized"
                case Failed():
                    row["status"] = "error"
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


def _encode_query_image(data: bytes) -> bytes:
    """The query image's display copy: turned upright by its EXIF orientation, as
    the browser would show the upload, and shrunk to _QUERY_IMAGE_MAX_DIMENSION.
    """
    size = (_QUERY_IMAGE_MAX_DIMENSION, _QUERY_IMAGE_MAX_DIMENSION)
    with Image.open(io.BytesIO(data)) as image:
        # a JPEG decodes straight at a fraction of its size, far faster than in full
        image.draft("RGB", size)
        upright = ImageOps.exif_transpose(image).convert("RGB")
    upright.thumbnail(size)
    buffer = io.BytesIO()
    upright.save(buffer, "JPEG", quality=_DISPLAY_JPEG_QUALITY)
    return buffer.getvalue()


def _encode_jpeg(img: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _DISPLAY_JPEG_QUALITY]
    )
    if not ok:
        raise ValueError("Failed to encode the card image as JPEG")
    return buffer.tobytes()
