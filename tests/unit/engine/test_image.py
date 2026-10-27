"""prepare_image: the one way the engine's input pixels are produced."""

import io

import numpy as np
import pytest
from PIL import Image

from mtg_recognition.engine.image import prepare_image
from mtg_recognition.settings import settings

_EXIF_ORIENTATION = 0x0112
_ROTATE_90_CW = 6


def _encoded(width: int, height: int, fmt: str = "PNG", **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 100, 50)).save(
        buffer, format=fmt, **save_kwargs
    )
    return buffer.getvalue()


def test_exif_orientation_is_applied():
    exif = Image.Exif()
    exif[_EXIF_ORIENTATION] = _ROTATE_90_CW
    sideways = _encoded(40, 20, "JPEG", exif=exif)

    assert prepare_image(sideways).shape == (40, 20, 3)


def test_a_large_image_is_shrunk_to_the_max_dimension_on_its_longest_edge(
    monkeypatch,
):
    monkeypatch.setattr(settings, "MAX_IMAGE_DIMENSION", 100)

    assert prepare_image(_encoded(400, 300)).shape == (75, 100, 3)


def test_a_small_image_is_left_untouched():
    img = prepare_image(_encoded(30, 20))

    assert img.shape == (20, 30, 3)
    assert (img == np.array([50, 100, 200], dtype=np.uint8)).all()  # BGR


def test_a_path_is_accepted(tmp_path):
    path = tmp_path / "card.png"
    path.write_bytes(_encoded(30, 20))

    assert prepare_image(path).shape == (20, 30, 3)


def test_bytes_that_are_not_an_image_raise():
    with pytest.raises(ValueError):
        prepare_image(b"not an image")
