import numpy as np
from PIL import Image

from mtg_recognition.engine.ocr.reading import (
    CardReader,
    OcrResult,
    PaddleOcr,
    Reading,
    ReadingDebug,
    upscale_for_ocr,
)

# A 1000 x 700 card: the name crop is rows 40:110, columns 42:630; the crop below
# the text box is rows 880:1000, columns 35:665.
CARD_H, CARD_W = 1000, 700
NAME_SHAPE = (70, 588, 3)
BELOW_TEXT_BOX_SHAPE = (120, 630, 3)
NAME_PIXEL, BELOW_TEXT_BOX_PIXEL = 1, 2


def card_image() -> np.ndarray:
    """A blank card with the name and below-text-box regions painted in."""
    img = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
    img[40:110, 42:630] = NAME_PIXEL
    img[880:1000, 35:665] = BELOW_TEXT_BOX_PIXEL
    return img


def ocr_reading(by_pixel: dict[int, OcrResult]):
    """An OCR step that reads each crop by the pixel value it is painted with."""

    def ocr(crops: list[np.ndarray]) -> list[OcrResult]:
        results = []
        for crop in crops:
            [value] = np.unique(crop)
            results.append(by_pixel.get(int(value), OcrResult([], [], [])))
        return results

    return ocr


def lower(texts: list[str], boxes=None) -> OcrResult:
    """Tokens below the text box, by default on one line at the bottom."""
    if boxes is None:
        boxes = [(10 + 100 * i, 80, 100 + 100 * i, 110) for i in range(len(texts))]
    return OcrResult(texts, boxes, [0.9] * len(texts))


class TestCardReaderRead:
    def test_the_ocr_step_gets_the_name_crop_then_the_crop_below_the_text_box(self):
        seen = []

        def ocr(crops):
            seen.extend(crop.shape for crop in crops)
            return [OcrResult([], [], []) for _ in crops]

        CardReader(ocr=ocr).read(card_image())

        assert seen == [NAME_SHAPE, BELOW_TEXT_BOX_SHAPE]

    def test_the_name_crop_feeds_the_name_and_the_lower_crop_the_other_fields(self):
        ocr = ocr_reading(
            {
                NAME_PIXEL: OcrResult(["Enforcer Griffin"], [(0, 0, 500, 60)], [0.99]),
                BELOW_TEXT_BOX_PIXEL: lower(
                    ["011/264", "WAR·EN", "BOMAS GNMR", "TM & 2019 Wizards", "3/4"]
                ),
            }
        )

        reading = CardReader(ocr=ocr).read(card_image())

        assert reading == Reading(
            name="Enforcer Griffin",
            set_code_candidates=["WAR"],
            number="011",
            base_set_size="264",
            artist="BOMAS GNMR",
            set_release_year="2019",
            power="3",
            toughness="4",
        )

    def test_flavor_text_is_dropped_at_the_thresholds_production_uses(self):
        # The crop below the text box is 120 x 630: tokens starting above y = 9.6
        # and ending left of x = 567 are flavor text.
        ocr = ocr_reading(
            {
                BELOW_TEXT_BOX_PIXEL: lower(
                    ["trials", "3/3", "BUY CHRISTIAN", "048/271", "TM & 2023 Wizards"],
                    boxes=[
                        (300, 8, 530, 20),  # top-left: flavor text
                        (575, 2, 600, 20),  # top-right: P/T
                        (10, 15, 300, 40),  # below the threshold: kept
                        (10, 80, 150, 120),
                        (400, 80, 630, 120),
                    ],
                )
            }
        )

        reading = CardReader(ocr=ocr).read(card_image(), debug=True)

        assert reading.debug.texts == [
            "3/3",
            "BUY CHRISTIAN",
            "048/271",
            "TM & 2023 Wizards",
        ]
        assert (reading.power, reading.toughness) == ("3", "3")

    def test_empty_ocr_results_give_an_empty_reading(self):
        reading = CardReader(ocr=ocr_reading({})).read(card_image())

        assert reading == Reading()

    def test_debug_attaches_the_tokens_below_the_text_box_and_the_annotated_image(
        self,
    ):
        annotated = Image.new("RGB", (4, 4))
        below = OcrResult(
            ["048/271", "ONE·EN"],
            [(10, 80, 150, 120), (200, 80, 300, 120)],
            [0.98, 0.93],
            annotated_image=annotated,
        )
        ocr = ocr_reading({BELOW_TEXT_BOX_PIXEL: below})

        reading = CardReader(ocr=ocr).read(card_image(), debug=True)

        assert reading.debug == ReadingDebug(
            annotated_image=annotated,
            texts=["048/271", "ONE·EN"],
            scores=[0.98, 0.93],
            boxes=[(10, 80, 150, 120), (200, 80, 300, 120)],
        )

    def test_without_debug_nothing_is_attached(self):
        ocr = ocr_reading({BELOW_TEXT_BOX_PIXEL: lower(["048/271"])})

        reading = CardReader(ocr=ocr).read(card_image())

        assert reading.debug is None


class PaddleResult(dict):
    """A stand-in for one PaddleOCR result: a dict with an annotated image."""

    def __init__(self, annotated: Image.Image, **fields):
        super().__init__(**fields)
        self.img = {"ocr_res_img": annotated}


class StubPaddleOCR:
    """Finds one token spanning the whole of each image it is sent."""

    def __init__(self):
        self.annotated = Image.new("RGB", (4, 4))

    def predict(self, images, **_):
        return [
            PaddleResult(
                self.annotated,
                rec_texts=["token"],
                rec_boxes=np.array([[0, 0, img.shape[1], img.shape[0]]]),
                rec_scores=[0.9],
            )
            for img in images
        ]


class BrokenPaddleOCR:
    def predict(self, images, **_):
        raise RuntimeError("inference failed")


class TestPaddleOcr:
    def test_boxes_are_in_the_pixels_of_the_crops_it_was_given(self):
        # The crop below the text box is upscaled and the tiny name crop padded
        # before inference; the boxes still span exactly the crops passed in.
        name_crop = np.zeros((10, 100, 3), dtype=np.uint8)
        below_text_box_crop = np.zeros((120, 630, 3), dtype=np.uint8)

        name, below = PaddleOcr(paddleocr=StubPaddleOCR())(
            [name_crop, below_text_box_crop]
        )

        assert name.boxes == [(0, 0, 100, 10)]
        assert below.boxes == [(0, 0, 630, 120)]

    def test_texts_scores_and_the_annotated_image_are_passed_through(self):
        paddleocr = StubPaddleOCR()
        crops = [np.zeros((70, 588, 3), np.uint8), np.zeros((120, 630, 3), np.uint8)]

        _, below = PaddleOcr(paddleocr=paddleocr)(crops)

        assert below.texts == ["token"]
        assert below.scores == [0.9]
        assert below.annotated_image is paddleocr.annotated

    def test_a_failed_prediction_gives_empty_results(self):
        crops = [np.zeros((70, 588, 3), np.uint8), np.zeros((120, 630, 3), np.uint8)]

        results = PaddleOcr(paddleocr=BrokenPaddleOCR())(crops)

        assert results == [OcrResult([], [], []), OcrResult([], [], [])]


class TestUpscaleForOcr:
    def test_small_text_is_upscaled_to_the_target_height(self):
        # Text is ~15% of a 120 px crop (18 px); 48 px needs a scale of 8/3.
        img = np.zeros((120, 630, 3), dtype=np.uint8)

        assert upscale_for_ocr(img).shape == (320, 1680, 3)

    def test_the_scale_is_capped_at_four(self):
        img = np.zeros((20, 100, 3), dtype=np.uint8)

        assert upscale_for_ocr(img).shape == (80, 400, 3)

    def test_large_text_is_never_downscaled(self):
        img = np.zeros((400, 630, 3), dtype=np.uint8)

        assert upscale_for_ocr(img).shape == (400, 630, 3)
