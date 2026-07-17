from PIL import Image
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from fast_plate_ocr.inference.hub import download_model

from config.config import config

# Plates wider than they are tall by this ratio are treated as a normal single-line
# plate. Anything squarer/taller than this is assumed to be a stacked two-line plate
# (common on Indian two-wheelers) and gets split before OCR. This is a heuristic based
# on typical plate proportions, not a hard rule — tune it against your own dataset if
# single-line plates are getting misclassified as two-line or vice versa.
TWO_LINE_ASPECT_THRESHOLD = 2.2


def load_ocr_model():
    """
    Loads and returns a fast-plate-ocr LicensePlateRecognizer. Call this once and
    reuse it across images, same reasoning as loading the YOLO model once in
    detection.py — the model load itself is the expensive part, not each prediction.

    Unlike EasyOCR this is a model trained specifically on cropped license plates
    rather than a general-purpose text reader, so it should be meaningfully more
    accurate on plate character recognition specifically.

    Weights are downloaded (on first run only) into config.ocr_dir rather than the
    library's default ~/.cache/fast-plate-ocr, so model files stay inside the repo's
    models/ directory alongside the YOLO weights.
    """
    onnx_path, plate_config_path = download_model(
        config.ocr_model_name,
        save_directory=config.ocr_dir / config.ocr_model_name,
    )
    return LicensePlateRecognizer(
        onnx_model_path=onnx_path,
        plate_config_path=plate_config_path,
    )


def _looks_two_line(img: Image.Image) -> bool:
    return (img.width / img.height) < TWO_LINE_ASPECT_THRESHOLD


def _split_two_line(img: Image.Image):
    """
    Naive top/bottom split down the vertical midpoint. Works when both rows are
    roughly equal height, which covers most stacked-plate layouts, but will cut
    into characters if the two rows are uneven — there's no line-detection here,
    just a fixed 50/50 split.
    """
    mid = img.height // 2
    top = img.crop((0, 0, img.width, mid))
    bottom = img.crop((0, mid, img.width, img.height))
    return top, bottom


def recognize_plate(model: LicensePlateRecognizer, cropped_plate: Image.Image):
    """
    Runs OCR on a single cropped plate image (the kind returned by crop_detections
    in detection.py) and returns the best-guess plate text, or None if nothing
    readable was found.

    Plates that look like stacked two-line layouts (by aspect ratio) are split into
    top/bottom halves and recognized separately, then joined — the underlying model
    is single-line, so this handles the layout problem outside the model itself.

    model.run() returns a list of PlatePrediction objects (even without
    return_confidence=True) — the actual text is in .plate, not the object itself.
    """
    if _looks_two_line(cropped_plate):
        top, bottom = _split_two_line(cropped_plate)
        top_result = model.run(np.array(top.convert("RGB")))
        bottom_result = model.run(np.array(bottom.convert("RGB")))
        top_text = top_result[0].plate if top_result else ""
        bottom_text = bottom_result[0].plate if bottom_result else ""
        combined = f"{top_text}{bottom_text}".strip()
        return combined or None

    result = model.run(np.array(cropped_plate.convert("RGB")))
    if not result:
        return None
    return result[0].plate
