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
        save_directory=config.ocr_pretrained_dir / config.ocr_model_name,
    )
    return LicensePlateRecognizer(
        onnx_model_path=onnx_path,
        plate_config_path=plate_config_path,
    )

def load_finetuned_ocr_model():
    """
    Loads the fine-tuned, single-line-only model (see config.ocr_finetuned_model_onnx)
    for use as the primary OCR path on single-line plates. Measured at ~93% whole-plate
    accuracy on a held-out Indian plate validation set, vs. ~58% for the base pretrained
    model on the same distribution — this is the model recognize_plate() should be
    given for its `single_line_model` argument in production use.
 
    Raises FileNotFoundError with a clear message if the exported ONNX file isn't
    present yet, rather than a confusing error from deeper in fast-plate-ocr. — run
    scripts/finetune_ocr.py and export to ONNX first if this fires.
    """
    if not config.ocr_finetuned_model_onnx.exists():
        raise FileNotFoundError(
            f"Fine-tuned OCR model not found at {config.ocr_finetuned_model_onnx}. "
            "Run scripts/finetune_ocr.py, then export the chosen .keras checkpoint "
            "to ONNX (fast_plate_ocr export-onnx) before calling this."
        )
    return LicensePlateRecognizer(
        onnx_model_path=config.ocr_finetuned_model_onnx,
        plate_config_path=config.ocr_finetune_plate_config,
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

 
def validate_ocr_result(text: str, avg_char_confidence: float):
    """
    Decides whether an OCR prediction is trustworthy enough to log, vs. flagged for
    human review. Two independent checks:
      - length: predictions shorter than config.ocr_min_valid_length are almost always
        partial/garbage reads, not real plates, regardless of how confident the model
        claims to be about those few characters
      - confidence: avg_char_confidence below config.ocr_min_avg_char_confidence means
        the model itself wasn't sure, even if the text looks plausible

    Returns (is_valid: bool, reason: str) — reason is "" when valid, otherwise a short
    string explaining which check failed, useful for logging/debugging why something
    got flagged rather than just a bare rejection.

    Deliberately does NOT include a plate-format/regex check (e.g. state-code prefix
    patterns) — I don't have a confirmed, verified specification of your actual target
    plate format(s) to encode correctly, and guessing one wrong would silently reject
    valid plates or accept invalid ones. Add that check yourself once you have a
    format you've actually confirmed, rather than trusting one I'd have to guess at.
    """
    if len(text) < config.ocr_min_valid_length:
        return False, f"too short ({len(text)} chars, min {config.ocr_min_valid_length})"
    if avg_char_confidence < config.ocr_min_avg_char_confidence:
        return False, f"low confidence ({avg_char_confidence:.2f}, min {config.ocr_min_avg_char_confidence})"
    return True, ""


def recognize_plate_with_validation(
    single_line_model: LicensePlateRecognizer,
    two_line_model: LicensePlateRecognizer,
    cropped_plate: Image.Image,
):
    """
    Like recognize_plate(), but requests per-character confidence from the model
    (return_confidence=True) and runs validate_ocr_result() on the result — use this
    in the batch/production path where results get logged to a database, so garbage
    reads get flagged instead of silently written as fact. recognize_plate() itself
    is left unchanged for the GUI/labeling tools, which don't need this.

    Returns a dict: {"text": str | None, "avg_char_confidence": float, "is_valid": bool, "reason": str}

    NOTE ON char_probs: fast-plate-ocr's return_confidence=True populates
    PlatePrediction.char_probs, but I have not independently confirmed whether it's
    aligned to the full fixed-length output (including padding slots) or only the
    decoded plate's actual characters — the docs don't spell this out explicitly.
    This function defensively slices char_probs to len(text) before averaging, on
    the assumption padding (if present) trails the real characters. Print
    pred.char_probs against a known plate once and sanity-check this assumption
    before trusting avg_char_confidence in production — if it's wrong, the
    confidence numbers here are misleading, not just imprecise.
    """
    if _looks_two_line(cropped_plate):
        top, bottom = _split_two_line(cropped_plate)
        top_result = two_line_model.run(np.array(top.convert("RGB")), return_confidence=True)
        bottom_result = two_line_model.run(np.array(bottom.convert("RGB")), return_confidence=True)

        top_pred = top_result[0] if top_result else None
        bottom_pred = bottom_result[0] if bottom_result else None

        top_text = top_pred.plate if top_pred else ""
        bottom_text = bottom_pred.plate if bottom_pred else ""
        text = f"{top_text}{bottom_text}".strip()

        top_probs = list(top_pred.char_probs[:len(top_text)]) if top_pred is not None and top_pred.char_probs is not None else []
        bottom_probs = list(bottom_pred.char_probs[:len(bottom_text)]) if bottom_pred is not None and bottom_pred.char_probs is not None else []
        all_probs = top_probs + bottom_probs
    else:
        result = single_line_model.run(np.array(cropped_plate.convert("RGB")), return_confidence=True)
        pred = result[0] if result else None
        text = pred.plate if pred else ""
        all_probs = list(pred.char_probs[:len(text)]) if pred is not None and pred.char_probs is not None else []

    if not text or not all_probs:
        return {"text": None, "avg_char_confidence": 0.0, "is_valid": False, "reason": "no text detected"}

    avg_conf = float(sum(all_probs) / len(all_probs))
    is_valid, reason = validate_ocr_result(text, avg_conf)
    return {"text": text, "avg_char_confidence": avg_conf, "is_valid": is_valid, "reason": reason}


def recognize_plate(
    single_line_model: LicensePlateRecognizer,
    two_line_model: LicensePlateRecognizer,
    cropped_plate: Image.Image,
):
    """
    Runs OCR on a single cropped plate image and returns the best-guess plate text, 
    or None if nothing readable was found.
 
    Takes two separate model instances rather than one:
      - single_line_model: the fine-tuned model (load_finetuned_ocr_model()),
        used directly on plates that look single-line by aspect ratio.
      - two_line_model: the base pretrained model (load_ocr_model()), used on the
        split top/bottom halves of plates that look two-line.
 
    These are deliberately different models. The fine-tune set (516 labeled images,
    ~410 after train/val split) contained too few two-line examples to fine-tune on
    safely — fine-tuning on an overwhelmingly single-line dataset risks making the
    model MORE single-line-biased and actively worse at two-line recognition than
    the base pretrained model already was. So two-line plates deliberately skip the
    fine-tuned model entirely and stay on the untouched base model until a proper
    two-line-specific dataset justifies fine-tuning that path too.
 
    model.run() returns a list of PlatePrediction objects (even without
    return_confidence=True) — the actual text is in .plate, not the object itself.
    """
    if _looks_two_line(cropped_plate):
        top, bottom = _split_two_line(cropped_plate)
        top_result = two_line_model.run(np.array(top.convert("RGB")))
        bottom_result = two_line_model.run(np.array(bottom.convert("RGB")))
        top_text = top_result[0].plate if top_result else ""
        bottom_text = bottom_result[0].plate if bottom_result else ""
        combined = f"{top_text}{bottom_text}".strip()
        return combined or None
 
    result = single_line_model.run(np.array(cropped_plate.convert("RGB")))
    if not result:
        return None
    return result[0].plate