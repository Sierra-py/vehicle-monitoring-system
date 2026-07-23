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
 
