from PIL import Image
import numpy as np
import easyocr

from config.config import config


def load_ocr_model():
    """
    Downloads the ocr model into the ocr/pretrained directory. Loads and returns an EasyOCR reader.
    Reuse the returned reader across images — loading it is expensive, and doing
    it per-image would repeat that cost for no benefit, same reasoning as loading the
    YOLO model once in detection.py.
    """
    config.ocr_pretrained_dir.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(
        config.ocr_languages,
        gpu= config.ocr_use_gpu,
        model_storage_directory=str(config.ocr_pretrained_dir),
    )



PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MIN_CONF_PER_LINE = 0.2       # discard low-confidence noise reads (stray marks, bolts, etc.)
UPSCALE_MIN_HEIGHT_PX = 64    # crops shorter than this get upscaled before OCR
 


def recognize_plate(reader, cropped_plate: Image.Image):
    """
    Runs OCR on a single cropped plate image (the kind returned by crop_detections
    in detection.py) and returns the best-guess plate text, or None if nothing
    readable was found.
 
    Handles two-line plates: EasyOCR returns one text region per detected line, so
    instead of keeping only the single highest-confidence region (which silently
    drops the second line on stacked plates), all regions above MIN_CONF_PER_LINE
    are kept and joined in top-to-bottom reading order.
    """
    img = cropped_plate
 
    # Small crops (typical once cropped tightly around just the plate) hurt OCR
    # accuracy — upscale before reading if the crop is short.
    if img.height < UPSCALE_MIN_HEIGHT_PX:
        scale = UPSCALE_MIN_HEIGHT_PX / img.height
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
 
    result = reader.readtext(np.array(img), allowlist=PLATE_ALLOWLIST)
    if not result:
        return None
 
    # Each item is (bbox, text, confidence). bbox is 4 corner points;
    # bbox[0][1] is the top-left corner's y-coordinate, used to sort lines
    # top-to-bottom rather than in whatever order EasyOCR happened to return them.
    lines = [(bbox[0][1], text, conf) for bbox, text, conf in result if conf >= MIN_CONF_PER_LINE]
    if not lines:
        return None
 
    lines.sort(key=lambda l: l[0])
    return "".join(text for _, text, _ in lines)
