from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw
from config.config import config


def _overlap_coefficient(box1, box2):
    """
    Intersection area divided by the SMALLER box's area (not union, unlike standard IoU).
    Stays high when one box is mostly contained within another regardless of how
    different their sizes are — standard IoU drops fast when boxes differ a lot in
    size even if they cover the same object, which is exactly the case this exists
    to catch (a tight plate box vs. a looser box that also includes surrounding
    bumper/frame, both detecting the same physical plate).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    smaller = min(area1, area2)
    return inter / smaller if smaller > 0 else 0


def _dedupe_overlapping_boxes(boxes_with_conf, overlap_threshold=config.yolo_dedup_overlap_threshold):
    """
    Second-pass dedup on top of ultralytics' built-in NMS. Standard NMS uses IoU
    (intersection/union), which can fail to merge two boxes covering the same object
    when they're very different sizes — see _overlap_coefficient. This pass greedily
    keeps the highest-confidence box in each overlapping cluster and discards the rest.

    boxes_with_conf: list of (box_xyxy, conf) tuples. Returns the same shape, deduped.
    """
    sorted_boxes = sorted(boxes_with_conf, key=lambda bc: bc[1], reverse=True)
    kept = []
    for box, conf in sorted_boxes:
        if not any(_overlap_coefficient(box, kept_box) >= overlap_threshold for kept_box, _ in kept):
            kept.append((box, conf))
    return kept


def crop_detections(
    model: YOLO,
    img_path: Path,
    conf_threshold: float = config.yolo_confidence_threshold,
    iou: float = config.yolo_nms_iou,
    min_crop_height_px: int = config.yolo_min_crop_height_px,
    dedup_overlap_threshold: float = config.yolo_dedup_overlap_threshold):
    """
    Run inference on a single image. Returns a tuple:
      - crops: list of (cropped_image, confidence) tuples for detections above conf_threshold
        AND at least min_crop_height_px tall, after duplicate-box removal
      - annotated: a copy of the full original image with a bounding box and confidence
        label drawn for each surviving detection (including undersized ones, for debugging
        visibility — the size filter only affects what reaches OCR)

    iou controls ultralytics' own NMS for this call — a first pass. dedup_overlap_threshold
    controls a SECOND pass (_dedupe_overlapping_boxes) that catches same-plate duplicate
    boxes standard IoU-based NMS misses when the boxes are different sizes (see
    _overlap_coefficient's docstring). Both exist because lowering iou alone has a limit:
    push it too low and it starts merging genuinely separate nearby objects instead of
    just duplicates.

    min_crop_height_px filters out detections too small to ever be OCR-readable regardless
    of how confident YOLO is that something plate-shaped is there — a small, far-away plate
    can have high detection confidence while still being physically too few pixels tall for
    the OCR model to read. This is a separate axis from conf_threshold for the same reason
    dedup is: it's a "can this possibly be used" filter, not a "how sure is YOLO" filter.

    Shared by both batch and single-image flows so the detection/crop logic only lives
    in one place.
    """
    results = model(str(img_path), iou=iou, verbose=False)
    img = Image.open(img_path).convert("RGB")

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    BOX_COLOR = "red"
    UNDERSIZED_BOX_COLOR = "orange"  # detected + confident, but too small to reach OCR
    BOX_WIDTH = 3
    LABEL_TEXT_COLOR = "white"

    # Collect all raw candidate boxes above conf_threshold first, across all results —
    # dedup needs to compare boxes against each other, so it can't happen inline in a
    # single pass over r.boxes the way drawing/cropping used to.
    raw_boxes = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            raw_boxes.append((box.xyxy[0].tolist(), conf))

    deduped_boxes = _dedupe_overlapping_boxes(raw_boxes, dedup_overlap_threshold)

    crops = []
    for (x1, y1, x2, y2), conf in deduped_boxes:
        too_small = (y2 - y1) < min_crop_height_px
        box_color = UNDERSIZED_BOX_COLOR if too_small else BOX_COLOR

        if not too_small:
            cropped = img.crop((x1, y1, x2, y2))
            crops.append((cropped, conf))

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=BOX_WIDTH)

        label = f"{conf:.2f}"
        # textbbox gives the rendered size of the label at (0, 0); used to size a
        # solid background box behind the text so it stays readable over busy
        # image content instead of floating as bare, hard-to-read text.
        text_bbox = draw.textbbox((0, 0), label)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        label_bg = [x1, max(0, y1 - text_h - 4), x1 + text_w + 4, y1]
        draw.rectangle(label_bg, fill=box_color)
        draw.text((x1 + 2, max(0, y1 - text_h - 2)), label, fill=LABEL_TEXT_COLOR)

    return crops, annotated

# ---------------------------------------------------------------------------

def run_inference_and_crop(model_path: Path, source_dir: Path, annotated_dir: Path, crops_dir: Path):
    model = YOLO(str(model_path))

    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    print(f"Running inference on {len(image_paths)} images")

    for img_path in image_paths:
        crops, annotated = crop_detections(model, img_path)

        annotated.save(annotated_dir / f"{img_path.stem}_annotated.jpg")

        for i, (cropped, conf) in enumerate(crops):
            crop_filename = f"{img_path.stem}_plate{i}_conf{conf:.2f}.jpg"
            cropped.save(crops_dir / crop_filename)

    print(f"Annotated images saved to {annotated_dir}")
    print(f"Cropped plates saved to {crops_dir}")

#-------------------------------------------------------------------------------------

import re

# Matches the filename convention crops are saved with elsewhere in this module:
# {source_image_stem}_plate{index}_conf{confidence}.jpg
CROP_FILENAME_RE = re.compile(r"^(?P<source_stem>.+)_plate(?P<idx>\d+)_conf(?P<conf>[\d.]+)$")

# Rejects crops with a shape no real plate would plausibly have (extreme slivers,
# near-square blobs) — these are almost always YOLO false positives, not real plates
# of either layout. Deliberately WIDER than TWO_LINE_ASPECT_THRESHOLD in src/ocr.py:
# this filter's job is "could this possibly be a plate at all", not "is this single-
# or two-line" — that routing decision happens separately, later, at inference time.
MIN_PLAUSIBLE_ASPECT_RATIO = 0.8
MAX_PLAUSIBLE_ASPECT_RATIO = 6.0


def _parse_crop_filename(path: Path):
    """Returns (source_stem, confidence) for a crop filename, or None if it doesn't match the expected pattern."""
    match = CROP_FILENAME_RE.match(path.stem)
    if not match:
        return None
    return match.group("source_stem"), float(match.group("conf"))


def _passes_aspect_ratio_filter(img: Image.Image) -> bool:
    ratio = img.width / img.height
    return MIN_PLAUSIBLE_ASPECT_RATIO <= ratio <= MAX_PLAUSIBLE_ASPECT_RATIO


def select_labeling_candidates(crops_dir: Path):
    """
    From a folder of crops produced by run_inference_and_crop (named
    {source_stem}_plate{i}_conf{conf}.jpg), returns {source_stem: crop_path} —
    one candidate per source image, picked as the highest-confidence crop among
    that image's detections that also passes the aspect-ratio plausibility filter.

    This exists to cut down what needs manual labeling: a single source image can
    produce multiple overlapping/duplicate detections of the same plate, and some
    detections are outright false positives with implausible shapes. Neither is
    worth spending labeling time reviewing, so both get filtered out before the
    labeling tool ever shows them.

    Crops whose filename doesn't match the expected naming convention are skipped
    and counted separately — this only works on output from run_inference_and_crop,
    not arbitrary image folders.
    """
    best_by_source = {}  # source_stem -> (crop_path, confidence)
    skipped_unparseable = 0
    skipped_aspect_ratio = 0

    crop_paths = list(crops_dir.glob("*.jpg")) + list(crops_dir.glob("*.png"))
    for crop_path in crop_paths:
        parsed = _parse_crop_filename(crop_path)
        if parsed is None:
            skipped_unparseable += 1
            continue
        source_stem, conf = parsed

        with Image.open(crop_path) as img:
            if not _passes_aspect_ratio_filter(img):
                skipped_aspect_ratio += 1
                continue

        current_best = best_by_source.get(source_stem)
        if current_best is None or conf > current_best[1]:
            best_by_source[source_stem] = (crop_path, conf)

    print(f"Candidates selected: {len(best_by_source)}")
    print(f"Skipped — unparseable filename: {skipped_unparseable}")
    print(f"Skipped — failed aspect-ratio filter: {skipped_aspect_ratio}")

    return {stem: path for stem, (path, _conf) in best_by_source.items()}