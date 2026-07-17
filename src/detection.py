from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw
from config.config import config


def crop_detections(model: YOLO, img_path: Path, conf_threshold: float = config.yolo_confidence_threshold):
    """
    Run inference on a single image. Returns a tuple:
      - crops: list of (cropped_image, confidence) tuples for detections above conf_threshold
      - annotated: a copy of the full original image with a bounding box and confidence
        label drawn for each detection above conf_threshold

    Shared by both batch and single-image flows so the detection/crop logic only lives
    in one place.
    """
    results = model(str(img_path), verbose=False)
    img = Image.open(img_path).convert("RGB")

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    BOX_COLOR = "red"
    BOX_WIDTH = 3
    LABEL_TEXT_COLOR = "white"
    LABEL_BG_COLOR = "red"

    crops = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cropped = img.crop((x1, y1, x2, y2))
            crops.append((cropped, conf))

            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)

            label = f"{conf:.2f}"
            # textbbox gives the rendered size of the label at (0, 0); used to size a
            # solid background box behind the text so it stays readable over busy
            # image content instead of floating as bare, hard-to-read text.
            text_bbox = draw.textbbox((0, 0), label)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            label_bg = [x1, max(0, y1 - text_h - 4), x1 + text_w + 4, y1]
            draw.rectangle(label_bg, fill=LABEL_BG_COLOR)
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
