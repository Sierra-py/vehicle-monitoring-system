# scripts/inspect_misses.py
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from config.config import config

CONF_THRESHOLD = config.yolo_confidence_threshold       # threshold for "counts as a real detection" in the miss calculation
LOW_CONF_DISPLAY = 0.1                                  # show ALL predictions above this, even weak ones, for diagnosis
IOU_MATCH_THRESHOLD = 0.5

def load_gt_boxes(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.split())
            x1 = (xc - w/2) * img_w
            y1 = (yc - h/2) * img_h
            x2 = (xc + w/2) * img_w
            y2 = (yc + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes

def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def find_missed_detections(model_path, val_images_dir, val_labels_dir, output_dir):
    model = YOLO(str(model_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    miss_count = 0

    for img_path in val_images_dir.glob("*.jpg"):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        label_path = val_labels_dir / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(label_path, w, h)
        if not gt_boxes:
            continue

        # run at LOW threshold so we can see weak/near-miss predictions too
        results = model(str(img_path), conf=LOW_CONF_DISPLAY, verbose=False)
        all_preds = [(box.xyxy[0].tolist(), float(box.conf[0])) for box in results[0].boxes]

        # "counted as found" only uses predictions above the real threshold
        confident_preds = [p for p, c in all_preds if c >= CONF_THRESHOLD]

        missed = []
        for gt in gt_boxes:
            if not any(iou(gt, pred) >= IOU_MATCH_THRESHOLD for pred in confident_preds):
                missed.append(gt)

        if missed:
            miss_count += 1
            draw = ImageDraw.Draw(img)

            # ground truth misses in red
            for box in missed:
                draw.rectangle(box, outline="red", width=4)

            # ALL model predictions in green (even weak ones), labeled with confidence
            for box, conf in all_preds:
                color = "green" if conf >= CONF_THRESHOLD else "yellow"
                draw.rectangle(box, outline=color, width=2)
                draw.text((box[0], max(0, box[1] - 12)), f"{conf:.2f}", fill=color)

            img.save(output_dir / f"missed_{img_path.name}")

    print(f"Images with at least one missed plate: {miss_count}")
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    find_missed_detections(
        model_path=config.yolo_runs_dir / "yolo_pretrained_100ep" / "weights" / "best.pt",
        val_images_dir=config.plate_dataset_dir / "val" / "images",
        val_labels_dir=config.plate_dataset_dir / "val" / "labels",
        output_dir=config.processed_data_dir / "missed_detections"
    )