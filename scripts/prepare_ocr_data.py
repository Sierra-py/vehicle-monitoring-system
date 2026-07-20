"""
Converts a labeled ocr_labels.csv (produced by scripts/label_ocr_data.py) into the
train/val CSV format fast-plate-ocr's training CLI expects:

    dataset/
    ├── train/
    │   ├── annotations.csv   (image_path,plate_text)
    │   └── images/
    └── val/
        ├── annotations.csv
        └── images/

Deliberately excludes anything marked UNREADABLE (from the labeling tool's Esc/skip
action) and anything flagged two-line by the same aspect-ratio heuristic used at
inference time (src.ocr._looks_two_line). This is intentional, not an oversight:
fine-tuning on full single-line crops only avoids skewing the model further away
from the already-thin two-line data, per the two-line handling discussion — the
two-line path stays on split+recognize with the un-fine-tuned model behavior for
now, rather than being folded into a fine-tune that would mostly consist of
single-line examples and risk degrading further.

Usage:
    python scripts/prepare_ocr_finetune_data.py --labels data/processed/ocr_labels/Indian_LPR_deduped/ocr_labels.csv --images data/raw/Indian_LPR_deduped
"""
import argparse
import csv
import shutil
from pathlib import Path
import random

from PIL import Image
from config.config import config
from src.ocr import _looks_two_line

VAL_FRACTION = 0.2
RANDOM_SEED = 42


def load_clean_rows(labels_csv: Path, images_dir: Path):
    kept, skipped_unreadable, skipped_two_line, skipped_missing = [], 0, 0, 0

    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            text = row["corrected_text"].strip()

            if text == "UNREADABLE" or not text:
                skipped_unreadable += 1
                continue

            img_path = images_dir / row["filename"]
            if not img_path.exists():
                skipped_missing += 1
                continue

            with Image.open(img_path) as img:
                if _looks_two_line(img):
                    skipped_two_line += 1
                    continue

            kept.append((img_path, text))

    print(f"Kept: {len(kept)}")
    print(f"Skipped — unreadable/empty: {skipped_unreadable}")
    print(f"Skipped — two-line (excluded from fine-tune set): {skipped_two_line}")
    print(f"Skipped — image file missing: {skipped_missing}")
    return kept


def write_split(rows, split_dir: Path):
    images_out = split_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    with open(split_dir / "annotations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "plate_text"])
        for img_path, text in rows:
            dest_name = img_path.name
            shutil.copy(img_path, images_out / dest_name)
            # paths in the CSV are relative to the CSV's own location, per fast-plate-ocr docs
            writer.writerow([f"images/{dest_name}", text])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to ocr_labels.csv")
    parser.add_argument("--images", required=True, help="Folder containing the original cropped images referenced in the labels CSV")
    parser.add_argument("--out", default=None, help="Output dataset folder (default: data/processed/ocr_finetune_dataset)")
    args = parser.parse_args()

    labels_csv = Path(args.labels)
    images_dir = Path(args.images)
    out_dir = Path(args.out) if args.out else config.processed_data_dir / "ocr_finetune_dataset"

    rows = load_clean_rows(labels_csv, images_dir)
    if len(rows) < 20:
        raise ValueError(f"Only {len(rows)} usable rows after filtering — too few to split/fine-tune on meaningfully.")

    random.seed(RANDOM_SEED)
    random.shuffle(rows)

    val_count = max(1, int(len(rows) * VAL_FRACTION))
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]

    write_split(train_rows, out_dir / "train")
    write_split(val_rows, out_dir / "val")

    print(f"\nTrain: {len(train_rows)}  Val: {len(val_rows)}")
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()