"""Run OCR over a folder of cropped plate images (e.g. output of batch_infer.py)."""
from pathlib import Path
from config.config import config
from src.ocr import load_ocr_model, recognize_plate
from PIL import Image


def run_ocr_on_crops(crops_dir: Path):
    reader = load_ocr_model()
    crop_paths = list(crops_dir.glob("*.jpg")) + list(crops_dir.glob("*.png"))
    print(f"Running OCR on {len(crop_paths)} cropped plates")

    results = {}
    for crop_path in crop_paths:
        img = Image.open(crop_path).convert("RGB")
        text = recognize_plate(reader, img)
        results[crop_path.name] = text
        print(f"{crop_path.name}: {text}")

    return results


if __name__ == "__main__":
    run_ocr_on_crops(config.processed_data_dir / "ocr_test_crops")