"""Run OCR over a folder of cropped plate images (e.g. output of batch_infer.py)."""
from pathlib import Path
from config.config import config
from src.ocr import load_ocr_model, load_finetuned_ocr_model, recognize_plate
from PIL import Image
import json

def run_ocr_on_crops(crops_dir: Path):
    single_line_model = load_finetuned_ocr_model()
    two_line_model = load_ocr_model()
    crop_paths = list(crops_dir.glob("*.jpg")) + list(crops_dir.glob("*.png"))
    print(f"Running OCR on {len(crop_paths)} cropped plates")

    results = {}
    for crop_path in crop_paths:
        img = Image.open(crop_path).convert("RGB")
        text = recognize_plate(single_line_model, two_line_model, img)
        results[crop_path.name] = text
        print(f"{crop_path.name}: {text}")

    return results

if __name__ == "__main__":
    result = run_ocr_on_crops(config.raw_data_dir / "Indian_LPR" / "images")
    output_path = config.processed_data_dir / "extracted_ocr_text.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\nResults written to {output_path}")