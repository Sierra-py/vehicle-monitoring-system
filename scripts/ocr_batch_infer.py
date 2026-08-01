"""Run OCR over a folder of cropped plate images (e.g. output of batch_infer.py)."""
from pathlib import Path
from config.config import config
from src.ocr import load_ocr_model, load_finetuned_ocr_model, recognize_plate_with_validation
from PIL import Image
import json
import argparse

def run_ocr_on_crops(crops_dir: Path, use_finetuned: bool = False):
    if use_finetuned:
        single_line_model = load_finetuned_ocr_model()
    else:
        single_line_model = load_ocr_model()
    two_line_model = load_ocr_model()
    crop_paths = list(crops_dir.glob("*.jpg")) + list(crops_dir.glob("*.png"))
    print(f"Running OCR on {len(crop_paths)} cropped plates")

    results = {}
    flagged_count = 0
    for crop_path in crop_paths:
        img = Image.open(crop_path).convert("RGB")
        outcome = recognize_plate_with_validation(single_line_model, two_line_model, img)
        results[crop_path.name] = outcome
        if not outcome["is_valid"]:
            flagged_count += 1
        status = "OK" if outcome["is_valid"] else f"FLAGGED ({outcome['reason']})"
        print(f"{crop_path.name}: {outcome['text']}  [{status}]")

    print(f"\n{flagged_count} / {len(crop_paths)} flagged for review")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run OCR inference in batch of images.")
    parser.add_argument("--input", type=str, default=config.processed_data_dir / "Indian_LPR_deduped",
                         help="Path to folder of cropped plate images. Opens a folder-picker dialog if omitted.")
    parser.add_argument("--output", type=str, default="extracted_ocr_text",
                         help="Path to save the json file of this batch.")
    parser.add_argument("--use_finetuned", type=str, default=True,
                        help = "Use the pretrained model for single line plates")
    args = parser.parse_args()

    crops_dir = Path(args.input)
    use_finetuned = bool(args.use_finetuned)
    result = run_ocr_on_crops(crops_dir, use_finetuned)
    output_path = Path(args.output + ".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\nResults written to {output_path}")
if __name__ == "__main__":
    main()