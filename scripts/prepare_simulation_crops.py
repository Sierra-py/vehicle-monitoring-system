from config.config import config
from src.detection import run_inference_and_crop, select_labeling_candidates
from src.ocr import load_finetuned_ocr_model
from src.ui import label_ocr_predictions


def main():
    source_dir = config.simulation_data_dir / "images"
    annotated_dir = config.simulation_data_dir / "annotated"
    crops_dir = config.simulation_data_dir / "crops"
    candidates_dir = config.simulation_data_dir / "labeling_candidates"
    output_csv = config.simulation_data_dir / "whitelist_labels.csv"

    if not any(source_dir.glob("*.jpg")) and not any(source_dir.glob("*.png")):
        print(f"No images found in {source_dir} - add simulations images first.")
        return 

    print(f"Running YOLO detection on image in {source_dir}...")
    run_inference_and_crop(
        model_path=config.yolo100ep_best_weights,
        source_dir=source_dir,
        annotated_dir=annotated_dir,
        crops_dir=crops_dir,
    )

    print(f"Selecting one best candidate crop per source image...")

    candidates = select_labeling_candidates(crops_dir)

    candidates_dir.mkdir(parents=True, exist_ok=True)
    for stem, path in candidates.items():
        dest = candidates_dir / path.name
        if not dest.exists():
            dest.write_bytes(path.read_bytes())

    print(f"{len(candidates)} candidate images ready for review in {candidates_dir}")
    print("Opening labeling UI - type the TRUE plate text for each image, or accept the prediction if correct.")

    ocr_model = load_finetuned_ocr_model()
    label_ocr_predictions(ocr_model, candidates_dir, output_csv)

    print(f"Labels saved to {output_csv}")
    print("Next: run scripts/build_whitelist.py to pick which of these become the whitelist.")


if __name__ == "__main__":
    main()