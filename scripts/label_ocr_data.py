"""
Interactive OCR ground-truth labeling tool.

Run against any folder of cropped plate images. Each run's labels are stored
under data/processed/ocr_labels/<input_folder_name>/ocr_labels.csv, keyed by
the input folder's own name — so labeling multiple batches (e.g. as new data
comes in) never overwrites a previous batch's labels, and each batch's
progress/resume state stays independent.

Usage:
    # explicit paths via CLI
    python scripts/label_ocr_data.py --input data/raw/Indian_LPR_cleaned

    # optionally override the output folder name (defaults to the input folder's name)
    python scripts/label_ocr_data.py --input data/raw/batch_2 --name batch_2_oct

    # no arguments: falls back to a folder-select dialog
    python scripts/label_ocr_data.py
"""
import argparse
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from config.config import config
from src.ocr import load_ocr_model, load_finetuned_ocr_model
from src.ui import label_ocr_predictions


def prompt_for_input_dir() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(title="Select folder of cropped plate images to label")
    root.destroy()
    return Path(selected) if selected else None


def main():
    parser = argparse.ArgumentParser(description="Label cropped plate images with OCR-assisted correction.")
    parser.add_argument("--input", type=str, default=None,
                         help="Path to folder of cropped plate images. Opens a folder-picker dialog if omitted.")
    parser.add_argument("--name", type=str, default=None,
                         help="Name for this batch's output folder under data/processed/ocr_labels/. "
                              "Defaults to the input folder's own name.")
    parser.add_argument("--use-finetuned", action="store_true",
                         help="Draft predictions using the fine-tuned model instead of the base "
                              "pretrained model. Gives better starting drafts (less typing) once a "
                              "fine-tuned model exists — but note the fine-tuned model was trained "
                              "on single-line plates only, so its drafts on two-line crops may be "
                              "less reliable than the base model's for this labeling purpose.")
    args = parser.parse_args()

    input_dir = Path(args.input) if args.input else prompt_for_input_dir()

    if input_dir is None:
        print("No input folder selected. Exiting.")
        return
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {input_dir}")

    batch_name = args.name or input_dir.name
    output_dir = config.processed_data_dir / "ocr_labels" / batch_name
    output_csv = output_dir / "ocr_labels.csv"

    print(f"Input:  {input_dir}")
    print(f"Output: {output_csv}")

    ocr_model = load_finetuned_ocr_model() if args.use_finetuned else load_ocr_model()
    label_ocr_predictions(ocr_model=ocr_model, crops_dir=input_dir, output_csv=output_csv)


if __name__ == "__main__":
    main()