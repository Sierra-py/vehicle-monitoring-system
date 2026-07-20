"""
Fine-tunes the pretrained fast-plate-ocr model on your labeled Indian plate dataset.

fast-plate-ocr's training is a CLI tool (built on Keras 3), separate from the ONNX
inference path used in src/ocr.py — this script just builds the right command and
runs it via subprocess, rather than reimplementing training logic. See:
https://ankandrew.github.io/fast-plate-ocr/latest/training/cli/train/

PREREQUISITES — do these once, manually, before running this script:
  1. pip install "fast-plate-ocr[train]"
  2. Set a Keras backend, e.g. (PowerShell): $env:KERAS_BACKEND = "tensorflow"
  3. Download the .keras (not .onnx) version of your base model from the
     fast-plate-ocr GitHub release assets (release tag: arg-plates) and place it at
     the path config.ocr_finetune_base_weights points to.
  4. Run scripts/prepare_ocr_finetune_data.py first to produce the train/val CSVs
     this script expects at config.ocr_finetune_dataset_dir.

This fine-tunes on SINGLE-LINE plates only (see prepare_ocr_finetune_data.py for
why two-line examples are excluded) — the two-line split+recognize path in
src/ocr.py is unaffected by this fine-tune unless you explicitly point recognize_plate
at the fine-tuned weights for that path too, which is not done automatically.

Usage:
    python scripts/finetune_ocr.py
    python scripts/finetune_ocr.py --epochs 20 --batch-size 16   # override defaults
"""
import argparse
import shutil
import subprocess
import sys

from config.config import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.ocr_finetune_epochs)
    parser.add_argument("--batch-size", type=int, default=config.ocr_finetune_batch_size)
    parser.add_argument("--model-config", type=str, default=None,
                         help="Path to model_config.yaml. If omitted, you must supply one — "
                              "see fast-plate-ocr repo's models/ folder for examples matching "
                              "your base model architecture (cct-s-v2 etc).")
    parser.add_argument("--plate-config", type=str, default=None,
                         help="Path to plate_config.yaml defining the character set. "
                              "Use the repo's default latin/global config unless Indian "
                              "plates need characters outside it (they generally don't — "
                              "uppercase A-Z, 0-9).")
    args = parser.parse_args()

    if shutil.which("fast-plate-ocr") is None:
        print("fast-plate-ocr CLI not found. Install training extras first:")
        print('  pip install "fast-plate-ocr[train]"')
        sys.exit(1)

    if not config.ocr_finetune_base_weights.exists():
        print(f"Base .keras weights not found at {config.ocr_finetune_base_weights}")
        print("Download the .keras version of your base model from the fast-plate-ocr")
        print("GitHub release assets (tag: arg-plates) and place it at that path.")
        sys.exit(1)

    train_csv = config.ocr_finetune_dataset_dir / "train" / "annotations.csv"
    val_csv = config.ocr_finetune_dataset_dir / "val" / "annotations.csv"
    if not train_csv.exists() or not val_csv.exists():
        print(f"Expected train/val CSVs not found under {config.ocr_finetune_dataset_dir}")
        print("Run scripts/prepare_ocr_finetune_data.py first.")
        sys.exit(1)

    if args.model_config is None or args.plate_config is None:
        print("--model-config and --plate-config are required.")
        print("These come from the fast-plate-ocr repo itself — clone it and point at:")
        print("  models/<architecture>.yaml         (matching your base model)")
        print("  config/<language>_plates.yaml       (e.g. a latin/global plate config)")
        sys.exit(1)

    config.ocr_finetune_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "fast-plate-ocr", "train",
        "--model-config-file", args.model_config,
        "--plate-config-file", args.plate_config,
        "--annotations", str(train_csv),
        "--val-annotations", str(val_csv),
        "--weights-path", str(config.ocr_finetune_base_weights),  # this makes it a fine-tune, not from-scratch
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--output-dir", str(config.ocr_finetune_output_dir),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"\nDone. Fine-tuned weights and logs are in {config.ocr_finetune_output_dir}")
    print("Next: evaluate on your held-out val set before swapping this into src/ocr.py.")


if __name__ == "__main__":
    main()