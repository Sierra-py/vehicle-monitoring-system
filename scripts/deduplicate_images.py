"""Removes exact duplicate images from the input directory and saves them in a output directory.

Usage:  IndianLPR has duplicate image of every image making the data bloated. 
        This removes the duplicate and saves unique images in the output folder.
"""
import hashlib
from pathlib import Path
import argparse

def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def dedupe(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes = {}
    duplicates = 0

    for img_path in sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))):
        h = file_hash(img_path)
        if h in seen_hashes:
            duplicates += 1
            continue
        seen_hashes[h] = img_path.name
        (output_dir / img_path.name).write_bytes(img_path.read_bytes())

    print(f"Kept {len(seen_hashes)} unique images, removed {duplicates} exact duplicates")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dedupe(Path(args.input), Path(args.output))