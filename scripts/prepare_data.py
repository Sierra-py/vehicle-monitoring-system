"""Script to download data from plate ndjson file"""
import json
import requests
from pathlib import Path
from config.config import config

def download_plate_dataset():
    config.plate_dataset_dir.mkdir(parents=True, exist_ok=True)

    with open(config.plate_ndjson) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "image":
                continue

            split = rec["split"]  # train / valid / test
            img_dir = config.plate_dataset_dir / split / "images"
            label_dir = config.plate_dataset_dir / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            img_path = img_dir / rec["file"]
            if not img_path.exists():  # skip re-download if already present
                resp = requests.get(rec["url"])
                img_path.write_bytes(resp.content)

            label_path = label_dir / (Path(rec["file"]).stem + ".txt")
            boxes = rec.get("annotations", {}).get("boxes", [])
            with open(label_path, "w") as lf:
                for box in boxes:
                    lf.write(" ".join(map(str, box)) + "\n")

    print(f"Dataset downloaded to {config.plate_dataset_dir}")

def write_data_yaml():
    content = f"""path: {config.plate_dataset_dir.as_posix()}
train: train/images
val: val/images
test: test/images
names:
  0: license_plate
"""
    config.plate_data_yaml.write_text(content)
    print(f"data.yaml written to {config.plate_data_yaml}")

if __name__ == "__main__":
    download_plate_dataset()
    write_data_yaml()