"""Run this to find the missed detections on val set by the yolo model after training"""
from config.config import config
from src.metrics import find_missed_detections


if __name__ == "__main__":
    find_missed_detections(
        model_path=config.yolo100ep_best_weights,
        val_images_dir=config.plate_dataset_dir / "val" / "images",
        val_labels_dir=config.plate_dataset_dir / "val" / "labels",
        output_dir=config.processed_data_dir / "missed_detections"
    )