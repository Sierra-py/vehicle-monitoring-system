"""Script to run the inference on all test set"""
from config.config import config
from src.detection import run_inference_and_crop

if __name__ == "__main__":
    # Batch mode (original behavior)
    run_inference_and_crop(
        model_path=config.yolo100ep_best_weights,
        source_dir=config.plate_dataset_dir / "test" / "images",  # or your own held-out images
        annotated_dir=config.processed_data_dir / "ocr_test_annotated",
        crops_dir=config.processed_data_dir / "ocr_test_crops"
    )