"""Script to run the YOLO model inference on all test set"""
from config.config import config
from src.detection import run_inference_and_crop
import argparse
from pathlib import Path



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default=config.plate_dataset_dir / "test" / "images", 
                        help = "Write the path to input data folder.")
    parser.add_argument("--anno", type=str, default=config.processed_data_dir / "ocr_test_annotated", 
                        help = "Write the path where to annotated cropped images.")
    parser.add_argument("--crop", type=str, default=config.processed_data_dir / "ocr_test_crops", 
                        help = "Write the path where to save cropped images.")

    args = parser.parse_args()

    source_dir = Path(args.input)
    annotated_dir = Path(args.anno)
    crops_dir = Path(args.crop)


    run_inference_and_crop(
        model_path=config.yolo100ep_best_weights,
        source_dir=source_dir,  
        annotated_dir=annotated_dir,
        crops_dir=crops_dir
    )