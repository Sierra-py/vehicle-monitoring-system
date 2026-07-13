import argparse
from ultralytics import YOLO
from config.config import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to run folder to resume, e.g. train-4")
    args = parser.parse_args()

    if args.resume:
        checkpoint = config.yolo_runs_dir / args.resume / "weights" / "last.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint}")
        model = YOLO(str(checkpoint))
        model.train(resume=True)
    else:
        model = YOLO(str(config.yolo_pretrained_dir / config.yolo_model_variant))
        model.train(
            data=str(config.plate_data_yaml),
            epochs=config.yolo_epochs,
            imgsz=config.yolo_imgsz,
            batch=config.yolo_batch,
            patience=15,
            project=str(config.yolo_runs_dir),
            name="yolo_pretrained_100ep"
        )

if __name__ == "__main__":
    main()