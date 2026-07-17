from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    # --- Base paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"

    # --- Dataset: plate detection ---
    plate_ndjson: Path = raw_data_dir / "plate.ndjson"
    plate_dataset_dir: Path = raw_data_dir / "plate"      # images/labels land here
    plate_data_yaml: Path = plate_dataset_dir / "data.yaml"

    # --- YOLO model (namespaced, self-contained) ---
    models_dir: Path = project_root / "models"
    yolo_dir: Path = models_dir / "yolo"
    yolo_pretrained_dir: Path = yolo_dir / "pretrained"
    yolo_runs_dir: Path = yolo_dir / "runs"
    yolo100ep_best_weights: Path = yolo_runs_dir / "yolo_pretrained_100ep" / "weights" / "best.pt"

    yolo_model_variant: str = "yolo26n.pt"
    yolo_epochs: int = 100
    yolo_imgsz: int = 640
    yolo_batch: int = 32
    yolo_confidence_threshold: float = 0.5

    # --- OCR (plate text recognition) ---
    ocr_dir: Path = models_dir/ "ocr"
    ocr_pretrained_dir: Path = ocr_dir / "pretrained"
    ocr_languages: list[str] = ["en"]
    ocr_use_gpu: bool = True

    # --- Secrets ---
    # postgres_url: str = Field(...)
    # redis_url: str = Field(default="redis://localhost:6379")

    class Config:
        env_file = ".env"

config = Config()