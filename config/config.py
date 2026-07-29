from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
class Config(BaseSettings):
    # --- Base paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    simulation_data_dir: Path = data_dir / "simulation"
    cropped_img_dir: Path = processed_data_dir / "ocr_test_crops"

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
    yolo_confidence_threshold: float = 0.3

    # --- OCR (plate text recognition) ---
    ocr_dir: Path = models_dir/ "ocr"
    ocr_pretrained_dir: Path = ocr_dir / "pretrained"
    ocr_model_name: str = "cct-s-v2-global-model"
    ocr_languages: list[str] = ["en"]
    ocr_use_gpu: bool = True

    ocr_finetune_base_weights: Path = ocr_pretrained_dir / ocr_model_name /"cct_s_v2_global.keras"
    ocr_finetune_model_config: Path = ocr_pretrained_dir / ocr_model_name /"cct_s_v2_global_model_config.yaml"
    ocr_finetune_plate_config: Path = ocr_pretrained_dir / ocr_model_name/ "cct_s_v2_global_plate_config.yaml"
    ocr_finetune_dataset_dir: Path = processed_data_dir / "ocr_finetune_dataset"
    ocr_finetune_output_dir: Path = ocr_dir / "finetuned"
    ocr_finetune_epochs: int = 30
    ocr_finetune_batch_size: int = 32
    ocr_finetuned_model_onnx: Path = ocr_dir / "finetuned" / "best.onnx"


    # --- Secrets ---
    # postgres_url: str = Field(...)
    # redis_url: str = Field(default="redis://localhost:6379")

    class Config:
        env_file = ".env"

config = Config()