"""Run this file to infer a single file using dialog box"""
from scripts.infer import run_inference_on_upload
from config.config import config
run_inference_on_upload(config.yolo_runs_dir/"yolo_pretrained_100ep"/"weights"/"best.pt", output_dir=config.yolo_runs_dir)