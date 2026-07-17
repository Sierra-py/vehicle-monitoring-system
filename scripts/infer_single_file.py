"""Run this file to infer a single file using dialog box"""
from scripts.infer import run_inference_on_upload
from config.config import config
run_inference_on_upload(config.yolo100ep_best_weights, output_dir=config.yolo_runs_dir)