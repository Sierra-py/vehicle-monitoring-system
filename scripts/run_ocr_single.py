"""Run this file to test OCR on a single cropped plate image using a file dialog."""
from src.ui import run_ocr_on_upload
from src.ocr import load_ocr_model

reader = load_ocr_model()
run_ocr_on_upload(reader)