"""Run this file to test OCR on a single cropped plate image using a file dialog."""
from src.ui import run_ocr_on_upload
from src.ocr import load_ocr_model, load_finetuned_ocr_model

single_line_model = load_finetuned_ocr_model()  # fine-tuned, ~93% on held-out Indian plates
two_line_model = load_ocr_model()                # base pretrained, used for two-line split path

run_ocr_on_upload(single_line_model, two_line_model)