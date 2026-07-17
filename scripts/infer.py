# scripts/infer_and_crop.py
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw
from config.config import config


def crop_detections(model: YOLO, img_path: Path, conf_threshold: float = config.yolo_confidence_threshold):
    """
    Run inference on a single image. Returns a tuple:
      - crops: list of (cropped_image, confidence) tuples for detections above conf_threshold
      - annotated: a copy of the full original image with a bounding box and confidence
        label drawn for each detection above conf_threshold

    Shared by both batch and single-image flows so the detection/crop logic only lives
    in one place.
    """
    results = model(str(img_path), verbose=False)
    img = Image.open(img_path).convert("RGB")

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    BOX_COLOR = "red"
    BOX_WIDTH = 3
    LABEL_TEXT_COLOR = "white"
    LABEL_BG_COLOR = "red"

    crops = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cropped = img.crop((x1, y1, x2, y2))
            crops.append((cropped, conf))

            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)

            label = f"{conf:.2f}"
            # textbbox gives the rendered size of the label at (0, 0); used to size a
            # solid background box behind the text so it stays readable over busy
            # image content instead of floating as bare, hard-to-read text.
            text_bbox = draw.textbbox((0, 0), label)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            label_bg = [x1, max(0, y1 - text_h - 4), x1 + text_w + 4, y1]
            draw.rectangle(label_bg, fill=LABEL_BG_COLOR)
            draw.text((x1 + 2, max(0, y1 - text_h - 2)), label, fill=LABEL_TEXT_COLOR)

    return crops, annotated


def run_inference_and_crop(model_path: Path, source_dir: Path, annotated_dir: Path, crops_dir: Path):
    model = YOLO(str(model_path))

    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    print(f"Running inference on {len(image_paths)} images")

    for img_path in image_paths:
        crops, annotated = crop_detections(model, img_path)

        annotated.save(annotated_dir / f"{img_path.stem}_annotated.jpg")

        for i, (cropped, conf) in enumerate(crops):
            crop_filename = f"{img_path.stem}_plate{i}_conf{conf:.2f}.jpg"
            cropped.save(crops_dir / crop_filename)

    print(f"Annotated images saved to {annotated_dir}")
    print(f"Cropped plates saved to {crops_dir}")


def _show_crops_with_save(crops, annotated, img_stem: str, default_dir: Path):
    """
    Displays the full annotated image at the top (boxes + confidence labels drawn on),
    followed by each individual crop in its own row with a Save button. Saving is
    per-crop and opens a native save-as dialog seeded with a sensible default
    filename/location — nothing is written to disk unless the user clicks Save.
    """
    import tkinter as tk
    from tkinter import filedialog
    from PIL import ImageTk

    THUMB_MAX_SIDE = 300          # crop thumbnail display size only
    ANNOTATED_MAX_SIDE = 500      # annotated overview display size only

    window = tk.Tk()
    window.title(f"Detections — {img_stem}")

    # Keep references to PhotoImage objects alive for the life of the window,
    # otherwise tkinter garbage-collects them and the images go blank.
    window._photo_refs = []

    # --- Annotated overview at the top ---
    overview_frame = tk.Frame(window, padx=10, pady=10)
    overview_frame.pack(fill="x")

    overview_thumb = annotated.copy()
    overview_thumb.thumbnail((ANNOTATED_MAX_SIDE, ANNOTATED_MAX_SIDE))
    overview_photo = ImageTk.PhotoImage(overview_thumb)
    window._photo_refs.append(overview_photo)

    tk.Label(overview_frame, text="Detections overview", font=("Segoe UI", 10, "bold")).pack(anchor="w")
    tk.Label(overview_frame, image=overview_photo).pack(anchor="w")

    tk.Frame(window, height=2, bd=1, relief="sunken").pack(fill="x", padx=10, pady=5)

    # --- Individual crops below, each with its own Save button ---
    for i, (cropped, conf) in enumerate(crops):
        row = tk.Frame(window, padx=10, pady=10)
        row.pack(fill="x")

        thumb = cropped.copy()
        thumb.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE))
        photo = ImageTk.PhotoImage(thumb)
        window._photo_refs.append(photo)

        tk.Label(row, image=photo).pack(side="left", padx=(0, 10))
        tk.Label(row, text=f"Detection {i}\nconfidence: {conf:.2f}").pack(side="left", padx=(0, 10))

        def make_save_handler(image_to_save=cropped, index=i):
            def handler():
                default_name = f"{img_stem}_plate{index}_conf{conf:.2f}.jpg"
                save_path = filedialog.asksaveasfilename(
                    title="Save cropped plate",
                    initialdir=str(default_dir),
                    initialfile=default_name,
                    defaultextension=".jpg",
                    filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")],
                )
                if save_path:
                    image_to_save.save(save_path)
                    print(f"Saved: {save_path}")
            return handler

        tk.Button(row, text="Save", command=make_save_handler()).pack(side="left")

    window.mainloop()


def run_inference_on_upload(model_path: Path, output_dir: Path):
    """
    Opens a tkinter file dialog so the user can pick a single image, runs inference on it,
    and displays an annotated overview plus per-crop Save buttons.
    Nothing is saved to disk unless the user chooses to via a Save button.

    Runs locally only — tkinter needs a display, so this won't work headless/over SSH
    without X forwarding.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # hide the empty root window, we only want the file dialog
    root.attributes("-topmost", True)  # bring the dialog to the front

    file_path = filedialog.askopenfilename(
        title="Select an image for plate detection",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG")],
    )
    root.destroy()

    if not file_path:
        print("No file selected.")
        return

    img_path = Path(file_path)
    model = YOLO(str(model_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    crops, annotated = crop_detections(model, img_path)
    if not crops:
        print(f"No plates detected above confidence {config.yolo_confidence_threshold} in {img_path.name}")
        return

    _show_crops_with_save(crops, annotated, img_path.stem, output_dir)


if __name__ == "__main__":
    # Batch mode (original behavior)
    run_inference_and_crop(
        model_path=config.yolo100ep_best_weights,
        source_dir=config.plate_dataset_dir / "test" / "images",  # or your own held-out images
        annotated_dir=config.processed_data_dir / "ocr_test_annotated",
        crops_dir=config.processed_data_dir / "ocr_test_crops"
    )