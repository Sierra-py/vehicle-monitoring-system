from pathlib import Path
from ultralytics import YOLO
import tkinter as tk
import csv
from tkinter import filedialog
from PIL import Image, ImageTk
from config.config import config
from src.detection import crop_detections
from src.ocr import recognize_plate


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

# ------------------------------------------------------------------------------

def run_inference_on_upload(model_path: Path, output_dir: Path):
    """
    Opens a tkinter file dialog so the user can pick a single image, runs inference on it,
    and displays an annotated overview plus per-crop Save buttons.
    Nothing is saved to disk unless the user chooses to via a Save button.

    Runs locally only — tkinter needs a display, so this won't work headless/over SSH
    without X forwarding.
    """

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

# ------------------------------------------------------------------------------

def run_ocr_on_upload(ocr_model):
    """
    Opens a tkinter file dialog so the user can pick a single CROPPED plate image
    (not a full frame — this expects the same kind of image crop_detections
    produces), runs OCR on it, and displays the image alongside the predicted text.

    Takes an already-loaded OCR model (from src.ocr.load_ocr_model()) rather than
    loading it internally, since OCR model load is comparatively expensive and you'll
    likely want to test several images in a row without reloading each time.

    Runs locally only — tkinter needs a display, so this won't work headless/over SSH
    without X forwarding.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select a cropped plate image for OCR",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG")],
    )
    root.destroy()

    if not file_path:
        print("No file selected.")
        return

    img_path = Path(file_path)
    img = Image.open(img_path).convert("RGB")
    predicted_text = recognize_plate(ocr_model, img)

    window = tk.Tk()
    window.title(f"OCR result — {img_path.name}")

    DISPLAY_MAX_SIDE = 400
    display_img = img.copy()
    display_img.thumbnail((DISPLAY_MAX_SIDE, DISPLAY_MAX_SIDE))
    photo = ImageTk.PhotoImage(display_img)
    window._photo_ref = photo  # keep alive, same reason as in _show_crops_with_save

    tk.Label(window, image=photo).pack(padx=10, pady=10)

    result_text = predicted_text if predicted_text else "(no text detected)"
    tk.Label(
        window,
        text=result_text,
        font=("Segoe UI", 16, "bold"),
        fg="black" if predicted_text else "gray",
    ).pack(padx=10, pady=(0, 10))

    window.mainloop()

# ------------------------------------------------------------------------------

def label_ocr_predictions(ocr_model, crops_dir: Path, output_csv: Path):
    """
    Fast correction-labeling UI for building an OCR ground-truth dataset. Shows each
    crop with the model's predicted text pre-filled and pre-selected in an editable
    box — press Enter to accept as-is, or just start typing to overwrite it if the
    prediction is wrong. This keeps labeling fast: correct predictions cost one
    keypress, wrong ones cost only as much typing as the correction itself, never a
    full plate typed from a blank field.
 
    Resumes automatically: filenames already present in output_csv are skipped, so
    labeling can be done across multiple sessions without redoing work. Safe to call
    against a different crops_dir / output_csv pair each time — nothing here assumes
    a single fixed dataset, so this can be rerun as new data folders come in.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
 
    crop_paths = sorted(list(crops_dir.glob("*.jpg")) + list(crops_dir.glob("*.png")))
    if not crop_paths:
        print(f"No images found in {crops_dir}")
        return
 
    labeled = set()
    if output_csv.exists():
        with open(output_csv, newline="") as f:
            labeled = {row["filename"] for row in csv.DictReader(f)}
 
    remaining = [p for p in crop_paths if p.name not in labeled]
    print(f"{len(labeled)} already labeled, {len(remaining)} remaining in {crops_dir.name}")
 
    if not remaining:
        print("Nothing left to label.")
        return
 
    write_header = not output_csv.exists()
    csv_file = open(output_csv, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["filename", "predicted_text", "corrected_text", "was_correct"])
 
    state = {"index": 0}
 
    window = tk.Tk()
    window.title(f"OCR labeling — {crops_dir.name}")
 
    img_label = tk.Label(window)
    img_label.pack(padx=10, pady=10)
 
    progress_label = tk.Label(window, font=("Segoe UI", 9), fg="gray")
    progress_label.pack()
 
    text_var = tk.StringVar()
    entry = tk.Entry(window, textvariable=text_var, font=("Segoe UI", 18), justify="center", width=20)
    entry.pack(padx=10, pady=10)
    entry.focus()
 
    window._photo_ref = None  # keep alive, same reason as elsewhere in this module
 
    def load_current():
        i = state["index"]
        if i >= len(remaining):
            csv_file.close()
            window.destroy()
            print(f"All done. Labels saved to {output_csv}")
            return
 
        path = remaining[i]
        img = Image.open(path).convert("RGB")
        predicted = recognize_plate(ocr_model, img) or ""
 
        state["current_path"] = path
        state["current_predicted"] = predicted
 
        display_img = img.copy()
        display_img.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(display_img)
        window._photo_ref = photo
        img_label.configure(image=photo)
 
        text_var.set(predicted)
        entry.select_range(0, tk.END)
        progress_label.configure(text=f"{i + 1} / {len(remaining)}  —  {path.name}")
 
    def submit(event=None):
        path = state["current_path"]
        predicted = state["current_predicted"]
        corrected = text_var.get().strip()
        was_correct = (corrected == predicted)
 
        writer.writerow([path.name, predicted, corrected, was_correct])
        csv_file.flush()
 
        state["index"] += 1
        load_current()
 
    def skip(event=None):
        # For genuinely unreadable/garbage crops — marked explicitly rather than
        # silently dropped, so the resume logic still treats it as handled.
        path = state["current_path"]
        writer.writerow([path.name, state["current_predicted"], "UNREADABLE", False])
        csv_file.flush()
        state["index"] += 1
        load_current()
 
    entry.bind("<Return>", submit)
    window.bind("<Escape>", skip)
 
    hint = tk.Label(
        window,
        text="Enter = accept/save text shown  |  Esc = mark unreadable/skip",
        font=("Segoe UI", 9), fg="gray"
    )
    hint.pack(pady=(0, 10))
 
    load_current()
    window.mainloop()
