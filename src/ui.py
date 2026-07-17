from pathlib import Path
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from config.config import config
from src.detection import crop_detections


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

