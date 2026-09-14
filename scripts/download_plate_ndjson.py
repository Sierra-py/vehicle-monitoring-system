"""Run this script to download the plate.ndjson file from ultralytics."""
import os
import requests
from config.config import config

# the url expired had to download manually
url = "https://storage.googleapis.com/alpha-ultralytics-eu/exports/6a4cc0f6d3678f738f45ad35/plate.ndjson?X-Goog-Algorithm=GOOG4-HMAC-SHA256&X-Goog-Credential=GOOG1EVYATYKKOGSZSQSTG4P6ISYXQTE4HWDCBAWNAEWGN34SPK6JC6CK22HP%2F20260817%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260817T095636Z&X-Goog-Expires=604800&X-Goog-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22plate.ndjson%22&X-Goog-Signature=c3d44325c9f4a79c5cf131537377b43d24a4f8afcd28deb2fec317fc3dc97107"
save_path = config.plate_ndjson
try:
    # Create the destination directory if it doesn't exist
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

    print(f"Downloaded successfully to: {save_path}")

except requests.RequestException as e:
    print(f"Download failed: {e}")
except OSError as e:
    print(f"Could not save the file: {e}")