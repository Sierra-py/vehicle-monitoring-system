"""Rename all file in a folder to serially numbered.
Usage: python -m scripts.rename --input 'path/to/folder' """
import os
import sys
from config.config import config
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input",type = str, default=config.simulation_data_dir, help="Write the path of the folder")
args = parser.parse_args()

# Folder containing the files
folder_path = Path(args.input)

if not folder_path.is_dir():
    sys.exit("The input directory is invalid")

# Get all files (ignore subdirectories)
files = [f for f in os.listdir(folder_path)
         if os.path.isfile(os.path.join(folder_path, f))]

# Sort files alphabetically
files.sort()

# Rename files
for index, filename in enumerate(files, start=1):
    old_path = os.path.join(folder_path, filename)

    # Get file extension
    _, extension = os.path.splitext(filename)

    # Create new filename
    new_filename = f"{index:03d}{extension}"
    new_path = os.path.join(folder_path, new_filename)

    os.rename(old_path, new_path)
    print(f"{filename} -> {new_filename}")

print("Renaming completed.")