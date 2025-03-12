import time
import os
import glob

import tifffile
import numpy as np

import lib.phase as phase

RAD = 50
LAMBDA = 671e-9
DIR_PATH = "./data/"
OUTPUT_PATH = "./output/naive/"


def modify_path(given_path: str, new_base: str) -> str:
    parts = given_path.split(os.sep)[2:]
    return os.path.join(new_base, *parts[:-1], os.path.splitext(parts[-1])[0])


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    image_paths = glob.glob(os.path.join(DIR_PATH, "*.tiff"))
    if not image_paths:
        print("No images found in the data directory.")
        return

    for img_path in image_paths:
        img = tifffile.imread(img_path)
        height_matrix = phase.fft(img, RAD, LAMBDA)
        save_path = modify_path(img_path, OUTPUT_PATH)
        np.save(save_path, height_matrix)

    print("✅ Processing Complete!")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"⏱️ Execution Time: {end_time - start_time:.4f} seconds")
