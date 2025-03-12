import os
import glob
import time

import ray
import tifffile
import numpy as np

import lib.phase as phase

RAD = 50
LAMBDA = 671e-9
DIR_PATH = "./data/"
OUTPUT_PATH = "./output/ray/"


def modify_path(given_path: str, new_base: str) -> str:
    parts = given_path.split(os.sep)[2:]
    return os.path.join(new_base, *parts[:-1], os.path.splitext(parts[-1])[0])


@ray.remote(num_gpus=1)
def process_image(img_path: str) -> str:
    # Load the image from disk.
    img = tifffile.imread(img_path)
    # Process the image using your GPU-accelerated function.
    height_matrix = phase.fft(img, RAD, LAMBDA)
    # Determine and create the necessary output directory.
    save_path = modify_path(img_path, OUTPUT_PATH)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the processed result.
    np.save(save_path, height_matrix)
    return img_path


def main() -> int:
    # Initialize Ray.
    ray.init()

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    image_paths = glob.glob(os.path.join(DIR_PATH, "*.tiff"))
    if not image_paths:
        print("No images found in the data directory.")
        ray.shutdown()
        return 0

    # Launch a Ray task for each image.
    tasks = [process_image.remote(img_path) for img_path in image_paths]
    # Wait for all tasks to complete.
    ray.get(tasks)

    ray.shutdown()
    print("✅ Processing Complete!")
    return len(image_paths)


if __name__ == "__main__":
    start_time = time.perf_counter()
    imcount = main()
    end_time = time.perf_counter()
    print(f"⏱️ Execution Time: {end_time - start_time:.4f} seconds")
    if end_time - start_time > 0:
        print(f"FPS: {imcount / (end_time - start_time):.2f}")
