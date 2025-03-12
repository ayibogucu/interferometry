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
OUTPUT_PATH = "./output/ray_workers/"
WORKER_NUM = 4


def modify_path(given_path: str, new_base: str) -> str:
    parts = given_path.split(os.sep)[2:]
    return os.path.join(new_base, *parts[:-1], os.path.splitext(parts[-1])[0])


@ray.remote
class Worker:
    def process_image(self, img_path: str):
        # Read the image (I/O-bound)
        img = tifffile.imread(img_path)
        # Process the image (CPU-bound)
        height_matrix = phase.fft(img, RAD, LAMBDA)
        # Create output directory if needed and save the result (I/O-bound)
        save_path = modify_path(img_path, OUTPUT_PATH)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, height_matrix)
        return img_path


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    image_paths = glob.glob(os.path.join(DIR_PATH, "*.tiff"))
    if not image_paths:
        print("No images found in the data directory.")
        return

    # Initialize Ray.
    ray.init(ignore_reinit_error=True)

    # Create a pool of persistent worker actors.
    workers = [Worker.remote() for _ in range(WORKER_NUM)]

    # Distribute the images to workers in a round-robin fashion.
    futures = [
        workers[i % WORKER_NUM].process_image.remote(img_path)
        for i, img_path in enumerate(image_paths)
    ]

    # Wait for all tasks to complete.
    results = ray.get(futures)

    ray.shutdown()

    return len(results)


if __name__ == "__main__":
    start_time = time.perf_counter()
    imcount = main()
    end_time = time.perf_counter()
    print(f"⏱️ Execution Time: {end_time - start_time:.4f} seconds")
    print(f"FPS: {imcount / (end_time - start_time)}")
