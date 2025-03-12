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
OUTPUT_PATH = "./output/ray_pipeline/"


def modify_path(given_path: str, new_base: str) -> str:
    parts = given_path.split(os.sep)[2:]
    return os.path.join(new_base, *parts[:-1], os.path.splitext(parts[-1])[0])


@ray.remote
def producer(img_path: str):
    img = tifffile.imread(img_path)
    return (img_path, img)


@ray.remote
def processor(data):
    img_path, img = data
    height_matrix = phase.fft(img, RAD, LAMBDA)
    return (img_path, height_matrix)


@ray.remote
def consumer(data):
    img_path, height_matrix = data
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

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Launch the pipeline: each image goes through producer, then processor, then consumer.
    prod_futures = [producer.remote(img_path) for img_path in image_paths]
    proc_futures = [processor.remote(prod_future) for prod_future in prod_futures]
    cons_futures = [consumer.remote(proc_future) for proc_future in proc_futures]

    # Wait for all consumer tasks to finish.
    processed_images = ray.get(cons_futures)
    print(f"Processed images: {processed_images}")

    ray.shutdown()
    print("✅ Processing Complete!")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"⏱️ Execution Time: {end_time - start_time:.4f} seconds")
