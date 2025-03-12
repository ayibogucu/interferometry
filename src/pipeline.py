import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import tifffile
import numpy as np

# Import the batched FFT function
from lib.phase import batch_fft

# Constants
RAD = 50
LAMBDA = 671e-9
DIR_PATH = "./data/"
OUTPUT_PATH = "./output/parallel_batched/"


def modify_path(given_path: str, new_base: str) -> str:
    parts = given_path.split(os.sep)[2:]
    return os.path.join(new_base, *parts[:-1], os.path.splitext(parts[-1])[0])


def process_batch(batch_paths: list, rad: int, wavelength: float) -> int:
    """
    Process a batch of images:
      1. Loads images from disk.
      2. Stacks them into a NumPy array.
      3. Processes the batch on the GPU via batch_fft.
      4. Saves each processed image.
    Returns the number of images processed.
    """
    images = []
    for path in batch_paths:
        try:
            img = tifffile.imread(path)
            images.append(img)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    if not images:
        return 0

    # Stack images to form a batch (shape: B x H x W)
    imgs_np = np.stack(images, axis=0)

    # Process the batch on the GPU
    processed_batch = batch_fft(imgs_np, rad, wavelength)

    # Save each processed image
    for i, path in enumerate(batch_paths):
        try:
            save_path = modify_path(path, OUTPUT_PATH)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, processed_batch[i])
        except Exception as e:
            print(f"Error saving {path}: {e}")

    return len(batch_paths)


def main(batch_size: int = 10, num_workers: int = 4) -> int:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    image_paths = glob.glob(os.path.join(DIR_PATH, "*.tiff"))
    if not image_paths:
        print("No images found in the data directory.")
        return 0

    # Create batches of image paths
    batches = [
        image_paths[i : i + batch_size] for i in range(0, len(image_paths), batch_size)
    ]

    total_processed = 0
    # Use ProcessPoolExecutor to run batches in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_batch, batch, RAD, LAMBDA) for batch in batches
        ]
        for future in as_completed(futures):
            try:
                total_processed += future.result()
            except Exception as e:
                print(f"Error processing a batch: {e}")

    return total_processed


if __name__ == "__main__":
    start_time = time.perf_counter()
    count = main(
        batch_size=55, num_workers=12
    )  # Adjust batch size and number of workers as needed.
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print("✅ Processing Complete!")
    print(f"⏱️ Execution Time: {total_time:.4f} seconds")
    if total_time > 0:
        print(f"FPS: {count / total_time:.2f}")
