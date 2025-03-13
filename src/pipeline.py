import os
import glob
import time
import tifffile
import numpy as np
from dask.base import compute
from dask.delayed import delayed

# Import the batched FFT function from your phase file.
from lib.phase import batch_fft

# Constants
RAD = 50
LAMBDA = 671e-9
DIR_PATH = "./data/"
OUTPUT_PATH = "./output/dask_full/"
BATCH_SIZE = 16


def modify_path(given_path: str, new_base: str) -> str:
    parts = given_path.split(os.sep)[2:]
    return os.path.join(new_base, *parts[:-1], os.path.splitext(parts[-1])[0])


@delayed
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

    # Process the batch on the GPU using your CuPy-based FFT function.
    processed_batch = batch_fft(imgs_np, rad, wavelength)

    # Save each processed image.
    for i, path in enumerate(batch_paths):
        try:
            save_path = modify_path(path, OUTPUT_PATH)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, processed_batch[i])
        except Exception as e:
            print(f"Error saving {path}: {e}")

    return len(batch_paths)


def main() -> int:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    image_paths = glob.glob(os.path.join(DIR_PATH, "*.tiff"))
    if not image_paths:
        print("No images found in the data directory.")
        return 0

    # Create batches of image paths.
    batches = [
        image_paths[i : i + BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)
    ]

    # Wrap each batch processing call as a Dask delayed task.
    tasks = [process_batch(batch, RAD, LAMBDA) for batch in batches]

    # Trigger the computation of the entire task graph.
    results = compute(*tasks)
    total_processed = sum(results)
    return total_processed


if __name__ == "__main__":
    start_time = time.perf_counter()
    imcount = main()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print("✅ Processing Complete!")
    print(f"Processed {imcount} images.")
    print(f"⏱️ Execution Time: {total_time:.4f} seconds")
    if total_time > 0:
        print(f"FPS: {imcount / total_time:.2f}")
