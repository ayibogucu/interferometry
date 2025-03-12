import asyncio
import glob
import concurrent.futures
import os
from functools import partial

import lib.phase as phase
import lib.aux as aux

BATCH_SIZE = 4
MASK_RADIUS = 50
LAMBDA = 671e-9
DIR_PATH = r"./data/"


async def main():
    # Ensure output directory exists.
    os.makedirs("./output", exist_ok=True)

    # Discover image paths.
    image_paths = glob.glob(os.path.join(DIR_PATH, "*.tiff"))
    if not image_paths:
        print("No images found in the data directory.")
        return

    # Create asynchronous queues for the pipeline.
    load_queue = asyncio.Queue(maxsize=5)
    save_queue = asyncio.Queue(maxsize=5)

    # Wrap the FFT function with additional parameters.
    process_fn = partial(phase.fft_batch, radius=MASK_RADIUS, wavelength=LAMBDA)

    # Create a ProcessPoolExecutor for CPU-bound processing.
    proc_executor = concurrent.futures.ProcessPoolExecutor()

    # Create tasks for each stage.
    prod_task = asyncio.create_task(aux.producer(load_queue, image_paths, BATCH_SIZE))
    proc_task = asyncio.create_task(
        aux.processor(load_queue, save_queue, process_fn, proc_executor)
    )
    cons_task = asyncio.create_task(aux.consumer(save_queue))

    # Wait for all tasks to complete.
    await asyncio.gather(prod_task, proc_task, cons_task)
    proc_executor.shutdown()
    print("✅ Batch Processing Complete!")


if __name__ == "__main__":
    asyncio.run(main())
