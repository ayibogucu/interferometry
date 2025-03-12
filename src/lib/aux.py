import asyncio
import concurrent.futures
import os
from functools import partial

import numpy as np
import tifffile


async def producer(
    load_queue: asyncio.Queue, image_paths: list, batch_size: int
) -> None:
    loop = asyncio.get_running_loop()
    batch = []
    for path in image_paths:
        try:
            # Offload tifffile.imread to a thread to avoid blocking.
            image = await loop.run_in_executor(None, tifffile.imread, path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        batch.append(image)
        if len(batch) == batch_size:
            await load_queue.put(np.stack(batch))
            batch = []
    if batch:
        await load_queue.put(np.stack(batch))
    # Signal termination.
    await load_queue.put(None)


async def processor(
    load_queue: asyncio.Queue,
    save_queue: asyncio.Queue,
    process_fn: partial,
    proc_executor: concurrent.futures.ProcessPoolExecutor,
) -> None:
    loop = asyncio.get_running_loop()
    while True:
        batch = await load_queue.get()
        if batch is None:
            await save_queue.put(None)
            break
        try:
            # Offload the CPU-bound FFT work to the process pool.
            processed = await loop.run_in_executor(proc_executor, process_fn, batch)
            await save_queue.put(processed)
        except Exception as e:
            print(f"Error processing batch: {e}")


async def consumer(save_queue: asyncio.Queue) -> None:
    loop = asyncio.get_running_loop()
    batch_idx = 0
    while True:
        batch = await save_queue.get()
        if batch is None:
            break
        tasks = []
        for i, img in enumerate(batch):
            out_path = os.path.join("./output", f"processed_{batch_idx}_{i}.tiff")
            # Offload tifffile.imwrite to a thread.
            tasks.append(loop.run_in_executor(None, tifffile.imwrite, out_path, img))
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error writing batch {batch_idx}: {e}")
        batch_idx += 1
