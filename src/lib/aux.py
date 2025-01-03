import tifffile
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor


def tiff_to_array_batch(path_batch):
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(tifffile.imread, path_batch))
    return np.stack(images, axis=0)


def get_files_recursive(directory):
    all_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))

    return all_files
