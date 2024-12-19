import cv2
import numpy as np
import cupy as cp
import os
# import cucim


# TODO: LOOK AT CUCIM WITH NDIVIA DGS APIS AND NOT USE CV2 FOR THIS.
def tiff_to_array_batch_gpu(path_batch):
    first_image = cv2.imread(path_batch[0], cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        raise FileNotFoundError(f"Unable to load image: {path_batch[0]}")

    depth = len(path_batch)
    height, width = first_image.shape
    array_batch = cp.empty((depth, height, width), dtype=first_image.dtype)

    array_batch[0] = cp.array(first_image)

    for i, path in enumerate(path_batch[1:], start=1):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")

        array_batch[i] = cp.array(image)

    return array_batch


def tiff_to_array_batch(path_batch):
    first_image = cv2.imread(path_batch[0], cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        raise FileNotFoundError(f"Unable to load image: {path_batch[0]}")

    depth = len(path_batch)
    height, width = first_image.shape
    array_batch = np.empty((depth, height, width), dtype=first_image.dtype)

    for i, path in enumerate(path_batch):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")
        array_batch[i] = image
    return array_batch


def get_files_recursive(directory):
    all_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))

    return all_files


def plot_batch(array_batch, window_name: str):
    for array in array_batch:
        array_normalized = cv2.normalize(
            src=array,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        array_normalized = array_normalized.astype("uint8")

        cv2.imshow(window_name, array_normalized)
        while True:
            key = cv2.waitKey(100)  # Check every 100 ms
            if key != -1:  # If a key is pressed
                break
            # Check if the window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
