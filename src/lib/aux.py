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


def plot_batch(
    array_batch, window_name: str, display_time: int = 0, auto_close: bool = False
):
    """
    Displays a batch of images in a window with a specified display time for each image.

    Parameters:
        array_batch (list of np.ndarray): List of images to display.
        window_name (str): Name of the display window.
        display_time (int): Time in milliseconds to display each image. 0 means wait for a key press.
        auto_close (bool): Whether to automatically close the window after displaying images.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for array in array_batch:
        # Normalize image for display
        array_normalized = cv2.normalize(
            src=array,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        ).astype("uint8")

        # Show the image
        cv2.imshow(window_name, array_normalized)

        while True:
            # Check for key press or timeout
            key = cv2.waitKey(display_time if display_time > 0 else 1)
            if key == 27:  # Exit if 'Esc' key is pressed
                return
            # Check if the window was closed manually
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                return
            # Break the loop for this image if no timeout was set
            if display_time > 0:
                break

    # Automatically close the window if specified
    if auto_close:
        cv2.destroyWindow(window_name)
