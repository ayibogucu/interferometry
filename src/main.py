import cupy as cp
import cv2
from skimage.restoration import unwrap_phase

# import matplotlib.pyplot as plt
import os


def tiff_to_array_batch(path_batch):
    first_image = cv2.imread(path_batch[0], cv2.IMREAD_UNCHANGED)
    if first_image is None:
        raise FileNotFoundError(f"Unable to load image: {path_batch[0]}")

    depth = len(path_batch)
    height, width = first_image.shape
    array_batch = cp.empty((depth, height, width), dtype=first_image.dtype)

    array_batch[0] = cp.array(first_image)

    for i, path in enumerate(path_batch[1:], start=1):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")

        array_batch[i] = cp.array(image)

    return array_batch


def get_files_recursive(directory):
    all_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))

    return all_files


def phase_algorithm(array):
    pass


def OPD(array):
    pass


def main():
    all_files = get_files_recursive("/data")
    BATCH_SIZE = 4

    for idx in range(0, len(all_files), BATCH_SIZE):
        ## INPUT HANDLEDED IN BATCHES
        input_batch = all_files[idx : idx + BATCH_SIZE]
        array_batch = tiff_to_array_batch(input_batch)

        ## PHASE ALGORITHM
        phase_batch = phase_algorithm(array_batch)

        ## UNWRAP
        shit = unwrap_phase(phase_batch)

        ##ODP
        poopoo = OPD(shit)

        for i in range(BATCH_SIZE):
            file_name = f"poopoo{i+1}.tiff"
            cv2.imwrite(file_name, poopoo)
