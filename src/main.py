import tifffile as tiff
import numpy as np
import cv2
import os
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt

# import cupy as np
# import cucim


def tiff_to_array_batch(path_batch):
    first_image = cv2.imread(path_batch[0], cv2.IMREAD_UNCHANGED)
    if first_image is None:
        raise FileNotFoundError(f"Unable to load image: {path_batch[0]}")

    depth = len(path_batch)
    height, width = first_image.shape
    array_batch = np.empty((depth, height, width), dtype=first_image.dtype)

    for i, path in enumerate(path_batch):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")
        array_batch[i] = image
    return array_batch


# def tiff_to_array_batch(path_batch):
#     first_image = cv2.imread(path_batch[0], cv2.IMREAD_UNCHANGED)
#     if first_image is None:
#         raise FileNotFoundError(f"Unable to load image: {path_batch[0]}")
#
#     depth = len(path_batch)
#     height, width = first_image.shape
#     array_batch = np.empty((depth, height, width), dtype=first_image.dtype)
#
#     array_batch[0] = np.array(first_image)
#
#     for i, path in enumerate(path_batch[1:], start=1):
#         image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#         if image is None:
#             raise FileNotFoundError(f"Unable to load image: {path}")
#
#         array_batch[i] = np.array(image)
#
#     return array_batch


def get_files_recursive(directory):
    all_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))

    return all_files


def phase_algorithm(array_batch: np.ndarray, radius: int):
    batch_size, height, width = array_batch.shape

    dft = np.fft.fft2(array_batch, axes=(-2, -1))
    dft_shift = np.fft.fftshift(dft, axes=(-2, -1))

    mask = np.zeros((batch_size, height, width), dtype=np.uint8)
    cx, cy = width // 2, height // 2

    for i in range(batch_size):
        cv2.circle(mask[i], (cx, cy), radius, (255, 255, 255), -1)
    mask = 255 - mask

    # apply mask to dft_shift
    dft_shift_masked = np.multiply(dft_shift, mask)

    # shift origin from center to upper left corner
    back_ishift_masked = np.fft.ifftshift(dft_shift_masked, axes=(-2, -1))

    # do idft saving as complex output
    img_filtered_complex = np.fft.ifft2(back_ishift_masked, axes=(-2, -1))

    phase_filtered = np.angle(img_filtered_complex)

    return phase_filtered


def unwrap(phase_batch):
    unwrapped_phase_batch = np.apply_along_axis(unwrap_phase, 1, phase_batch)
    return unwrapped_phase_batch


def OPD(phase_batch, wavelength):
    return np.multiply(phase_batch, wavelength) / (2 * np.pi)


def plot_results(array_batch):
    # Plot the first image (for example)
    plt.figure(figsize=(10, 5))
    plt.imshow(array_batch[0], cmap="gray")
    plt.title("Optical Path Difference (OPD) of First Image")
    plt.colorbar()
    plt.show()


def main2():
    MASK_RADIUS = 32
    WAVELENGTH = 700e-9

    path_batch = ["1.tiff", "2.tiff", "3.tiff", "4.tiff"]
    images = [tiff.imread(path) for path in path_batch]
    array_batch = np.array(images)  ## PHASE ALGORITHM

    array_batch = phase_algorithm(array_batch, MASK_RADIUS)

    ## UNWRAP
    array_batch = unwrap(array_batch)

    ##ODP
    array_batch = OPD(array_batch, WAVELENGTH)

    ## PLOT THIS SHIT PLEASE CHAT GIPPITY
    plot_results(array_batch)


def main():
    BATCH_SIZE = 4
    MASK_RADIUS = 32
    WAVELENGTH = 700e-9

    all_files = get_files_recursive("/data")
    for idx in range(0, len(all_files), BATCH_SIZE):
        ## INPUT HANDLEDED IN BATCHES
        file_path = all_files[idx : idx + BATCH_SIZE]
        array_batch = tiff_to_array_batch(file_path)

        print(array_batch[0].shape)

        ## PHASE ALGORITHM
        array_batch = phase_algorithm(array_batch, MASK_RADIUS)

        ## UNWRAP
        array_batch = unwrap(array_batch)

        ##ODP
        array_batch = OPD(array_batch, WAVELENGTH)

        ## PLOT THIS SHIT PLEASE CHAT GIPPITY
        plot_results(array_batch)


# if __name__ == "__main__":
main()
main2()
print("pooppo")
