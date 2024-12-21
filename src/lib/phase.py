import numpy as np
import cv2


def fft_batch(array_batch: np.ndarray, radius: int):
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
