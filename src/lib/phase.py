import numpy as np
from skimage.restoration import unwrap_phase


def fft(img: np.ndarray, rad: int, wavelength: float) -> np.ndarray:
    fft_image = np.fft.fft2(img)
    height, width = fft_image.shape
    cy, cx = height // 2, width // 2

    roi = np.abs(fft_image[cy:height, :cx])
    local_max_y, global_max_x = np.unravel_index(np.argmax(roi), roi.shape)
    global_max_y = cy + local_max_y

    y, x = np.ogrid[:height, :width]
    dist_sq = (y - global_max_y) ** 2 + (x - global_max_x) ** 2
    fft_image[dist_sq > rad**2] = 0

    fft_centered_image = np.roll(
        np.roll(fft_image, -global_max_y, axis=0), -global_max_x, axis=1
    )

    ifft_centered_image = np.fft.ifft2(fft_centered_image)

    phase_matrix = np.angle(ifft_centered_image)
    phase_matrix_unwrapped = unwrap_phase(phase_matrix)
    return phase_matrix_unwrapped * wavelength * 0.15915494309189535
