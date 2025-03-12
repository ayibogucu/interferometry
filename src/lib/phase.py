import cupy as cp
import numpy as np
from skimage.restoration import unwrap_phase


def fft(img: np.ndarray, rad: int, wavelength: float) -> cp.ndarray:
    img = cp.asarray(img)
    fft_image = cp.fft.fft2(img)

    height, width = fft_image.shape
    cy, cx = height // 2, width // 2

    # Define ROI boundaries
    x_min, x_max = 0, cx
    y_min, y_max = cy, height

    # Crop the magnitude image to the ROI
    roi = cp.abs(fft_image[y_min:y_max, x_min:x_max])

    # Find the coordinates of the maximum value in the ROI
    local_max_y, local_max_x = cp.unravel_index(cp.argmax(roi), roi.shape)

    # Map back to the full image coordinates
    global_max_y = y_min + local_max_y
    global_max_x = x_min + local_max_x

    # Create a mask with a circle at the detected maximum
    y, x = cp.ogrid[:height, :width]
    dist_sq = (y - global_max_y) ** 2 + (x - global_max_x) ** 2
    fft_image[dist_sq > rad**2] = 0

    # Shift the FFT to center the sideband
    fft_centered_image = cp.roll(
        cp.roll(fft_image, -global_max_y, axis=0), -global_max_x, axis=1
    )

    # Perform inverse FFT
    ifft_centered_image = cp.fft.ifft2(fft_centered_image)

    # Calculate the phase and unwrap it
    phase_matrix = cp.asnumpy(cp.angle(ifft_centered_image))
    phase_matrix_unwrapped = unwrap_phase(phase_matrix)
    return phase_matrix_unwrapped * wavelength * 0.15915494309189535
