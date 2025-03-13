import numpy as np
from skimage.restoration import unwrap_phase


def batch_fft(imgs: np.ndarray, rad: int, wavelength: float) -> np.ndarray:
    # Convert the batch to a CuPy array

    # Compute FFT on each image in the batch along the last two axes.
    fft_images = np.fft.fft2(imgs, axes=(-2, -1))  # shape: (B, H, W)
    B, H, W = fft_images.shape
    cy, cx = H // 2, W // 2

    # Define ROI boundaries (common for all images, assuming same shape).
    x_min, x_max = 0, cx
    y_min, y_max = cy, H

    # Compute ROI on the magnitude image in batch.
    roi = np.abs(fft_images[:, y_min:y_max, x_min:x_max])  # shape: (B, H-cy, cx)
    B, roi_h, roi_w = roi.shape

    # Flatten the ROI per image and find the index of the maximum value.
    roi_flat = roi.reshape(B, -1)  # shape: (B, roi_h*roi_w)
    argmaxes = np.argmax(roi_flat, axis=1)  # shape: (B,)

    # Compute local (within ROI) max coordinates using division and modulo.
    local_max_y = argmaxes // roi_w
    local_max_x = argmaxes % roi_w

    # Map local coordinates back to full-image coordinates.
    global_max_y = y_min + local_max_y  # shape: (B,)
    global_max_x = x_min + local_max_x  # shape: (B,)

    # Create a coordinate grid for each image.
    y_grid = np.arange(H).reshape(1, H, 1)  # shape: (1, H, 1)
    x_grid = np.arange(W).reshape(1, 1, W)  # shape: (1, 1, W)

    # Expand global max coordinates to allow broadcasting.
    global_max_y_exp = global_max_y.reshape(B, 1, 1)  # shape: (B, 1, 1)
    global_max_x_exp = global_max_x.reshape(B, 1, 1)  # shape: (B, 1, 1)

    # Compute squared distance from the detected maximum for each image.
    dist_sq = (y_grid - global_max_y_exp) ** 2 + (
        x_grid - global_max_x_exp
    ) ** 2  # shape: (B, H, W)

    # Apply circular mask: zero out FFT coefficients outside the circle.
    mask = dist_sq > rad**2
    fft_images[mask] = 0

    # Shift the FFT so that the detected maximum is at the origin.
    # Since np.roll does not support different shifts for each image in one call,
    # we loop over the batch dimension.
    fft_centered_list = []
    for i in range(B):
        shift_y = int(global_max_y[i].item())
        shift_x = int(global_max_x[i].item())
        shifted = np.roll(fft_images[i], shift=-shift_y, axis=0)
        shifted = np.roll(shifted, shift=-shift_x, axis=1)
        fft_centered_list.append(shifted)
    fft_centered = np.stack(fft_centered_list, axis=0)  # shape: (B, H, W)

    # Inverse FFT on the batch.
    ifft_centered = np.fft.ifft2(fft_centered, axes=(-2, -1))

    # Compute phase.
    phase_matrix = np.angle(ifft_centered)  # shape: (B, H, W)

    # Convert to NumPy for phase unwrapping (skimage works on npU).
    phase_matrix_np = phase_matrix

    # Unwrap the phase per image.
    unwrapped_list = [unwrap_phase(phase_matrix_np[i]) for i in range(B)]
    unwrapped = np.stack(unwrapped_list, axis=0)

    # Multiply by wavelength constant and scaling factor.
    return unwrapped * wavelength * 0.15915494309189535
