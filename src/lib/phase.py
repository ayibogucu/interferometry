import numpy as np
from skimage.restoration import unwrap_phase


def fft_batch(array_batch: np.ndarray, radius: int, wavelength: float) -> np.ndarray:
    batch_size, height, width = array_batch.shape
    cy, cx = height // 2, width // 2
    y_min, y_max = cy, height
    x_min, x_max = 0, cx

    # Compute FFT for each image
    fft = np.fft.fft2(array_batch)

    # Define the ROI for the sideband (3rd quadrant)
    roi = np.abs(fft[:, y_min:y_max, x_min:x_max])

    # Compute per-image max index in ROI (flattened)
    roi_flat = roi.reshape(batch_size, -1)
    local_max_idx = np.argmax(roi_flat, axis=1)
    local_max_y = local_max_idx // roi.shape[2]
    local_max_x = local_max_idx % roi.shape[2]

    # Convert to global image coordinates
    global_max_y = y_min + local_max_y
    global_max_x = x_min + local_max_x

    # Precompute meshgrid once for one image
    y_grid = np.arange(height)
    x_grid = np.arange(width)
    Y, X = np.meshgrid(y_grid, x_grid, indexing="ij")  # shape (height, width)

    # Compute mask for the sideband for each image via broadcasting
    mask = (
        (Y - global_max_y[:, None, None]) ** 2 + (X - global_max_x[:, None, None]) ** 2
    ) > radius**2
    fft[mask] = 0

    # Compute per-image shifts for both axes (vectorized)
    shift_y = global_max_y
    shift_x = global_max_x
    new_y_idx = (
        np.arange(height)[None, :] - shift_y[:, None]
    ) % height  # shape (batch, height)
    new_x_idx = (
        np.arange(width)[None, :] - shift_x[:, None]
    ) % width  # shape (batch, width)

    # Use advanced indexing to perform a per-image roll without explicit loops over pixels
    batch_idx = np.arange(batch_size)[:, None, None]
    fft_shifted = fft[batch_idx, new_y_idx[:, :, None], new_x_idx[:, None, :]]

    # Inverse FFT on the last two axes
    ifft_result = np.fft.ifft2(fft_shifted, axes=(-2, -1))

    # Extract phase
    phase = np.angle(ifft_result)
    # For 2D phase unwrapping, we must apply unwrap_phase to each image.
    # This unavoidable loop only iterates over the batch dimension.
    unwrapped_phase = np.array([unwrap_phase(im) for im in phase])

    # Multiply by constant factor
    return unwrapped_phase * wavelength * 0.15915494309189535
