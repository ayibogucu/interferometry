import cv2
import numpy as np
import tifffile
from skimage.restoration import unwrap_phase


def process_interferometry_image(image_path, mask_radius=32):
    # Step 1: Load the interferometry image
    image = tifffile.imread(image_path)

    # Step 2: Apply FFT to the image
    fft = np.fft.fft2(image)

    # Step 3: Apply fftshift to center the frequencies
    fft_shift = np.fft.fftshift(fft)

    # Step 4: Compute magnitude of FFT for visualization
    fft_mag = np.abs(fft)
    fft_shift_mag = np.abs(fft_shift)

    # Step 5: Create a circular mask (removes low frequencies)
    mask = np.zeros_like(image)
    height, width = mask.shape
    cy, cx = height // 2, width // 2
    cv2.circle(mask, (cx, cy), mask_radius, (255, 255, 255), -1)
    mask = 255 - mask  # Invert mask to remove DC component

    # Step 6: Apply the mask to the shifted FFT
    fft_masked = np.multiply(mask, fft_shift)

    # Step 7: Apply inverse FFT to get the processed image
    ifft_masked_back = np.fft.ifft2(fft_masked)

    # Step 8: Extract phase from the inverse FFT result
    ifft_masked_back_angle = np.angle(ifft_masked_back)

    # Step 9: Unwrap the phase
    unwrapped_phase = unwrap_phase(ifft_masked_back_angle)

    # Step 10: Normalize unwrapped phase for visualization
    unwrapped_normalized = np.interp(
        unwrapped_phase, (unwrapped_phase.min(), unwrapped_phase.max()), (0, 255)
    ).astype(np.uint8)

    # Step 11: Display results
    cv2.imshow("Original Image", image)
    cv2.imshow(
        "FFT Magnitude (Log Scale)", np.log1p(fft_mag) / np.log1p(fft_mag).max()
    )  # Log scale for FFT magnitude
    cv2.imshow(
        "Shifted FFT Magnitude (Log Scale)",
        np.log1p(fft_shift_mag) / np.log1p(fft_shift_mag).max(),
    )  # Log scale for shifted FFT magnitude
    cv2.imshow(
        "Masked FFT Magnitude", np.abs(fft_masked) / np.abs(fft_masked).max()
    )  # Magnitude of masked FFT
    cv2.imshow(
        "Magnitude of Inverse FFT",
        np.abs(ifft_masked_back) / np.abs(ifft_masked_back).max(),
    )  # Magnitude of the processed image
    cv2.imshow(
        "Phase Information", ifft_masked_back_angle
    )  # Phase information from inverse FFT
    cv2.imshow(
        "Unwrapped Phase", unwrapped_normalized
    )  # Unwrapped phase for visualization

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
process_interferometry_image("./off-axis-data/1.tiff", mask_radius=32)
