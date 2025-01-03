import numpy as np
import tifffile
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.restoration import unwrap_phase


def process_interferometry_image(image_path, R0=50, R=50):
    # Step 1: Load the interferometry image (in tiff format)
    hologram = tifffile.imread(image_path)
    N1, N2 = hologram.shape

    # Step 2: Display the off-axis hologram
    plt.figure()
    plt.imshow(np.flipud(np.rot90(hologram)), cmap="gray")
    plt.title("Off-axis hologram")
    plt.show()

    # Step 3: Calculate Fourier Transform and show the absolute value in log scale
    spectrum = fftshift(fft2(fftshift(hologram)))
    spectrum_abs = np.abs(spectrum)

    plt.figure()
    plt.imshow(np.flipud(np.rot90(np.log1p(spectrum_abs))), cmap="gray")
    plt.title("Fourier Spectrum in Log Scale / a.u.")
    plt.show()

    # Step 4: Blocking the central part of the spectrum (removing low-frequency components)
    spectrum_abs1 = np.zeros_like(spectrum_abs)
    for ii in range(N1):
        for jj in range(N2):
            x = ii - N1 // 2
            y = jj - N2 // 2
            if np.sqrt(x**2 + y**2) > R0:
                spectrum_abs1[ii, jj] = spectrum_abs[ii, jj]

    # Step 5: Block half of the spectrum
    spectrum_abs1[: N1 // 2, :] = 0

    # Step 6: Find the position of the side-band in the spectrum
    maximum = np.max(spectrum_abs1)
    y0, x0 = np.unravel_index(np.argmax(spectrum_abs1), spectrum_abs1.shape)

    # Step 7: Shift the complex-valued spectrum to the center
    spectrum2 = np.zeros_like(
        spectrum_abs1, dtype=complex
    )  # Ensure complex type for the spectrum
    x0 -= N1 // 2
    y0 -= N2 // 2

    # Ensure proper bounds when shifting the spectrum
    x_shift = int(x0)
    y_shift = int(y0)

    for ii in range(N1):
        for jj in range(N2):
            new_x = ii + x_shift
            new_y = jj + y_shift

            # Apply boundary check to avoid out-of-bounds access
            new_x = (new_x + N1) % N1  # Wrap around with modulo
            new_y = (new_y + N2) % N2  # Wrap around with modulo

            spectrum2[ii, jj] = spectrum[new_x, new_y]

    # Step 8: Visualize the shifted spectrum (in log scale)
    spectrum_abs2 = np.abs(spectrum2)
    plt.figure()
    plt.imshow(np.log1p(spectrum_abs2), cmap="gray")
    plt.title("Shifted Spectrum in Log Scale / a.u.")
    plt.show()

    # Step 9: Select the central part of the complex-valued spectrum
    spectrum3 = np.zeros_like(spectrum2)
    for ii in range(N1):
        for jj in range(N2):
            x = ii - N1 // 2
            y = jj - N2 // 2
            if np.sqrt(x**2 + y**2) < R:
                spectrum3[ii, jj] = spectrum2[ii, jj]

    # Step 10: Visualize the central part of the spectrum (in log scale)
    spectrum_abs3 = np.abs(spectrum3)
    plt.figure()
    plt.imshow(np.log1p(spectrum_abs3), cmap="gray")
    plt.title("Fourier Spectrum in Log Scale / a.u. (Central Part)")
    plt.show()

    # Step 11: Inverse FFT of the selected spectrum
    reconstruction = ifftshift(ifft2(ifftshift(spectrum3)))
    rec_abs = np.abs(reconstruction)
    l = np.angle(reconstruction)

    # Step 12: Display wrapped phase
    plt.figure()
    plt.imshow(np.flipud(np.rot90(l)), cmap="gray")
    plt.title("Reconstructed Phase (Wrapped) / rad")
    plt.show()

    # Step 13: Unwrap the phase
    unwrapped_phase = unwrap_phase(l)

    # Step 14: Display unwrapped phase
    plt.figure()
    plt.imshow(np.flipud(np.rot90(unwrapped_phase)), cmap="gray")
    plt.title("Unwrapped Phase / rad")
    plt.show()

    return unwrapped_phase


# Example usage
unwrapped_phase = process_interferometry_image("./off-axis-data/1.tiff", R0=50, R=50)
