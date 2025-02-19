# TODO: PYTHON 3.12. CUCIM SUPPORTS TIFF.
import cupy as cp
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
import tifffile
from lib.plot import plot_surface


# Define an in-place kernel: for each pixel, compute its x, y coordinates from the flat index.
# If the pixel is outside the circle centered at (global_max_x, global_max_y), set it to 0.
circle_mask_kernel = cp.ElementwiseKernel(
    "T fft, int32 width, int32 cx, int32 cy, int32 rad2",  # input parameters
    "T out",  # output parameter
    """
    int x = i % width;     // compute x-coordinate from flat index i
    int y = i / width;     // compute y-coordinate
    int dx = x - cx;
    int dy = y - cy;
    // If outside the circle, set output to 0, otherwise retain fft value.
    out = (dx*dx + dy*dy > rad2) ? (T)0 : fft;
    """,
    "circle_mask",
)


# VARIABLES
RAD = 50
LAMBDA = 671e-9


# IO
image = tifffile.imread("./data/100.tiff")
image = cp.array(image)


# fft
fft_image = cp.fft.fft2(image)
height, width = fft_image.shape
cy, cx = height // 2, width // 2


# FIND THE MAX IDX OF THE SIDEBAND
x_min, x_max = 0, cx
y_min, y_max = cy, height
roi = cp.abs(fft_image[y_min:y_max, x_min:x_max])
max_index = cp.argmax(roi)
local_max_y, local_max_x = cp.unravel_index(max_index, roi.shape)
global_max_y = y_min + local_max_y
global_max_x = x_min + local_max_x


# ISOLATING THE SIDEBAND
rad_squared = RAD * RAD
circle_mask_kernel(
    fft_image, width, global_max_x, global_max_y, rad_squared, out=fft_image
)


# SHIFTING THE SIDEBAND TO THE DC BAND
shift_y = -global_max_y
shift_x = -global_max_x
fft_centered_image = cp.roll(cp.roll(fft_image, shift_y, axis=0), shift_x, axis=1)


# IFFT
ifft_centered_image = cp.fft.ifft2(fft_centered_image)


# EXTRACTING PHASE
phase_matrix = cp.angle(ifft_centered_image)


# UNWRAPPING
phase_matrix_unwrapped = unwrap_phase(phase_matrix)
height_matrix = (phase_matrix_unwrapped * LAMBDA) / (2 * cp.pi)


# Visualizations
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[0, 0].imshow(image, cmap="gray")
ax[0, 0].set_title("Original Image")
ax[0, 0].axis("off")

ax[0, 1].imshow(cp.asnumpy(cp.log(cp.abs(fft_image) + 1)), cmap="gray")
ax[0, 1].set_title("FFT Magnitude (Original)")
ax[0, 1].axis("off")

ax[0, 2].imshow(cp.asnumpy(cp.log(cp.abs(fft_image) + 1)), cmap="gray")
ax[0, 2].set_title("Masked FFT Magnitude")
ax[0, 2].axis("off")

ax[1, 0].imshow(cp.asnumpy(cp.log(cp.abs(fft_centered_image) + 1)), cmap="gray")
ax[1, 0].set_title("Centered FFT Magnitude")
ax[1, 0].axis("off")

ax[1, 1].imshow(cp.asnumpy(phase_matrix), cmap="jet")
ax[1, 1].set_title("Phase (Before Unwrapping)")
ax[1, 1].axis("off")

ax[1, 2].imshow(cp.asnumpy(phase_matrix_unwrapped), cmap="jet")
ax[1, 2].set_title("Unwrapped Phase")
ax[1, 2].axis("off")

plt.tight_layout()
plt.show()


# 3D VISUALIZATION
plot_surface(cp.asnumpy(phase_matrix_unwrapped) * 20)
