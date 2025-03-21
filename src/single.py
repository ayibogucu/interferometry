import numpy as np
from skimage.restoration import unwrap_phase

import matplotlib.pyplot as plt
from PIL import Image

import lib.plot

RAD = 50
LAMBDA = 671e-9

image = Image.open("./data/-1.jpg")
fft_image = np.fft.fft2(image)

height, width = fft_image.shape
cy, cx = height // 2, width // 2

# Define ROI boundaries
x_min, x_max = 0, cx
y_min, y_max = cy, height

# Crop the magnitude image to the ROI
roi = np.abs(fft_image[y_min:y_max, x_min:x_max])

# Find the coordinates of the maximum value in the ROI
local_max_y, local_max_x = np.unravel_index(np.argmax(roi), roi.shape)

# Map back to the full image coordinates
global_max_y = y_min + local_max_y
global_max_x = x_min + local_max_x

# Create a mask with a circle at the detected maximum
y, x = np.ogrid[:height, :width]
dist_sq = (y - global_max_y) ** 2 + (x - global_max_x) ** 2
fft_image[dist_sq > RAD**2] = 0

# Compute the shift needed to move the sideband to the center
shift_y = -global_max_y
shift_x = -global_max_x

# Shift the FFT to center the sideband
fft_centered_image = np.roll(np.roll(fft_image, shift_y, axis=0), shift_x, axis=1)

# Perform inverse FFT
ifft_centered_image = np.fft.ifft2(fft_centered_image)

# Calculate the phase and unwrap it
phase_matrix = np.angle(ifft_centered_image)
phase_matrix_unwrapped = unwrap_phase(phase_matrix)
height_matrix = phase_matrix_unwrapped * LAMBDA * 0.15915494309189535

# Visualizations
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# Display the original image
ax[0, 0].imshow(image, cmap="gray")
ax[0, 0].set_title("Original Image")
ax[0, 0].axis("off")

# Display the FFT magnitude (before masking)
ax[0, 1].imshow(np.log(np.abs(fft_image) + 1), cmap="gray")
ax[0, 1].set_title("FFT Magnitude (Original)")
ax[0, 1].axis("off")

# Display the center-shifted FFT magnitude
ax[1, 0].imshow(np.log(np.abs(fft_centered_image) + 1), cmap="gray")
ax[1, 0].set_title("Centered FFT Magnitude")
ax[1, 0].axis("off")

# Display the phase image (before unwrapping)
ax[1, 1].imshow(phase_matrix, cmap="jet")
ax[1, 1].set_title("Phase (Before Unwrapping)")
ax[1, 1].axis("off")

# Display the unwrapped phase image
ax[1, 2].imshow(phase_matrix_unwrapped, cmap="jet")
ax[1, 2].set_title("Unwrapped Phase")
ax[1, 2].axis("off")

plt.tight_layout()
plt.show()

# Display the 3d surface plot
lib.plot.plot_mesh(phase_matrix_unwrapped)
