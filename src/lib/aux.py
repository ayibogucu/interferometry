import cv2
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
# from mpl_toolkits.mplot3d import Axes3D
# import cucim


# TODO: LOOK AT CUCIM WITH NDIVIA DGS APIS AND NOT USE CV2 FOR THIS.
def tiff_to_array_batch_gpu(path_batch):
    first_image = cv2.imread(path_batch[0], cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        raise FileNotFoundError(f"Unable to load image: {path_batch[0]}")

    depth = len(path_batch)
    height, width = first_image.shape
    array_batch = cp.empty((depth, height, width), dtype=first_image.dtype)

    array_batch[0] = cp.array(first_image)

    for i, path in enumerate(path_batch[1:], start=1):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")

        array_batch[i] = cp.array(image)

    return array_batch


def tiff_to_array_batch(path_batch):
    first_image = cv2.imread(path_batch[0], cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        raise FileNotFoundError(f"Unable to load image: {path_batch[0]}")

    depth = len(path_batch)
    height, width = first_image.shape
    array_batch = np.empty((depth, height, width), dtype=first_image.dtype)

    for i, path in enumerate(path_batch):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")
        array_batch[i] = image
    return array_batch


def get_files_recursive(directory):
    all_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))

    return all_files


def plot_batch(image_list, figsize=(6, 6), cmap="gray", titles=None):
    """
    Plots a series of 2D numpy arrays (images) interactively, showing one image at a time.

    Parameters:
        image_list (list of np.ndarray): List of 2D numpy arrays to be plotted.
        figsize (tuple): Figure size for the matplotlib figure.
        cmap (str): Colormap to be used for plotting images (default is 'gray').
        titles (list of str): Optional list of titles for each image.

    Raises:
        ValueError: If image_list is empty or if any element is not a 2D numpy array.
    """

    # Initialize state
    current_index = [0]  # Use a mutable object to track index across button callbacks
    n_images = len(image_list)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.2)  # Leave space for buttons

    # Display the initial image
    img_display = ax.imshow(image_list[current_index[0]], cmap=cmap)
    ax.axis("off")
    title = ax.set_title(titles[current_index[0]] if titles else "")

    # Button click event handlers
    def next_image(event):
        current_index[0] = (current_index[0] + 1) % n_images
        update_image()

    def prev_image(event):
        current_index[0] = (current_index[0] - 1) % n_images
        update_image()

    def update_image():
        img_display.set_data(image_list[current_index[0]])
        title.set_text(titles[current_index[0]] if titles else "")
        fig.canvas.draw_idle()

    # Add buttons
    axprev = plt.axes([0.2, 0.05, 0.2, 0.075])  # Position for previous button
    axnext = plt.axes([0.6, 0.05, 0.2, 0.075])  # Position for next button

    bnext = Button(axnext, "Next")
    bprev = Button(axprev, "Previous")

    bnext.on_clicked(next_image)
    bprev.on_clicked(prev_image)

    plt.show()


def plot_batch_3d_heightmaps(heightmaps):
    """
    Plots a batch of 3D heightmaps interactively, showing one heightmap at a time.

    Parameters:
        heightmaps (list of 2D np.ndarray): A list of 2D arrays representing the heightmaps.

    Raises:
        ValueError: If heightmaps is empty or if any element is not a 2D numpy array.
    """
    # Initialize state
    current_index = [0]
    n_heightmaps = len(heightmaps)

    # Create the figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.2, right=0.8)  # Leave space for buttons and colorbar

    # Create a separate axis for the colorbar
    cbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.4])  # Position for the colorbar

    # Generate the meshgrid for coordinates (use the first heightmap as reference)
    rows, cols = heightmaps[0].shape
    x = np.linspace(-cols / 2, cols / 2, cols)
    y = np.linspace(-rows / 2, rows / 2, rows)
    x, y = np.meshgrid(x, y)

    # Display the initial heightmap
    surface = [
        ax.plot_surface(
            x, y, heightmaps[current_index[0]], cmap="viridis", edgecolor="none"
        )
    ]
    title = ax.set_title(f"3D Heightmap {current_index[0] + 1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height (Z)")
    colorbar = fig.colorbar(surface[0], cax=cbar_ax)
    colorbar.set_label("Height")

    # Button click event handlers
    def next_heightmap(event):
        current_index[0] = (current_index[0] + 1) % n_heightmaps
        update_heightmap()

    def prev_heightmap(event):
        current_index[0] = (current_index[0] - 1) % n_heightmaps
        update_heightmap()

    def update_heightmap():
        # Clear the axis completely
        ax.clear()

        # Plot the new heightmap
        surface[0] = ax.plot_surface(
            x, y, heightmaps[current_index[0]], cmap="viridis", edgecolor="none"
        )
        ax.set_title(f"3D Heightmap {current_index[0] + 1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Height (Z)")

        # Update the colorbar
        colorbar.update_normal(surface[0])
        fig.canvas.draw_idle()

    # Add buttons
    axprev = plt.axes([0.2, 0.05, 0.2, 0.075])  # Position for previous button
    axnext = plt.axes([0.6, 0.05, 0.2, 0.075])  # Position for next button

    bnext = Button(axnext, "Next")
    bprev = Button(axprev, "Previous")

    bnext.on_clicked(next_heightmap)
    bprev.on_clicked(prev_heightmap)

    plt.show()
