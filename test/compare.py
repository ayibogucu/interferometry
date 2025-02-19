import numpy as np
from scipy.io import loadmat
from cv2 import minMaxLoc
import lib.plot
import matplotlib.pyplot as plt

python_array = np.load("./results/100.npy")
matlab_array = -loadmat("./results/unwrapped_phase.mat")["s"]
difference = np.abs(python_array - matlab_array)

mse = (difference**2).mean()
psnr = 10 * np.log10(np.max(matlab_array) ** 2 / mse)

min_val, max_val, min_loc, max_loc = minMaxLoc(difference)

print(f"MSE: {mse},  PSNR: {psnr}")
print(f"minimum difference is {min_val} at location {min_loc}")
print(f"maximum difference is {max_val} at location {max_loc}")


fig, ax = plt.subplots(2, 2, figsize=(15, 15))

ax[0, 0].imshow(python_array, cmap="jet")
ax[0, 0].set_title("Python Image")
ax[0, 0].axis("off")

ax[0, 1].imshow(matlab_array, cmap="jet")
ax[0, 1].set_title("Matlab Image")
ax[0, 1].axis("off")

cax = ax[1, 0].imshow(difference, cmap="jet", vmin=0, vmax=np.max(difference))
ax[1, 0].set_title("Difference Absolute")
ax[1, 0].axis("off")

fig.colorbar(cax, ax=ax[1, 0], orientation="vertical", fraction=0.02, pad=0.04)

ax[1, 1].axis("off")

plt.show()

lib.plot.plot_mesh(difference)
