from skimage.restoration import unwrap_phase
import numpy as np


def quality_guided(phase_batch):
    unwrapped_phase_batch = np.apply_along_axis(unwrap_phase, 1, phase_batch)
    return unwrapped_phase_batch
