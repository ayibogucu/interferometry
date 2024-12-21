import numpy as np


def OPD(phase_batch, wavelength):
    return np.multiply(phase_batch, wavelength) / (2 * np.pi)
