import numpy as np
import logging


def filon_trapezoidal_ft(f, z_values, spectral_values):
    z1 = np.tile(z_values[:-1:], (f.shape[1], 1)).T
    z2 = np.tile(z_values[1::], (f.shape[1], 1)).T
    f1 = f[:-1:]
    f2 = f[1::]
    res = np.empty((len(spectral_values), f.shape[1]), dtype=complex)
    for l_i, l in enumerate(spectral_values):
        logging.debug(l_i)
        if abs(l) > 1e-13:
            res[l_i,:] = sum((np.exp(-1j * z1 * l) * (f2 + f1 * (-1j * z1 * l + 1j * z2 * l - 1)) +
                            np.exp(-1j * z2 * l) * (f1 + 1j * f2 * (z1 * l - z2 * l + 1j))) / (l ** 2 * (z1 - z2)))
        else:
            res[l_i,:] = sum(1/2 * (f1 + f2) * (z2 - z1))

    return res