from copy import deepcopy

import numpy as np


class Field:
    def __init__(self, x_grid, z_grid, freq_hz, precision=1e-6):
        self.x_grid, self.z_grid = x_grid, z_grid
        self.field = np.zeros((x_grid.size, z_grid.size), dtype=complex)
        self.freq_hz = freq_hz
        self.precision = precision

    def value(self, x, z):
        return self.field[abs(self.x_grid - x).argmin(), abs(self.z_grid - z).argmin()]

    def path_loss(self, gamma=0):
        res = deepcopy(self)
        wavelength = 3e8 / self.freq_hz
        res.field = -20*np.log10(abs(self.field + 2e-16)) + 20*np.log10(4*np.pi) + \
             10*np.tile(np.log10(self.x_grid+2e-16), (self.z_grid.shape[0], 1)).transpose() - 30*np.log10(wavelength) + \
                    gamma * np.tile(self.x_grid, (self.z_grid.shape[0], 1)).transpose() * 1e-3

        return res


class Field3d:

    def __init__(self, x_grid, y_grid, z_grid):
        self.x_grid, self.y_grid, self.z_grid = x_grid, y_grid, z_grid
        self.field = np.zeros((x_grid.size, y_grid.size, z_grid.size), dtype=complex)