from copy import deepcopy

import numpy as np
import cmath as cm
import math as fm


class Field:
    def __init__(self, x_grid, z_grid, freq_hz, prop_factor=None, v_func=None, precision=1e-6):
        self.x_grid, self.z_grid = x_grid, z_grid
        self.freq_hz = freq_hz
        self.wavelength = 3e8 / self.freq_hz
        self.precision = precision
        self.log10 = False
        if prop_factor is not None:
            self.field = prop_factor - np.tile(10 * np.log10(self.x_grid), (self.z_grid.shape[0], 1)).T
            self.field -= 10 * fm.log10(self.wavelength)
            self.field /= 2
            self.field = np.nan_to_num(self.field)
            self.log10 = True
        elif v_func is not None:
            self.field = v_func / np.tile(self.x_grid, (self.z_grid.shape[0], 1)).T
        else:
            self.field = np.zeros((x_grid.size, z_grid.size), dtype=complex)

    def value(self, x, z):
        return self.field[abs(self.x_grid - x).argmin(), abs(self.z_grid - z).argmin()]

    def horizontal(self, z):
        return self.field[:, abs(self.z_grid - z).argmin()]

    def horizontal_over_terrain(self, z0, terrain):
        f = np.zeros(len(self.x_grid), dtype=complex)
        for i in range(0, len(self.x_grid)):
            f[i] = self.field[i, abs(self.z_grid - terrain.elevation(self.x_grid[i]) - z0).argmin()]
        return f

    def path_loss(self, gamma=0):
        res = deepcopy(self)
        wavelength = 3e8 / self.freq_hz
        res.field = -20*np.log10(abs(self.field + 2e-16)) + 20*np.log10(4*np.pi) + \
             10*np.tile(np.log10(self.x_grid+2e-16), (self.z_grid.shape[0], 1)).transpose() - 30*np.log10(wavelength) + \
                    gamma * np.tile(self.x_grid, (self.z_grid.shape[0], 1)).transpose() * 1e-3

        return res

    def v_func(self):
        res = deepcopy(self)
        wavelength = 3e8 / self.freq_hz
        if self.log10:
            res.field = self.field + 10 * np.tile(np.log10(self.x_grid + 2e-16), (self.z_grid.shape[0], 1)).transpose()
        else:
            res.field = 10*np.log10(abs(self.field + 2e-16)) + \
                        10*np.tile(np.log10(self.x_grid+2e-16), (self.z_grid.shape[0], 1)).transpose()

        return res


class Field3d:

    def __init__(self, x_grid, y_grid, z_grid):
        self.x_grid, self.y_grid, self.z_grid = x_grid, y_grid, z_grid
        self.field = np.zeros((x_grid.size, y_grid.size, z_grid.size), dtype=complex)


def areps_PF_ascii_reader(file_path):
    pass