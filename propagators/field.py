import numpy as np


class Field2d:

    def __init__(self, x_grid, z_grid, freq_hz, field=None):
        self.x_grid, self.z_grid = x_grid, z_grid
        self.freq_hz = freq_hz
        if field is None:
            self.field = np.zeros((x_grid.size, z_grid.size), dtype=complex)
        else:
            self.field = field

    def value(self, x, z):
        pass