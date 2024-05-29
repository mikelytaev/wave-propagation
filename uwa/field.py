from propagators.field import *
import numpy as np


class AcousticPressureField(Field2d):

    def __init__(self, x_grid, z_grid, freq_hz, field=None):
        super().__init__(x_grid, z_grid, freq_hz, field)

    def nearest_value(self, x_m: float, z_m: float):
        x_i = abs(self.x_grid - x_m).argmin()
        z_i = abs(self.z_grid - z_m).argmin()
        return self.x_grid[x_i], self.z_grid[z_i], self.field[x_i, z_i]
