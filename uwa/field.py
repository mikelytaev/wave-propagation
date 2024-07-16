from enum import Enum

from propagators.field import *
import numpy as np


class AcousticPressureField:

    class APFType(Enum):
        EXACT = 0,
        MEAN_SD = 1

    def __init__(self, x_grid, z_grid, freq_hz, field=None, mean_field=None, sd_field=None):
        self.x_grid, self.z_grid = x_grid, z_grid
        self.freq_hz = freq_hz
        if field is not None:
            self.field = field
            self.type = AcousticPressureField.APFType.EXACT
        elif mean_field is not None and sd_field is not None:
            self.field = mean_field
            self.sd_field = sd_field
            self.type = AcousticPressureField.APFType.MEAN_SD
        else:
            raise Exception('Unsupported field state')

    def nearest_value(self, x_m: float, z_m: float):
        x_i = abs(self.x_grid - x_m).argmin()
        z_i = abs(self.z_grid - z_m).argmin()
        return self.x_grid[x_i], self.z_grid[z_i], self.field[x_i, z_i]
