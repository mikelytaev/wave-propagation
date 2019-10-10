from propagators.field import *
import numpy as np


class AcousticPressureField(Field2d):

    def __init__(self, x_grid, z_grid, freq_hz, field=None):
        super().__init__(x_grid, z_grid, freq_hz, field)
