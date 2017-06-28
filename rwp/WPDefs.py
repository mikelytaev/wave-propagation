__author__ = 'Mikhail'

from enum import Enum
import numpy as np
from cmath import *


class Field:
    def __init__(self, x_grid, z_grid):
        self.x_grid, self.z_grid = x_grid, z_grid
        self.field = np.zeros((x_grid.size, z_grid.size))*(1+1j)


class EMEnvironment:
    class BoundaryType(Enum):
        IMPEDANCE = 0  # impedance boundary (alpha1*u(z)+alpha2*u'(z))_{z=0}=0
        CONST = 1  # constant refractive index in outer domain
        LIN = 2  # linear refractive index in outer domain
    lower_boundary_type = BoundaryType.IMPEDANCE
    upper_boundary_type = BoundaryType.CONST
    alpha1 = 1  # lower boundary
    alpha2 = 0

    z_max = 100.0

    def M_profile(self, x, z):
        return 0

    def terrain(self, x):
        return 0


def gauss_source(k0, z_s, beam_width, eval_angle):
    beam_width = beam_width * pi / 180
    eval_angle = eval_angle * pi / 180
    return lambda z: k0*beam_width/(2*sqrt(pi)*log10(2))*exp(-1j*k0*eval_angle)*exp(-pow(beam_width*k0*(z-z_s), 2)/(8*log10(2)))