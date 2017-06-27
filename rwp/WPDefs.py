from enum import Enum
import numpy as np


class Field:
    def __init__(self, r_grid, z_grid):
        self.r_grid, self.z_grid = r_grid, z_grid
        self.field = np.zeros((r_grid.size, z_grid.size))*(1+1j)


class EMEnvironment:
    class BoundaryType(Enum):
        IMPEDANCE = 0  # impedance boundary (alpha1*u(z)+alpha2*u'(z))_{z=0}=0
        CONST = 1  # constant refractive index in outer domain
        LIN = 2  # linear refractive index in outer domain
    lower_boundary_type = BoundaryType.IMPEDANCE
    upper_boundary_type = BoundaryType.CONST
    alpha1 = 1  # lower boundary
    alpha2 = 0

    z_max = 100

    def M_profile(self, x, z):
        return 0

    def terrain(self, x):
        return 0

