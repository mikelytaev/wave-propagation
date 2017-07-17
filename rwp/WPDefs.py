import numpy as np
from cmath import *


__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'

class Field:
    def __init__(self, x_grid, z_grid):
        self.x_grid, self.z_grid = x_grid, z_grid
        self.field = np.zeros((x_grid.size, z_grid.size))*(1+1j)


class ImpedanceBC:
    """
    Impedance boundary (alpha1*u(z)+alpha2*u'(z))_{z=0}=0
    """
    def __init__(self, alpha1, alpha2):
        self.alpha1 = alpha1
        self.alpha2 = alpha2


class TransparentConstBS:
    """
    Constant refractive index in outer domain
    """


class TransparentLinearBS:
    """
    Linear refractive index in outer domain
    """
    def __init__(self, mu_N):
        self.mu_n2 = 1 + 2 * mu_N * 1e-6


class EMEnvironment:
    lower_boundary = ImpedanceBC(1, 0)
    upper_boundary = TransparentConstBS()
    z_max = 100.0

    def __init__(self):
        self.N_profile = lambda x, z: 0
        self.terrain = lambda x: x * 0

    def n2_profile(self, x, z):
        return 1 + 2 * self.N_profile(x, z) * 1e-6

    def n_profile(self, x, z):
        1 + self.N_profile * 1e-6


def gauss_source(k0, z_s, beam_width, eval_angle):
    beam_width = beam_width * pi / 180
    eval_angle = eval_angle * pi / 180
    return lambda z: k0*beam_width/(2*sqrt(pi)*log10(2))*exp(-1j*k0*eval_angle)*exp(-pow(beam_width*k0*(z-z_s), 2)/(8*log10(2)))