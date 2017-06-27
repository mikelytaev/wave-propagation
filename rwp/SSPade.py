__author__ = 'Mikhail'

from math import *

import numpy as np
import scipy.linalg as la
from mpmath import *

from rwp.WPDefs import *


class PadePropagator:

    def __init__(self, env: EMEnvironment, wave_length: float, pade_order):
        self.k0 = (2 * pi) / wave_length
        self.dz = wave_length
        self.dx = 100 * wave_length
        self.env = env
        self.pade_order = pade_order

        def propagator_func(s):
            exp(j1*self.k0*self.dx*(sqrt(1+s)-1))

        t = taylor(propagator_func, 0, 2*pade_order-1)
        p, q = pade(t, pade_order-1, pade_order)
        self.a_coefs, self.b_coefs = -1/np.roots(p) + [0.0], -1/np.roots(q)

    def Crank_Nikolson_propagate(a, b, het, rhs):
        la.solve_banded((1, 1), )

    def propagate(self, max_range_m, start_field):
        x_grid = np.linspace(0, max_range_m, ceil(max_range_m / self.dx))
        self.dx = x_grid[1] - x_grid[0]
        z_grid = np.linspace(0, self.env.z_max, ceil((self.env.z_max - 0) / self.dz))
        self.dz = z_grid[1] - z_grid[0]
        field = Field(x_grid, z_grid)
        field.field[0,:] = map(start_field, z_grid)

        for x_i, x in enumerate(x_grid[1:], start=1):
            phi = field.field[x_i-1,:]
            for l in range(0, p):


