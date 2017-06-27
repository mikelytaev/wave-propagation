__author__ = 'Mikhail'

from math import *
import numpy as np
import scipy.linalg as la
from mpmath import *
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from functools import partial
from rwp.WPDefs import *


class PadePropagator:
    n_x, n_z = 0, 0

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


    def Crank_Nikolson_propagate(self, a, b, het, rhs):
        d_2 = 1/pow(self.k0*self.dz, 2) * diags([np.ones(self.n_z-1), -2*np.ones(self.n_z), np.ones(self.n_z-1)], [-1, 0, 1])
        left_matrix = eye(self.n_z) + b*(diags([het], [0]))
        right_matrix = eye(self.n_z) + a*(diags([het], [0]))
        return spsolve(left_matrix, right_matrix * rhs)


    def propagate(self, max_range_m, start_field):
        self.n_x = ceil(max_range_m / self.dx)
        x_grid, self.dx = np.linspace(0, max_range_m, self.n_x, retstep=True)
        self.n_z = ceil((self.env.z_max - 0) / self.dz)
        z_grid, self.dz = np.linspace(0, self.env.z_max, self.n_z, retstep=True)
        field = Field(x_grid, z_grid)
        field.field[0,:] = map(start_field, z_grid)

        for x_i, x in enumerate(x_grid[1:], start=1):
            phi = field.field[x_i-1, :]
            for (a, b) in zip(self.a_coefs, self.b_coefs):
                phi = self.Crank_Nikolson_propagate(a, b, map(partial(self.env.M_profile, x), z_grid))
            field.field[x_i, :] = phi

