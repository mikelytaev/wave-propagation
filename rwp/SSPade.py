__author__ = 'Mikhail'

from cmath import *
import numpy as np
import scipy.linalg as la
from mpmath import *
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from functools import partial
from rwp.WPDefs import *
from itertools import zip_longest
import numpy.polynomial.polynomial as poly
from sympy import symbols, Matrix
import math as fm
import cmath as cm
from transforms.fcc import FCCFourier


class PadePropagator:
    n_x, n_z = 0, 0

    def __init__(self, env: EMEnvironment, wave_length=1.0, pade_order=(1, 2), dx_wl=100, dz_wl=1):
        self.k0 = (2 * pi) / wave_length
        self.dz = dz_wl * wave_length
        self.dx = dx_wl * wave_length
        self.env = env
        self.pade_order = pade_order

        mp.dps = 15

        def propagator_func(s):
            return mp.exp(1j*self.k0*self.dx*(mp.sqrt(1+s)-1))

        t = taylor(propagator_func, 0, pade_order[0]+pade_order[1])
        p, q = pade(t, pade_order[0], pade_order[1])
        self.pade_coefs = list(zip_longest([-1/complex(v) for v in polyroots(p[::-1])],
                                           [-1/complex(v) for v in polyroots(q[::-1])], fillvalue=0.0j))

    def _Crank_Nikolson_propagate(self, a, b, het, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        '''
        Performs one Crank-Nikolson propagation step
        :param a: right-hand parameter
        :param b: left-hand parameter
        :param het: heterogeneity vector
        :param initial: initial value vector
        :param lower_bound: lower_bound[0]*u_0 + lower_bound[1]*u_1 = lower_bound[2]
        :param upper_bound: upper_bound[0]*u_{n-1} + upper_bound[1]*u_{n} = upper_bound[2]
        :return:
        '''
        d_2 = 1/cm.pow(self.k0*self.dz, 2) * diags([np.ones(self.n_z-1), -2*np.ones(self.n_z), np.ones(self.n_z-1)], [-1, 0, 1])
        left_matrix = eye(self.n_z) + b*(d_2 + diags([het], [0]))
        right_matrix = eye(self.n_z) + a*(d_2 + diags([het], [0]))
        rhs = right_matrix * initial
        left_matrix[0, 0], left_matrix[0, 1], rhs[0] = lower_bound
        left_matrix[-1, 0], left_matrix[-1, 1], rhs[-1] = upper_bound
        return spsolve(left_matrix, rhs)

    def propagate(self, max_range_m, start_field):
        self.n_x = fm.ceil(max_range_m / self.dx) + 1
        x_grid, self.dx = np.linspace(0, max_range_m, self.n_x, retstep=True)
        self.n_z = fm.ceil((self.env.z_max - 0) / self.dz) + 1
        z_grid, self.dz = np.linspace(0.0, self.env.z_max, self.n_z, retstep=True)
        field = Field(x_grid, z_grid)
        field.field[0, :] = list(map(start_field, z_grid))

        num_roots, den_roots = list(zip(*self.pade_coefs))
        xi = symbols('xi')
        tau = 1.0001
        if max(self.pade_order) == 1:
            matrix_t = Matrix([self.k0**2*(1-xi)/(-num_roots[0] + den_roots[0]*xi)])
            matrix_t0 = Matrix([d2a_n_eq_ba_n(matrix_t.subs(xi, t))])
        else:
            matrix_a = Matrix(np.diag(-np.diag(num_roots), 0) + np.diag(den_roots[0:-1], 1))
            matrix_a[-1, 0] = den_roots[-1]
            matrix_a[0, 1] *= xi
            matrix_b = Matrix(np.diag(np.ones(len(num_roots)), 0) + np.diag(-np.ones(len(num_roots))-1, 1))
            matrix_b[-1, 0] = -1
            matrix_b[0, 1] *= xi
            matrix_b *= self.k0
            matrix_ab = matrix_a**-1*matrix_b
            matrix_p, matrix_j = matrix_ab.jordan_form()
            matrix_p_inv = matrix_p.inv()
            #d2a_n_eq_ba_n(self.dz**2*)

        for x_i, x in enumerate(x_grid[1:], start=1):
            phi = field.field[x_i-1, :]
            for (a, b) in self.pade_coefs:
                phi = self._Crank_Nikolson_propagate(a, b, list(map(partial(self.env.M_profile, x), z_grid)), phi)
            field.field[x_i, :] = phi

        return field


def d2a_n_eq_ba_n(b):
    c1 = (b+2-cm.sqrt(b**2-4*b))/2
    c2 = 1 / c1
    return [c1, c2][abs(c1) > abs(c2)]