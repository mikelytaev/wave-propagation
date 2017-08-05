import numpy as np
import scipy.linalg as la
import mpmath
from rwp.WPDefs import *
from itertools import zip_longest
import math as fm
import cmath as cm
from transforms.fcc import FCCAdaptiveFourier
import logging
import pickle
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from rwp.contfrac import *
from rwp.propagators import *

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class PadePropagator:

    def __init__(self, env: EMEnvironment, wavelength=1.0, pade_order=(1, 2), dx_wl=100, dz_wl=1, tol=1e-11):
        self.env = env
        self.k0 = (2 * cm.pi) / wavelength
        self.n_z = fm.ceil((self.env.z_max - self.env.z_min) / (dz_wl * wavelength)) + 1
        self.z_computational_grid, self.dz = np.linspace(self.env.z_min, self.env.z_max, self.n_z, retstep=True)
        self.dx = dx_wl * wavelength
        self.pade_order = pade_order
        self.tol = tol

        mpmath.mp.dps = 63

        if self.env.N_profile is None and isinstance(self.env.lower_boundary, (ImpedanceBC, TransparentConstBS)) and \
                isinstance(self.env.upper_boundary, (ImpedanceBC, TransparentConstBS)):
            def diff2(s):
                return mpmath.acosh(1 + (self.k0 * self.dz) ** 2 * s / 2) ** 2 / (self.k0 * self.dz) ** 2
        else:
            def diff2(s):
                return s

        def propagator_func(s):
            return mpmath.mp.exp(1j*self.k0*self.dx*(mpmath.mp.sqrt(1+diff2(s))-1))

        t = mpmath.taylor(propagator_func, 0, pade_order[0]+pade_order[1])
        p, q = mpmath.pade(t, pade_order[0], pade_order[1])
        self.pade_coefs = list(zip_longest([-1/complex(v) for v in mpmath.polyroots(p[::-1], maxsteps=200)],
                                           [-1/complex(v) for v in mpmath.polyroots(q[::-1], maxsteps=200)], fillvalue=0.0j))

    def _Crank_Nikolson_propagate(self, a, b, het, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        """
        Performs one Crank-Nikolson propagation step
        :param a: right-hand parameter
        :param b: left-hand parameter
        :param het: heterogeneity vector
        :param initial: initial value vector
        :param lower_bound: lower_bound[0]*u_0 + lower_bound[1]*u_1 = lower_bound[2]
        :param upper_bound: upper_bound[0]*u_{n-1} + upper_bound[1]*u_{n} = upper_bound[2]
        :return:
        """
        return np.array(Crank_Nikolson_propagator(self.k0 * self.dz, a, b, het, initial, lower_bound, upper_bound))
        # d_2 = 1/(self.k0*self.dz)**2 * diags([np.ones(self.n_z-1), -2*np.ones(self.n_z), np.ones(self.n_z-1)], [-1, 0, 1])
        # left_matrix = eye(self.n_z) + b*(d_2 + diags([het], [0]))
        # right_matrix = eye(self.n_z) + a*(d_2 + diags([het], [0]))
        # rhs = right_matrix * initial
        # left_matrix[0, 0], left_matrix[0, 1], rhs[0] = lower_bound
        # left_matrix[-1, -2], left_matrix[-1, -1], rhs[-1] = upper_bound
        # return spsolve(left_matrix, rhs)

    def _calc_nlbc(self, max_range_m, beta, diff_eq_solution):
        n_x = fm.ceil(max_range_m / self.dx) + 1
        num_roots, den_roots = list(zip(*self.pade_coefs))
        m_size = len(self.pade_coefs)
        tau = 1.001
        if max(self.pade_order) == 1:
            def nlbc_transformed(t):
                return diff_eq_solution(self.k0 ** 2 * ((1 - t) / (-num_roots[0] + den_roots[0] * t) - beta) * self.dz ** 2)
        else:
            def nlbc_transformed(t):
                matrix_a = np.diag(den_roots, 0) - np.diag(num_roots[1:], -1)
                matrix_a[0, -1] = -num_roots[0]
                matrix_a[0, 0] *= t
                matrix_b = np.diag(-np.ones(m_size), 0) + np.diag(np.ones(m_size - 1), -1) + 0j
                matrix_b[0, -1] = 1
                matrix_b[0, 0] *= t
                matrix_b -= beta * matrix_a
                w, vr = la.eig(matrix_b, matrix_a, right=True)
                r = np.diag([diff_eq_solution(a * self.k0 ** 2 * self.dz**2) for a in w])
                res = vr.dot(r).dot(la.inv(vr))
                return res.reshape(m_size**2)

        fcca = FCCAdaptiveFourier(2 * fm.pi, -np.arange(0, n_x), rtol=self.tol)

        return (tau**np.repeat(np.arange(0, n_x)[:, np.newaxis], m_size ** 2, axis=1) / (2*fm.pi) *
                fcca.forward(lambda t: nlbc_transformed(tau * cm.exp(1j*t)), 0, 2*fm.pi)).reshape((n_x, m_size, m_size))

    def calc_lower_nlbc(self, max_range_m):
        beta = self.env.n2m1_profile(0, self.env.z_min)
        if isinstance(self.env.lower_boundary, TransparentConstBS):
            return self._calc_nlbc(max_range_m, beta, diff_eq_solution=lambda v: 1 / d2a_n_eq_ba_n(v))
        elif isinstance(self.env.lower_boundary, TransparentLinearBS):
            c = self.k0 ** 2 * self.env.lower_boundary.mu_n2m1 * self.dz ** 3
            return self._calc_nlbc(max_range_m, beta, lambda v: 1 / bessel_ratio(c=c, d=v, j=0, tol=self.tol))

    def calc_upper_nlbc(self, max_range_m):
        beta = self.env.n2m1_profile(0, self.env.z_max)
        if isinstance(self.env.upper_boundary, TransparentConstBS):
            return self._calc_nlbc(max_range_m, beta, diff_eq_solution=lambda v: 1 / d2a_n_eq_ba_n(v))
        elif isinstance(self.env.upper_boundary, TransparentLinearBS):
            c = -self.k0 ** 2 * self.env.upper_boundary.mu_n2m1 * self.dz ** 3
            return self._calc_nlbc(max_range_m, beta, lambda v: bessel_ratio(c=c, d=v, j=0, tol=self.tol))

    def propagate(self, max_range_m, start_field, *, lower_nlbc=np.array([]), upper_nlbc=np.array([]), n_dx_out=1, n_dz_out=1):
        n_x = fm.ceil(max_range_m / self.dx) + 1
        x_computational_grid = np.arange(0, n_x) * self.dx
        field = Field(x_computational_grid[::n_dx_out], self.z_computational_grid[::n_dz_out], precision=self.tol)
        phi = np.array([start_field(a) for a in self.z_computational_grid])
        field.field[0, :] = phi[::n_dz_out]
        phi_0, phi_J = np.zeros((n_x, max(self.pade_order)), dtype=complex), np.zeros((n_x, max(self.pade_order)), dtype=complex)

        if isinstance(self.env.lower_boundary, TransparentBS) and len(lower_nlbc) < n_x:
            lower_nlbc = self.calc_lower_nlbc(max_range_m)
        if isinstance(self.env.upper_boundary, TransparentBS) and len(upper_nlbc) < n_x:
            upper_nlbc = self.calc_upper_nlbc(max_range_m)

        for x_i, x in enumerate(x_computational_grid[1:], start=1):
            phi[self.env.impediment(x, self.z_computational_grid)] = 0
            het = self.env.n2m1_profile(x, self.z_computational_grid)
            if isinstance(self.env.lower_boundary, TransparentBS):
                lower_convolution = np.einsum('ijk,ik->j', lower_nlbc[1:x_i], phi_0[x_i-1:0:-1])
            if isinstance(self.env.upper_boundary, TransparentBS):
                upper_convolution = np.einsum('ijk,ik->j', upper_nlbc[1:x_i], phi_J[x_i-1:0:-1])
            for pc_i, (a, b) in enumerate(self.pade_coefs):
                if isinstance(self.env.lower_boundary, TransparentBS):
                    lower_bound = -lower_nlbc[0, pc_i, pc_i], 1, lower_convolution[pc_i] + lower_nlbc[0, pc_i].dot(phi_0[x_i])
                elif isinstance(self.env.lower_boundary, ImpedanceBC):
                    alpha1, alpha2 = self.env.lower_boundary.alpha1, self.env.lower_boundary.alpha2
                    gamma0 = 2 * (-alpha1 + alpha2 * self.dz) / (self.k0 * self.dz)**2 + alpha1*het[0]
                    gamma1 = 2 * alpha1 / (self.k0 * self.dz)**2 + alpha1*het[0]
                    lower_bound = alpha1 + b * gamma0, alpha1 + b * gamma1, \
                                  (alpha1 + a * gamma0) * phi[0] + (alpha1 + a * gamma1) * phi[1]

                if isinstance(self.env.upper_boundary, TransparentBS):
                    upper_bound = 1, -upper_nlbc[0, pc_i, pc_i], upper_convolution[pc_i] + upper_nlbc[0, pc_i].dot(phi_J[x_i])
                elif isinstance(self.env.upper_boundary, ImpedanceBC):
                    upper_bound = 0, 1, 0

                phi = self._Crank_Nikolson_propagate(a, b, het, phi, lower_bound=lower_bound, upper_bound=upper_bound)
                phi_0[x_i, pc_i], phi_J[x_i, pc_i] = phi[0], phi[-1]

            if divmod(x_i, n_dx_out)[1] == 0:
                field.field[divmod(x_i, n_dx_out)[0], :] = phi[::n_dz_out]
                logging.debug('SSPade propagation x = ' + str(x))

        return field


class NLBCManager:

    def __init__(self, name='nlbc'):
        self.file_name = name
        import os
        if os.path.exists(self.file_name):
            with open(self.file_name, 'rb') as f:
                self.nlbs_dict = pickle.load(f)
        else:
            self.nlbs_dict = {}

    def get_NLBC(self, propagator: PadePropagator, max_range_m):
        n_x = fm.ceil(max_range_m / propagator.dx) + 1
        lower_nlbc, upper_nlbc = None, None
        if isinstance(propagator.env.lower_boundary, TransparentBS):
            beta = propagator.env.n2m1_profile(0, propagator.env.z_min)
            q = 'lower', propagator.k0, propagator.dx, propagator.dz, propagator.pade_order, beta, propagator.env.lower_boundary
            if q not in self.nlbs_dict or self.nlbs_dict[q].shape[0] < n_x:
                self.nlbs_dict[q] = propagator.calc_lower_nlbc(max_range_m)
            lower_nlbc = self.nlbs_dict[q]

        if isinstance(propagator.env.upper_boundary, TransparentBS):
            beta = propagator.env.n2m1_profile(0, propagator.env.z_max)
            q = 'upper', propagator.k0, propagator.dx, propagator.dz, propagator.pade_order, beta, propagator.env.upper_boundary
            if q not in self.nlbs_dict or self.nlbs_dict[q].shape[0] < n_x:
                self.nlbs_dict[q] = propagator.calc_upper_nlbc(max_range_m)
            upper_nlbc = self.nlbs_dict[q]

        with open(self.file_name, 'wb') as f:
            pickle.dump(self.nlbs_dict, f)

        return lower_nlbc, upper_nlbc



def d2a_n_eq_ba_n(b):
    c1 = (b+2-cm.sqrt(b**2+4*b))/2
    c2 = 1.0 / c1
    return [c1, c2][abs(c1) > abs(c2)]


# def bessel_ratio(c, d, j, tol):
#     return lentz(lambda n: (-1)**(n+1) * 2.0 * (j + (2.0 + d) / c + n - 1) * (c / 2.0))


def lentz(cont_frac_seq, tol=1e-20):
    """
    Lentz W. J. Generating Bessel functions in Mie scattering calculations using continued fractions
    //Applied Optics. – 1976. – 15. – №. 3. – P. 668-671.
    :param cont_frac_seq: continued fraction sequence
    :param tol: absolute tolerance
    """
    num = cont_frac_seq(2) + 1.0 / cont_frac_seq(1)
    den = cont_frac_seq(2)
    y = cont_frac_seq(1) * num / den
    i = 3
    while abs(num / den - 1) > tol:
        num = cont_frac_seq(i) + 1.0 / num
        den = cont_frac_seq(i) + 1.0 / den
        y = y * num / den
        i += 1

    return y