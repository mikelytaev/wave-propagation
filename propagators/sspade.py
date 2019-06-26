import logging
import pickle

import math as fm
import cmath as cm
from dataclasses import dataclass
from enum import Enum

import mpmath
import numpy as np
from transforms.fcc_fourier import FCCAdaptiveFourier
from scipy import linalg as la

from propagators._utils import *
from propagators.contfrac import bessel_ratio_4th_order
from rwp.field import Field

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from propagators._cn_utils import *


class BoundaryCondition:
    pass


class RobinBC(BoundaryCondition):
    """
    q_{1}u+q_{2}u'=q_3
    """

    def __init__(self, q1: complex, q2: complex, q3: complex):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3


class TransparentBC(BoundaryCondition):
    pass


class TransparentConstBC(TransparentBC):
    pass


class TransparentLinearBC(TransparentBC):
    pass


class DiscreteLocalBC(BoundaryCondition):

    def __init__(self, q1: complex, q2: complex, q3: complex):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3


class DiscreteNonLocalBC(BoundaryCondition):

    def __init__(self, r0: float, r1: float, coefs: "numpy array"):
        self.r0 = r0
        self.r1 = r1
        self.coefs = coefs


class TerrainMethod(Enum):
    """
    Terrain accounting method
    pass_through: transparent boundary and varying refractive index
    staircase: impedance boundary and varying lower boundary height
    """
    no = 1,
    pass_through = 2,
    staircase = 3


class HelmholtzEnvironment:

    def __init__(self, x_max_m, lower_bc, upper_bc):
        self.x_max_m = x_max_m
        self.lower_bc = lower_bc
        self.upper_bc = upper_bc
        self.z_min = 0
        self.z_max = 300
        self.is_homogeneous = False
        self.n2minus1 = lambda x, z, freq_hz: 0
        self.rho = lambda x, z: 1
        self.use_rho = True
        self.terrain = lambda x: self.z_min
        self.knife_edges = []


class HelmholtzPropagatorStorage:

    def __init__(self):
        pass

    def get_lower_nlbc(self, *, k0, dx_wl, dz_wl, pade_order, z_order, sqrt_alpha, spe, beta, gamma, n_x):
        pass

    def set_lower_nlbc(self, *, k0, dx_wl, dz_wl, pade_order, z_order, sqrt_alpha, spe, beta, gamma, nlbc_coefs):
        pass

    def get_upper_nlbc(self, *, k0, dx_wl, dz_wl, pade_order, z_order, sqrt_alpha, spe, beta, gamma, n_x):
        pass

    def set_upper_nlbc(self, *, k0, dx_wl, dz_wl, pade_order, z_order, sqrt_alpha, spe, beta, gamma, nlbc_coefs):
        pass

@dataclass
class HelmholtzPropagatorComputationalParams:
    max_range_m: float
    max_height_m: float
    dx_wl: float
    dz_wl: float
    max_propagation_angle: float
    exp_pade_order: tuple = (1, 1)
    x_output_filter: int = 1
    z_output_filter: int = 1
    two_way: bool = False
    two_way_iter_num: int = 0
    two_way_threshold: float = 0.05
    standard_pe: bool = False
    sqrt_alpha: float = 0
    z_order: int = 4
    terrain_method: TerrainMethod = TerrainMethod.no
    tol: float = 1e-11
    storage: HelmholtzPropagatorStorage = None


@dataclass
class HelmholtzField:
    x_grid_m: np.ndarray = None
    z_grid_m: np.ndarray = None
    field: np.ndarray = None


class HelmholtzPadeSolver:

    def __init__(self, env: HelmholtzEnvironment, wavelength, freq_hz, params: HelmholtzPropagatorComputationalParams):
        self.env = env
        self.params = params
        self.wavelength = wavelength
        self.freq_hz = freq_hz
        self.k0 = (2 * cm.pi) / self.wavelength

        self._optimize_params()
        self.z_computational_grid, self.dz_m = np.linspace(self.env.z_min, self.env.z_max, self.n_z, retstep=True)
        self.dx_m = self.params.dx_wl * wavelength

        if self.z_order == 2:
            self.alpha = 0
        else:
            self.alpha = 1 / 12

        if self.env.is_homogeneous and self.terrain_method == TerrainMethod.staircase:
            def diff2(s):
                return mpmath.acosh(1 + (self.k0 * self.dz_m) ** 2 * s / 2) ** 2 / (self.k0 * self.dz_m) ** 2
        else:
            def diff2(s):
                return s

        self.pade_coefs = pade_propagator_coefs(pade_order=self.params.exp_pade_order, diff2=diff2, k0=self.k0,
                                                dx=self.dx_m, spe=self.params.standard_pe, alpha=self.params.sqrt_alpha)

        self.lower_nlbc = []
        self.upper_nlbc = []

    def _optimize_params(self):
        #optimize max angle
        self.params.max_angle = self.params.max_angle or self._optimal_angle()

        if self.env.is_homogeneous():
            self.params.z_order = float('inf')
            logging.info("using Pade approximation for diff2_z")

        self.params.z_order = self.params.z_order or 4

        (self.params.dx_wl, self.params.dz_wl, self.params.exp_pade_order) = \
            optimal_params(max_angle=self.params.max_angle, threshold=5e-3, dx=self.params.dx_wl, dz=self.params.dz_wl,
                           pade_order=self.params.exp_pade_order, z_order=self.params.z_order)

        if self.params.max_height_m is None:
            self.params.max_height_m = abs(self.env.z_max - self.env.z_min)
        else:
            self.params.max_height_m = max(self.params.max_height_m, abs(self.env.z_max - self.env.z_min))

        x_approx_sampling = 2000
        z_approx_sampling = 1000

        if self.terrain_method == TerrainMethod.pass_through:
            n_g = self.env.ground_material.complex_permittivity(self.freq_hz)
            self.params.dx_wl /= round(abs(cm.sqrt(n_g - 0.1)))
            self.params.dz_wl /= round(abs(cm.sqrt(n_g - 0.1)))

        self.params.dx_wl = min(self.params.dx_wl, self.params.max_range_m / self.wavelength / x_approx_sampling)
        self.params.dz_wl = min(self.params.dz_wl, self.params.max_height_m / self.wavelength / z_approx_sampling)
        self.n_x = fm.ceil(self.max_range_m / self.params.dx_wl / self.wavelength) + 1
        self.n_z = fm.ceil(self.params.max_height_m / (self.params.dz_wl * self.wavelength)) + 1

        self.params.x_output_filter = self.params.x_output_filter or fm.ceil(self.n_x / x_approx_sampling)
        self.params.z_output_filter = self.params.z_output_filter or fm.ceil(self.n_z / z_approx_sampling)

        logging.info("dx = " + str(self.params.dx_wl) + " wavelength")
        logging.info("dz = " + str(self.params.dz_wl) + " wavelength")
        logging.info("Pade order = " + str(self.params.exp_pade_order))

    def _optimal_angle(self):
        if len(self.env.knife_edges) > 0:
            return 85
        else:
            res = 3
            step = 10
            for x in np.arange(step, self.params.max_range_m, step):
                angle = cm.atan((self.env.terrain(x) - self.env.terrain(x - step)) / step) * 180 / cm.pi
                res = max(res, abs(angle))
            res = max(self.src.max_angle(), fm.ceil(res))
            return res

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
        if self.z_order == 2:
            return np.array(Crank_Nikolson_propagator((self.k0 * self.dz_m) ** 2, a, b, het, initial, lower_bound, upper_bound))
        else:
            return self._Crank_Nikolson_propagate_4th_order(a, b, het, initial, lower_bound, upper_bound)
        # d_2 = 1/(self.k0*self.dz)**2 * diags([np.ones(self.n_z-1), -2*np.ones(self.n_z), np.ones(self.n_z-1)], [-1, 0, 1])
        # left_matrix = eye(self.n_z) + b*(d_2 + diags([het], [0]))
        # right_matrix = eye(self.n_z) + a*(d_2 + diags([het], [0]))
        # rhs = right_matrix * initial
        # left_matrix[0, 0], left_matrix[0, 1], rhs[0] = lower_bound
        # left_matrix[-1, -2], left_matrix[-1, -1], rhs[-1] = upper_bound
        # return spsolve(left_matrix, rhs)

    def _Crank_Nikolson_propagate_4th_order(self, a, b, het, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        alpha = 1/12
        c_a = alpha * (self.k0 * self.dz_m) ** 2 + a + alpha * a * (self.k0 * self.dz_m) ** 2 * het
        c_b = alpha * (self.k0 * self.dz_m) ** 2 + b + alpha * b * (self.k0 * self.dz_m) ** 2 * het
        d_a = (self.k0 * self.dz_m) ** 2 * (1 - 2 * alpha) - 2 * a + a * (self.k0 * self.dz_m) ** 2 * het - 2 * a * alpha * (self.k0 * self.dz_m) ** 2 * het
        d_b = (self.k0 * self.dz_m) ** 2 * (1 - 2 * alpha) - 2 * b + b * (self.k0 * self.dz_m) ** 2 * het - 2 * b * alpha * (self.k0 * self.dz_m) ** 2 * het

        #rhs = tridiag_multiply(c_a[:-1:], d_a, c_a[1::], initial)
        rhs = d_a * initial
        rhs[1::] += c_a[:-1:] * initial[:-1:]
        rhs[:-1:] += c_a[1::] * initial[1::]
        d_b[0] = lower_bound[0]
        d_b[-1] = upper_bound[1]
        diag_1 = np.copy(c_b[1::])
        diag_1[0] = lower_bound[1]
        diag_m1 = np.copy(c_b[:-1:])
        diag_m1[-1] = upper_bound[0]
        rhs[0] = lower_bound[2]
        rhs[-1] = upper_bound[2]
        return np.array(tridiag_method(diag_m1, d_b, diag_1, rhs))
        #return np.array(Crank_Nikolson_propagator_4th_order((self.k0 * self.dz) ** 2, a, b, alpha, het, initial, lower_bound, upper_bound))

    def _Crank_Nikolson_propagate_4th_order_v_pol(self, a, b, het_func, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        alpha = self.alpha

        alpha_m = alpha * (het_func(self.z_computational_grid[1:-1:]) + 1)

        c_a_left = 1 / (het_func(self.z_computational_grid[1:-1:] - self.dz_m / 2) + 1) * \
                   (alpha_m * (self.k0 * self.dz_m) ** 2 + a * (het_func(self.z_computational_grid[1:-1:]) + 1) +
                    a * alpha_m * (self.k0 * self.dz_m) ** 2 * het_func(self.z_computational_grid[:-2:]))

        c_a_right = 1 / (het_func(self.z_computational_grid[1:-1:] + self.dz_m / 2) + 1) * \
                    (alpha_m * (self.k0 * self.dz_m) ** 2 + a * (het_func(self.z_computational_grid[1:-1:]) + 1) +
                     a * alpha_m * (self.k0 * self.dz_m) ** 2 * het_func(self.z_computational_grid[2::]))

        c_b_left = 1 / (het_func(self.z_computational_grid[1:-1:] - self.dz_m / 2) + 1) * \
                   (alpha_m * (self.k0 * self.dz_m) ** 2 + b * (het_func(self.z_computational_grid[1:-1:]) + 1) +
                    b * alpha_m * (self.k0 * self.dz_m) ** 2 * het_func(self.z_computational_grid[:-2:]))

        c_b_right = 1 / (het_func(self.z_computational_grid[1:-1:] + self.dz_m / 2) + 1) * \
                    (alpha_m * (self.k0 * self.dz_m) ** 2 + b * (het_func(self.z_computational_grid[1:-1:]) + 1) +
                     b * alpha_m * (self.k0 * self.dz_m) ** 2 * het_func(self.z_computational_grid[2::]))

        het_mid2 = 1 / (het_func(self.z_computational_grid[1:-1] + self.dz_m / 2) + 1) + \
                   1 / (het_func(self.z_computational_grid[1:-1] - self.dz_m / 2) + 1)

        d_a = (self.k0 * self.dz_m) ** 2 * (1 - alpha_m * het_mid2) - a * het_mid2 * (
                    het_func(self.z_computational_grid[1:-1:]) + 1) + a * (self.k0 * self.dz_m) ** 2 * het_func(
            self.z_computational_grid[1:-1:]) - a * alpha_m * (self.k0 * self.dz_m) ** 2 * het_func(
            self.z_computational_grid[1:-1:]) * het_mid2

        d_b = (self.k0 * self.dz_m) ** 2 * (1 - alpha_m * het_mid2) - b * het_mid2 * (
                het_func(self.z_computational_grid[1:-1:]) + 1) + b * (self.k0 * self.dz_m) ** 2 * het_func(
            self.z_computational_grid[1:-1:]) - b * alpha_m * (self.k0 * self.dz_m) ** 2 * het_func(
            self.z_computational_grid[1:-1:]) * het_mid2

        rhs = np.concatenate(([0], d_a, [0])) * initial
        rhs[1::] += np.concatenate((c_a_left, [0])) * initial[:-1:]
        rhs[:-1:] += np.concatenate(([0], c_a_right)) * initial[1::]
        rhs[0] = lower_bound[2]
        rhs[-1] = upper_bound[2]

        d_b = np.concatenate(([lower_bound[0]], d_b, [upper_bound[1]]))
        c_b_left = np.concatenate((c_b_left, [upper_bound[0]]))
        c_b_right = np.concatenate(([lower_bound[1]], c_b_right))

        return tridiag_method(c_b_left, d_b, c_b_right, rhs)

    def _calc_lower_lbc(self, *, local_bc: LocalBC, a, b, x, z_min, phi):
        q1, q2 = local_bc.q1, local_bc.q2
        r0 = q2 * (self.k0 * self.dz_m) ** 2 * (1 + b * (self.env.n2minus1(x, z_min, self.freq_hz))) + 2 * b * (self.dz_m * q1 - q2)
        r1 = 2 * b * q2
        r2 = q2 * (self.k0 * self.dz_m) ** 2 * (1 + a * (self.env.n2minus1(x, z_min, self.freq_hz))) + 2 * a * (self.dz_m * q1 - q2)
        r3 = 2 * a * q2
        return r0, r1, r2 * phi[0] + r3 * phi[1]

    def prepare_nlbc(self):
        if isinstance(self.env.lower_bc, TransparentBC):
            beta = self.env.n2minus1(0, self.env.z_min - 1)
            gamma = 0
            if self.params.storage:
                self.lower_nlbc = self.params.storage.get_lower_nlbc(k0=self.k0, dx_wl=self.params.dx_wl, dz_wl=self.params.dz_wl,
                                                                     pade_order=self.params.exp_pade_order, z_order=self.params.z_order,
                                                                     sqrt_alpha=self.params.sqrt_alpha, spe=self.params.spe, beta=beta,
                                                                     gamma=gamma, n_x=self.n_x)

                if self.lower_nlbc is None:
                    self.lower_nlbc = self.calc_lower_nlbc(beta)
                    self.params.storage.set_lower_nlbc(k0=self.k0, dx_wl=self.params.dx_wl, dz_wl=self.params.dz_wl,
                                                       pade_order=self.params.exp_pade_order,
                                                       z_order=self.params.z_order,
                                                       sqrt_alpha=self.params.sqrt_alpha, spe=self.params.spe,
                                                       beta=beta,
                                                       gamma=gamma, nlbc_coefs=self.lower_nlbc)
            else:
                self.lower_nlbc = self.calc_lower_nlbc(beta)

        if isinstance(self.env.upper_bc, TransparentBC):
            beta = self.env.n2minus1(0, self.env.z_max + 1)
            gamma = self.env.n2minus1(0, self.env.z_max + 1) - self.env.n2minus1(0, self.env.z_max)
            if self.params.storage:
                self.upper_nlbc = self.params.storage.get_upper_nlbc(k0=self.k0, dx_wl=self.params.dx_wl, dz_wl=self.params.dz_wl,
                                                                     pade_order=self.params.exp_pade_order, z_order=self.params.z_order,
                                                                     sqrt_alpha=self.params.sqrt_alpha, spe=self.params.spe, beta=beta,
                                                                     gamma=gamma, n_x=self.n_x)

                if self.upper_nlbc is None:
                    self.upper_nlbc = self.calc_upper_nlbc(beta, gamma)
                    self.params.storage.set_upper_nlbc(k0=self.k0, dx_wl=self.params.dx_wl, dz_wl=self.params.dz_wl,
                                                       pade_order=self.params.exp_pade_order,
                                                       z_order=self.params.z_order,
                                                       sqrt_alpha=self.params.sqrt_alpha, spe=self.params.spe,
                                                       beta=beta,
                                                       gamma=gamma, nlbc_coefs=self.lower_nlbc)
            else:
                self.upper_nlbc = self.calc_upper_nlbc(beta, gamma)


    def _calc_nlbc(self, diff_eq_solution_ratio):
        num_roots, den_roots = list(zip(*self.pade_coefs))
        m_size = len(self.pade_coefs)
        tau = 1.001
        if max(self.pade_order) == 1:
            def nlbc_transformed(t):
                return diff_eq_solution_ratio(((1 - t) / (-num_roots[0] + den_roots[0] * t)))
        else:
            def nlbc_transformed(t):
                ang = t / cm.pi
                t = tau * cm.exp(1j*t)
                matrix_a = np.diag(den_roots, 0) - np.diag(num_roots[1:], -1)
                matrix_a[0, -1] = -num_roots[0]
                matrix_a[0, 0] *= t
                matrix_b = np.diag(-np.ones(m_size), 0) + np.diag(np.ones(m_size - 1), -1) + 0j
                matrix_b[0, -1] = 1
                matrix_b[0, 0] *= t
                w, vr = la.eig(matrix_b, matrix_a, right=True)
                r = np.diag([diff_eq_solution_ratio(a, t) for a in w])
                res = vr.dot(r).dot(la.inv(vr))
                return res.reshape(m_size**2)

        int_eps = 0#1e-8
        fcca = FCCAdaptiveFourier(2 * fm.pi - 2 * int_eps, -np.arange(0, self.n_x), rtol=self.tol)

        coefs = (tau**np.repeat(np.arange(0, self.n_x)[:, np.newaxis], m_size ** 2, axis=1) / (2*fm.pi) *
                fcca.forward(lambda t: nlbc_transformed(t), -fm.pi, fm.pi)).reshape((self.n_x, m_size, m_size))

        return NonLocalBC(r0=1, r1=1, coefs=coefs)

    def calc_lower_nlbc(self, beta):
        logging.debug('Computing lower nonlocal boundary condition...')
        alpha = self.alpha

        def diff_eq_solution_ratio(s, xi):
            k_d = self.wavelength / 2 / self.dx_m
            k_x = k_d * self.k0 + 1j / self.dx_m * cm.log(xi)
            k_z = cm.sqrt(self.k0**2 - k_x ** 2)
            yy = cm.sqrt(self.k0**2 * (beta + 1) - k_x ** 2)
            refl_coef = (yy - (beta+1) * k_z) / (yy + (beta+1) * k_z)

            theta = 30
            refl_coef = cm.exp(-10*((k_x - self.k0 * cm.cos(theta * cm.pi / 180))) ** 2)
            # if abs(refl_coef) > 0.4:
            #     print("refl_coef = " + str(refl_coef) + "k_x / k = " + str(k_x / self.k0))

            #beta = 0
            a_m1 = 1 - alpha * (self.k0 * self.dz_m) ** 2 * (s - 0)
            a_1 = 1 - alpha * (self.k0 * self.dz_m) ** 2 * (s - 0)
            c = -2 + (2 * alpha - 1) * (self.k0 * self.dz_m) ** 2 * (s - 0)

            mu = sqr_eq(a_1, c, a_m1)
            return (1 / mu + refl_coef * mu) / (1 + refl_coef)

        return self._calc_nlbc(diff_eq_solution_ratio=diff_eq_solution_ratio)

    def calc_upper_nlbc(self, beta, gamma):
        logging.debug('Computing upper nonlocal boundary condition...')
        alpha = self.alpha
        if abs(gamma) < 10 * np.finfo(float).eps:

            def diff_eq_solution_ratio(s, xi):
                a_m1 = 1 - alpha * (self.k0 * self.dz_m) ** 2 * (s - beta)
                a_1 = 1 - alpha * (self.k0 * self.dz_m) ** 2 * (s - beta)
                c = -2 + (2 * alpha - 1) * (self.k0 * self.dz_m) ** 2 * (s - beta)
                return 1 / sqr_eq(a_1, c, a_m1)

            return self._calc_nlbc(diff_eq_solution_ratio=diff_eq_solution_ratio)
        else:
            b = alpha * gamma * self.dz_m * (self.k0 * self.dz_m) ** 2
            d = gamma * self.dz_m * (self.k0 * self.dz_m) ** 2 - 2 * b

            def diff_eq_solution_ratio(s, xi):
                a_m1 = 1 - alpha * (self.k0 * self.dz_m) ** 2 * (s - beta) - b
                a_1 = 1 - alpha * (self.k0 * self.dz_m) ** 2 * (s - beta) + b
                c = -2 + (2 * alpha - 1) * (self.k0 * self.dz_m) ** 2 * (s - beta)
                return bessel_ratio_4th_order(a_m1, a_1, b, c, d, len(self.z_computational_grid)-1, self.tol)

            return self._calc_nlbc(diff_eq_solution_ratio=diff_eq_solution_ratio)

    def propagate(self, initials: list, *, direction=1):
        self.prepare_nlbc()
        x_computational_grid = np.arange(0, self.n_x) * self.dx_m
        field = HelmholtzField(x_grid_m=x_computational_grid[::self.params.x_output_filter],
                               z_grid_m=self.z_computational_grid[::self.params.z_output_filter])
        reflected = [np.empty(0)] * self.n_x
        if direction == 1 and len(initials[0]) > 0:
            phi = initials[0]
        else:
            phi = self.z_computational_grid * 0j

        field.field[0, :] = phi[::self.params.z_output_filter]
        phi_0 = np.zeros((self.n_x, max(self.params.exp_pade_order)), dtype=complex)
        phi_J = np.zeros((self.n_x, max(self.params.exp_pade_order)), dtype=complex)

        if direction == 1:
            iterator = enumerate(x_computational_grid[1:], start=1)
        else:
            iterator = enumerate(x_computational_grid[-2::-1], start=1)
            initials = initials[::-1]

        edges_dict = {}
        for edge in self.env.knife_edges:
            x_i = int(round(edge.range / self.dx_m))
            if direction == 1:
                edges_dict[x_i] = edge
            else:
                edges_dict[self.n_x - x_i - 1] = edge

        for x_i, x in iterator:
            terr_i = int(round(self.env.terrain(x) / self.dz_m))

            if self.terrain_method == TerrainMethod.pass_through:
                het = self.env.n2m1_profile(x, self.z_computational_grid, self.freq_hz) + 0j
                het[0:terr_i:] = self.env.ground_material.complex_permittivity(self.freq_hz) - 1
            elif self.terrain_method == TerrainMethod.staircase:
                phi = phi[terr_i::]
                het = self.env.n2m1_profile(x, self.z_computational_grid[terr_i::], self.freq_hz) + 0j

            # process boundary conditions
            if isinstance(self.env.lower_bc, TransparentBC):
                lower_convolution = np.einsum('ijk,ik->j', self.lower_nlbc[1:x_i], phi_0[x_i-1:0:-1])
            if isinstance(self.env.upper_bc, TransparentBC):
                upper_convolution = np.einsum('ijk,ik->j', upper_bc.coefs[1:x_i], phi_J[x_i-1:0:-1])

            for pc_i, (a, b) in enumerate(self.pade_coefs):
                # process boundary conditions
                if isinstance(lower_bc, NonLocalBC):
                    lower_bound = -lower_bc.coefs[0, pc_i, pc_i], 1, lower_convolution[pc_i] + lower_bc.coefs[0, pc_i].dot(phi_0[x_i])
                elif isinstance(lower_bc, LocalBC):
                    lower_bound = self._calc_lower_lbc(local_bc=lower_bc, a=a, b=b, x=x, z_min=self.z_computational_grid[terr_i], phi=phi)

                if isinstance(upper_bc, NonLocalBC):
                    upper_bound = 1, -upper_bc.coefs[0, pc_i, pc_i], upper_convolution[pc_i] + upper_bc.coefs[0, pc_i].dot(phi_J[x_i])
                elif isinstance(upper_bc, LocalBC):
                    raise Exception("Not supported yet")

                # propagate
                if polarz == 'H':
                    phi = self._Crank_Nikolson_propagate(a, b, het, phi, lower_bound=lower_bound, upper_bound=upper_bound)
                else:
                    phi = self._Crank_Nikolson_propagate_4th_order_v_pol(a, b, lambda z: self.env.n2minus1(x, z, self.freq_hz), phi,
                                                                         lower_bound=lower_bound, upper_bound=upper_bound)

                phi_0[x_i, pc_i], phi_J[x_i, pc_i] = phi[0], phi[-1]

            if x_i in edges_dict:
                imp_i = self.z_computational_grid <= edges_dict[x_i].height
                reflected[x_i] = np.copy(phi[imp_i]) * cm.exp(1j * self.k0 * x_computational_grid[x_i])
                phi[imp_i] = 0
                if initials[x_i].size > 0:
                    phi[imp_i] = -initials[x_i] * cm.exp(-1j * self.k0 * x_computational_grid[x_i])

            if self.terrain_method == TerrainMethod.staircase:
                phi = np.concatenate((np.zeros(len(self.z_computational_grid) - len(phi)), phi))

            if divmod(x_i, n_dx_out)[1] == 0:
                field.field[divmod(x_i, n_dx_out)[0], :] = phi[::n_dz_out]
                logging.debug('SSPade propagation x = ' + str(x) + "   " + str(np.linalg.norm(phi[::n_dz_out])))

        field.field *= np.tile(np.exp(1j * self.k0 * x_computational_grid[::n_dx_out]), (len(self.z_computational_grid[::n_dz_out]), 1)).T
        if direction == 1:
            return field, reflected
        else:
            field.field = field.field[::-1, :]
            return field, reflected[::-1]


class PickleStorage(HelmholtzPropagatorStorage):

    def __init__(self, name='nlbc'):
        self.file_name = name
        import os
        if os.path.exists(self.file_name):
            with open(self.file_name, 'rb') as f:
                self.nlbc_dict = pickle.load(f)
        else:
            self.nlbc_dict = {}

    def get_lower_nlbc(self, propagator: HelmholtzPadeSolver, n_x):
        beta = propagator.env.ground_material.complex_permittivity(propagator.freq_hz) - 1
        gamma = 0
        q = 'lower', propagator.k0, propagator.dx_m, propagator.dz_m, propagator.pade_order, propagator.z_order, propagator.spe, beta, gamma
        if q not in self.nlbc_dict or self.nlbc_dict[q].coefs.shape[0] < n_x:
            self.nlbc_dict[q] = propagator.calc_lower_nlbc(beta)
        lower_nlbc = self.nlbc_dict[q]

        with open(self.file_name, 'wb') as f:
            pickle.dump(self.nlbc_dict, f)

        return lower_nlbc

    def get_upper_nlbc(self, propagator: HelmholtzPadeSolver, n_x):
        gamma = propagator.env.n2m1_profile(0, propagator.env.z_max+1, propagator.freq_hz) - propagator.env.n2m1_profile(0, propagator.env.z_max, propagator.freq_hz)
        beta = propagator.env.n2m1_profile(0, propagator.env.z_max, propagator.freq_hz) - gamma * propagator.env.z_max
        q = 'upper', propagator.k0, propagator.dx_m, propagator.dz_m, propagator.pade_order, propagator.z_order, propagator.spe, beta, gamma
        if q not in self.nlbc_dict or self.nlbc_dict[q].coefs.shape[0] < n_x:
            self.nlbc_dict[q] = propagator.calc_upper_nlbc(beta, gamma)
        upper_nlbc = self.nlbc_dict[q]

        with open(self.file_name, 'wb') as f:
            pickle.dump(self.nlbc_dict, f)

        return upper_nlbc