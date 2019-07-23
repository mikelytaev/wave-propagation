import logging
import pickle
import types
import time

import math as fm
import cmath as cm
from dataclasses import dataclass, field
from enum import Enum

import mpmath
import numpy as np
from transforms.fcc_fourier import FCCAdaptiveFourier
from scipy import linalg as la

from propagators._utils import *

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from propagators._cn_utils import *
from propagators.contfrac import bessel_ratio_4th_order


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

    def __init__(self, n2minus1=None):
        self.n2minus1 = n2minus1


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


@dataclass
class Edge:
    x: float
    z_min: float
    z_max: float


@dataclass
class HelmholtzEnvironment:
    x_max_m: float
    lower_bc: BoundaryCondition = TransparentConstBC()
    upper_bc: BoundaryCondition = TransparentConstBC()
    z_min: float = 0
    z_max: float = 300
    n2minus1: types.FunctionType = lambda x, z, freq_hz: z*0
    use_n2minus1: bool = True
    rho: types.FunctionType = lambda x, z: z*0+1
    use_rho: bool = True
    terrain: types.FunctionType = lambda x: x*0
    knife_edges: List[Edge] = field(default_factory=list)


class HelmholtzPropagatorStorage:

    def __init__(self):
        pass

    def get_lower_nlbc(self, *, k0, dx_wl, dz_wl, pade_order, z_order, sqrt_alpha, spe, beta, gamma, n_x) -> DiscreteNonLocalBC:
        pass

    def set_lower_nlbc(self, *, k0, dx_wl, dz_wl, pade_order, z_order, sqrt_alpha, spe, beta, gamma, nlbc: DiscreteNonLocalBC):
        pass

    def get_upper_nlbc(self, *, k0, dx_wl, dz_wl, pade_order, z_order, sqrt_alpha, spe, beta, gamma, n_x) -> DiscreteNonLocalBC:
        pass

    def set_upper_nlbc(self, *, k0, dx_wl, dz_wl, pade_order, z_order, sqrt_alpha, spe, beta, gamma, nlbc: DiscreteNonLocalBC):
        pass


@dataclass
class HelmholtzPropagatorComputationalParams:
    max_range_m: float = None
    max_height_m: float = None
    dx_wl: float = None
    dz_wl: float = None
    max_propagation_angle: float = None
    max_src_angle: float = 0
    exp_pade_order: tuple = None
    x_output_filter: int = None
    z_output_filter: int = None
    two_way: bool = None
    two_way_iter_num: int = 0
    two_way_threshold: float = 0.05
    standard_pe: bool = False
    sqrt_alpha: float = 0
    z_order: int = 4
    terrain_method: TerrainMethod = None
    tol: float = None
    storage: HelmholtzPropagatorStorage = None


class HelmholtzField:

    def __init__(self, x_grid_m: np.ndarray, z_grid_m: np.ndarray):
        self.x_grid_m = x_grid_m
        self.z_grid_m = z_grid_m
        self.field = np.zeros((x_grid_m.size, z_grid_m.size), dtype=complex)


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
        self.x_computational_grid = np.arange(0, self.n_x) * self.dx_m

        if self.params.z_order == 2:
            self.alpha = 0
        else:
            self.alpha = 1 / 12

        if self.params.terrain_method is None:
            self.params.terrain_method = TerrainMethod.no

        if not self.env.use_n2minus1 and not self.env.use_rho and self.params.terrain_method in [TerrainMethod.no, TerrainMethod.staircase]:
            def diff2(s):
                return mpmath.acosh(1 + (self.k0 * self.dz_m) ** 2 * s / 2) ** 2 / (self.k0 * self.dz_m) ** 2
        else:
            def diff2(s):
                return s

        self.pade_coefs = pade_propagator_coefs(pade_order=self.params.exp_pade_order, diff2=diff2, k0=self.k0,
                                                dx=self.dx_m, spe=self.params.standard_pe, alpha=self.params.sqrt_alpha)

        self.lower_bc = None
        self.upper_bc = None

    def _optimize_params(self):
        #optimize max angle
        self.params.max_propagation_angle = self.params.max_propagation_angle or self._optimal_propagation_angle()
        logging.info("Propagation angle = " + str(self.params.max_propagation_angle))

        if not self.env.use_n2minus1 and not self.env.use_rho and self.params.terrain_method in [TerrainMethod.no, TerrainMethod.staircase]:
            self.params.z_order = 5
            logging.info("using Pade approximation for diff2_z")

        self.params.z_order = self.params.z_order or 4

        if self.params.dx_wl is None or self.params.dz_wl is None or self.params.exp_pade_order is None:
            logging.debug("Calculating optimal computational grid parameters")
            (self.params.dx_wl, self.params.dz_wl, self.params.exp_pade_order) = \
                optimal_params(max_angle=self.params.max_propagation_angle, threshold=5e-3, dx=self.params.dx_wl,
                               dz=self.params.dz_wl, pade_order=self.params.exp_pade_order, z_order=self.params.z_order)

            if self.params.dx_wl is None or self.params.dz_wl is None or self.params.exp_pade_order is None:
                raise Exception("Optimization failed")

        if self.params.max_height_m is None:
            self.params.max_height_m = abs(self.env.z_max - self.env.z_min)
        else:
            self.params.max_height_m = max(self.params.max_height_m, abs(self.env.z_max - self.env.z_min))

        x_approx_sampling = 2000
        z_approx_sampling = 1000

        if self.params.terrain_method == TerrainMethod.pass_through:
            n_g = self.env.ground_material.complex_permittivity(self.freq_hz)
            self.params.dx_wl /= round(abs(cm.sqrt(n_g - 0.1)))
            self.params.dz_wl /= round(abs(cm.sqrt(n_g - 0.1)))

        self.params.max_range_m = self.params.max_range_m or self.env.x_max_m

        self.params.dx_wl = min(self.params.dx_wl, self.params.max_range_m / self.wavelength / x_approx_sampling)
        self.params.dz_wl = min(self.params.dz_wl, self.params.max_height_m / self.wavelength / z_approx_sampling)
        self.n_x = fm.ceil(self.params.max_range_m / self.params.dx_wl / self.wavelength) + 1
        self.n_z = fm.ceil(self.params.max_height_m / (self.params.dz_wl * self.wavelength)) + 1

        logging.info("dx = " + str(self.params.dx_wl) + " wavelengths")
        logging.info("dz = " + str(self.params.dz_wl) + " wavelengths")
        logging.info("Pade order = " + str(self.params.exp_pade_order))

        self.params.x_output_filter = self.params.x_output_filter or fm.ceil(self.n_x / x_approx_sampling)
        self.params.z_output_filter = self.params.z_output_filter or fm.ceil(self.n_z / z_approx_sampling)

        if not self.params.tol:
            self.params.tol = 1e-11 if (isinstance(self.env.lower_bc, TransparentLinearBC) or
            isinstance(self.env.upper_bc, TransparentLinearBC)) else 1e-7

        if self.params.two_way is None:
            self.params.two_way = False

    def _optimal_propagation_angle(self):
        if len(self.env.knife_edges) > 0:
            return 85
        else:
            res = 3
            step = 10
            for x in np.arange(step, self.params.max_range_m, step):
                angle = cm.atan((self.env.terrain(x) - self.env.terrain(x - step)) / step) * 180 / cm.pi
                res = max(res, abs(angle))
            res = max(self.params.max_src_angle, fm.ceil(res))
            return res

    def _Crank_Nikolson_propagate_deprecated(self, a, b, het, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
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

    def _Crank_Nikolson_propagate(self, a, b, m2minus1_func, rho_func, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        alpha_m = self.alpha * rho_func(self.z_computational_grid[1:-1:])

        c_a_left = 1 / rho_func(self.z_computational_grid[1:-1:] - self.dz_m / 2) * \
                   (alpha_m * (self.k0 * self.dz_m) ** 2 + a * rho_func(self.z_computational_grid[1:-1:]) +
                    a * alpha_m * (self.k0 * self.dz_m) ** 2 * m2minus1_func(self.z_computational_grid[:-2:]))

        c_a_right = 1 / rho_func(self.z_computational_grid[1:-1:] + self.dz_m / 2) * \
                    (alpha_m * (self.k0 * self.dz_m) ** 2 + a * rho_func(self.z_computational_grid[1:-1:]) +
                     a * alpha_m * (self.k0 * self.dz_m) ** 2 * m2minus1_func(self.z_computational_grid[2::]))

        c_b_left = 1 / rho_func(self.z_computational_grid[1:-1:] - self.dz_m / 2) * \
                   (alpha_m * (self.k0 * self.dz_m) ** 2 + b * rho_func(self.z_computational_grid[1:-1:]) +
                    b * alpha_m * (self.k0 * self.dz_m) ** 2 * m2minus1_func(self.z_computational_grid[:-2:]))

        c_b_right = 1 / rho_func(self.z_computational_grid[1:-1:] + self.dz_m / 2) * \
                    (alpha_m * (self.k0 * self.dz_m) ** 2 + b * rho_func(self.z_computational_grid[1:-1:]) +
                     b * alpha_m * (self.k0 * self.dz_m) ** 2 * m2minus1_func(self.z_computational_grid[2::]))

        het_mid2 = 1 / rho_func(self.z_computational_grid[1:-1] + self.dz_m / 2) + \
                   1 / rho_func(self.z_computational_grid[1:-1] - self.dz_m / 2)

        d_a = (self.k0 * self.dz_m) ** 2 * (1 - alpha_m * het_mid2) - a * het_mid2 * \
              rho_func(self.z_computational_grid[1:-1:]) + a * (self.k0 * self.dz_m) ** 2 * m2minus1_func(
            self.z_computational_grid[1:-1:]) - a * alpha_m * (self.k0 * self.dz_m) ** 2 * m2minus1_func(
            self.z_computational_grid[1:-1:]) * het_mid2

        d_b = (self.k0 * self.dz_m) ** 2 * (1 - alpha_m * het_mid2) - b * het_mid2 * \
              rho_func(self.z_computational_grid[1:-1:]) + b * (self.k0 * self.dz_m) ** 2 * m2minus1_func(
            self.z_computational_grid[1:-1:]) - b * alpha_m * (self.k0 * self.dz_m) ** 2 * m2minus1_func(
            self.z_computational_grid[1:-1:]) * het_mid2

        rhs = np.concatenate(([0], d_a, [0])) * initial
        rhs[1::] += np.concatenate((c_a_left, [0])) * initial[:-1:]
        rhs[:-1:] += np.concatenate(([0], c_a_right)) * initial[1::]
        rhs[0] = lower_bound[2]
        rhs[-1] = upper_bound[2]

        d_b = np.concatenate(([lower_bound[0]], d_b, [upper_bound[1]]))
        c_b_left = np.concatenate((c_b_left, [upper_bound[0]]))
        c_b_right = np.concatenate(([lower_bound[1]], c_b_right))

        return np.array(tridiag_method(c_b_left, d_b, c_b_right, rhs))

    def _calc_lower_lbc(self, *, local_bc: RobinBC, a, b, x, z_min, phi):
        q1, q2 = local_bc.q1, local_bc.q2
        r0 = q2 * (self.k0 * self.dz_m) ** 2 * (1 + b * (self.env.n2minus1(x, z_min, self.freq_hz))) + 2 * b * (self.dz_m * q1 - q2)
        r1 = 2 * b * q2
        r2 = q2 * (self.k0 * self.dz_m) ** 2 * (1 + a * (self.env.n2minus1(x, z_min, self.freq_hz))) + 2 * a * (self.dz_m * q1 - q2)
        r3 = 2 * a * q2
        return r0, r1, r2 * phi[0] + r3 * phi[1]

    def _calc_upper_lbc(self, *, local_bc: RobinBC, a, b, x, z_max, phi):
        q1, q2 = local_bc.q1, local_bc.q2
        r1 = q2 * (self.k0 * self.dz_m) ** 2 * (1 + b * (self.env.n2minus1(x, z_max, self.freq_hz))) + 2 * b * (self.dz_m * q1 - q2)
        r0 = 2 * b * q2
        r3 = q2 * (self.k0 * self.dz_m) ** 2 * (1 + a * (self.env.n2minus1(x, z_max, self.freq_hz))) + 2 * a * (self.dz_m * q1 - q2)
        r2 = 2 * a * q2
        return r0, r1, r2 * phi[-2] + r3 * phi[-1]

    def _prepare_boundary_conditions(self):
        if self.lower_bc is not None and self.upper_bc is not None:
            return

        if isinstance(self.env.lower_bc, TransparentBC) and \
                self.params.terrain_method in [TerrainMethod.pass_through, TerrainMethod.no]:
            beta = self.env.n2minus1(0, self.env.z_min - 1, self.freq_hz)
            gamma = 0
            if self.params.storage:
                self.lower_bc = self.params.storage.get_lower_nlbc(k0=self.k0, dx_wl=self.params.dx_wl, dz_wl=self.params.dz_wl,
                                                                   pade_order=self.params.exp_pade_order, z_order=self.params.z_order,
                                                                   sqrt_alpha=self.params.sqrt_alpha, spe=self.params.standard_pe, beta=beta,
                                                                   gamma=gamma, n_x=self.n_x)

                if self.lower_bc is None:
                    self.lower_bc = self._calc_lower_nlbc(beta)
                    self.params.storage.set_lower_nlbc(k0=self.k0, dx_wl=self.params.dx_wl, dz_wl=self.params.dz_wl,
                                                       pade_order=self.params.exp_pade_order,
                                                       z_order=self.params.z_order,
                                                       sqrt_alpha=self.params.sqrt_alpha, spe=self.params.standard_pe,
                                                       beta=beta,
                                                       gamma=gamma, nlbc=self.lower_bc)
            else:
                self.lower_bc = self._calc_lower_nlbc(beta)
        else:
            self.lower_bc = self.env.lower_bc

        if isinstance(self.env.upper_bc, TransparentBC):
            gamma = self.env.n2minus1(0, self.env.z_max + 1, self.freq_hz) - self.env.n2minus1(0, self.env.z_max, self.freq_hz)
            beta = self.env.n2minus1(0, self.env.z_max, self.freq_hz) - gamma * self.env.z_max
            if self.params.storage:
                self.upper_bc = self.params.storage.get_upper_nlbc(k0=self.k0, dx_wl=self.params.dx_wl, dz_wl=self.params.dz_wl,
                                                                   pade_order=self.params.exp_pade_order, z_order=self.params.z_order,
                                                                   sqrt_alpha=self.params.sqrt_alpha, spe=self.params.standard_pe, beta=beta,
                                                                   gamma=gamma, n_x=self.n_x)

                if self.upper_bc is None:
                    self.upper_bc = self._calc_upper_nlbc(beta, gamma)
                    self.params.storage.set_upper_nlbc(k0=self.k0, dx_wl=self.params.dx_wl, dz_wl=self.params.dz_wl,
                                                       pade_order=self.params.exp_pade_order,
                                                       z_order=self.params.z_order,
                                                       sqrt_alpha=self.params.sqrt_alpha, spe=self.params.standard_pe,
                                                       beta=beta,
                                                       gamma=gamma, nlbc=self.upper_bc)
            else:
                self.upper_bc = self._calc_upper_nlbc(beta, gamma)
        else:
            self.upper_bc = self.env.upper_bc

    def _calc_nlbc(self, diff_eq_solution_ratio):
        num_roots, den_roots = list(zip(*self.pade_coefs))
        m_size = len(self.pade_coefs)
        tau = 1.001
        if max(self.params.exp_pade_order) == 1:
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
                r = np.diag([diff_eq_solution_ratio(a) for a in w])
                res = vr.dot(r).dot(la.inv(vr))
                return res.reshape(m_size**2)

        fcca = FCCAdaptiveFourier(2 * fm.pi, -np.arange(0, self.n_x), rtol=self.params.tol)

        coefs = (tau**np.repeat(np.arange(0, self.n_x)[:, np.newaxis], m_size ** 2, axis=1) / (2*fm.pi) *
                fcca.forward(lambda t: nlbc_transformed(t), 0, 2*fm.pi)).reshape((self.n_x, m_size, m_size))

        return DiscreteNonLocalBC(r0=1, r1=1, coefs=coefs)

    def _calc_lower_nlbc(self, beta):
        logging.debug('Computing lower nonlocal boundary condition...')
        alpha = self.alpha

        def diff_eq_solution_ratio(s):
            # k_d = self.wavelength / 2 / self.dx_m
            # k_x = k_d * self.k0 + 1j / self.dx_m * cm.log(xi)
            # k_z = cm.sqrt(self.k0**2 - k_x ** 2)
            # yy = cm.sqrt(self.k0**2 * (beta + 1) - k_x ** 2)
            # refl_coef = (yy - (beta+1) * k_z) / (yy + (beta+1) * k_z)
            #
            # theta = 30
            # refl_coef = cm.exp(-10*((k_x - self.k0 * cm.cos(theta * cm.pi / 180))) ** 2)
            # if abs(refl_coef) > 0.4:
            #     print("refl_coef = " + str(refl_coef) + "k_x / k = " + str(k_x / self.k0))

            #beta = 0
            refl_coef = 0
            a_m1 = 1 - alpha * (self.k0 * self.dz_m) ** 2 * (s - 0)
            a_1 = 1 - alpha * (self.k0 * self.dz_m) ** 2 * (s - 0)
            c = -2 + (2 * alpha - 1) * (self.k0 * self.dz_m) ** 2 * (s - 0)

            mu = sqr_eq(a_1, c, a_m1)
            return (1 / mu + refl_coef * mu) / (1 + refl_coef)

        return self._calc_nlbc(diff_eq_solution_ratio=diff_eq_solution_ratio)

    def _calc_upper_nlbc(self, beta, gamma):
        logging.debug('Computing upper nonlocal boundary condition...')
        alpha = self.alpha
        if abs(gamma) < 10 * np.finfo(float).eps:

            def diff_eq_solution_ratio(s):
                a_m1 = 1 - alpha * (self.k0 * self.dz_m) ** 2 * (s - beta)
                a_1 = 1 - alpha * (self.k0 * self.dz_m) ** 2 * (s - beta)
                c = -2 + (2 * alpha - 1) * (self.k0 * self.dz_m) ** 2 * (s - beta)
                return 1 / sqr_eq(a_1, c, a_m1)

            return self._calc_nlbc(diff_eq_solution_ratio=diff_eq_solution_ratio)
        else:
            b = alpha * gamma * self.dz_m * (self.k0 * self.dz_m) ** 2
            d = gamma * self.dz_m * (self.k0 * self.dz_m) ** 2 - 2 * b

            def diff_eq_solution_ratio(s):
                a_m1 = 1 - alpha * (self.k0 * self.dz_m) ** 2 * (s - beta) - b
                a_1 = 1 - alpha * (self.k0 * self.dz_m) ** 2 * (s - beta) + b
                c = -2 + (2 * alpha - 1) * (self.k0 * self.dz_m) ** 2 * (s - beta)
                return bessel_ratio_4th_order(a_m1, a_1, b, c, d, len(self.z_computational_grid)-1, self.params.tol)

            return self._calc_nlbc(diff_eq_solution_ratio=diff_eq_solution_ratio)

    def _propagate(self, initials: list, direction=1):
        self._prepare_boundary_conditions()
        field = HelmholtzField(x_grid_m=self.x_computational_grid[::self.params.x_output_filter],
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
            iterator = enumerate(self.x_computational_grid[1:], start=1)
        else:
            iterator = enumerate(self.x_computational_grid[-2::-1], start=1)
            initials = initials[::-1]

        edges_dict = {}
        for edge in self.env.knife_edges:
            x_i = int(round(edge.x / self.dx_m))
            if direction == 1:
                edges_dict[x_i] = edge
            else:
                edges_dict[self.n_x - x_i - 1] = edge

        for x_i, x in iterator:
            terr_i = int(round(self.env.terrain(x) / self.dz_m))

            if self.params.terrain_method == TerrainMethod.pass_through:
                het = self.env.n2m1_profile(x, self.z_computational_grid, self.freq_hz) + 0j
                het[0:terr_i:] = self.env.ground_material.complex_permittivity(self.freq_hz) - 1
            elif self.params.terrain_method == TerrainMethod.staircase:
                phi = phi[terr_i::]

            # process boundary conditions
            if isinstance(self.lower_bc, DiscreteNonLocalBC):
                lower_convolution = np.einsum('ijk,ik->j', self.lower_bc.coefs[1:x_i], phi_0[x_i - 1:0:-1])
            if isinstance(self.upper_bc, DiscreteNonLocalBC):
                upper_convolution = np.einsum('ijk,ik->j', self.upper_bc.coefs[1:x_i], phi_J[x_i-1:0:-1])

            for pc_i, (a, b) in enumerate(self.pade_coefs):
                # process boundary conditions
                if isinstance(self.lower_bc, DiscreteNonLocalBC):
                    lower_bound = -self.lower_bc.coefs[0, pc_i, pc_i], 1, lower_convolution[pc_i] + self.lower_bc.coefs[0, pc_i].dot(phi_0[x_i])
                elif isinstance(self.lower_bc, RobinBC):
                    lower_bound = self._calc_lower_lbc(local_bc=self.lower_bc, a=a, b=b, x=x, z_min=self.z_computational_grid[terr_i], phi=phi)

                if isinstance(self.upper_bc, DiscreteNonLocalBC):
                    upper_bound = 1, -self.upper_bc.coefs[0, pc_i, pc_i], upper_convolution[pc_i] + self.upper_bc.coefs[0, pc_i].dot(phi_J[x_i])
                elif isinstance(self.upper_bc, RobinBC):
                    upper_bound = self._calc_upper_lbc(local_bc=self.lower_bc, a=a, b=b, x=x, z_max=self.z_computational_grid[-1], phi=phi)

                phi = self._Crank_Nikolson_propagate(a, b, lambda z: self.env.n2minus1(x, z, self.freq_hz),
                                                         lambda z: self.env.rho(x, z), phi,
                                                         lower_bound=lower_bound, upper_bound=upper_bound)

                phi_0[x_i, pc_i], phi_J[x_i, pc_i] = phi[0], phi[-1]

            if x_i in edges_dict:
                imp_i = np.logical_and(edges_dict[x_i].z_min <= self.z_computational_grid , self.z_computational_grid <= edges_dict[x_i].z_max)
                reflected[x_i] = np.copy(phi[imp_i]) * cm.exp(1j * self.k0 * self.x_computational_grid[x_i])
                phi[imp_i] = 0
                if initials[x_i].size > 0:
                    phi[imp_i] = -initials[x_i] * cm.exp(-1j * self.k0 * self.x_computational_grid[x_i])

            if self.params.terrain_method == TerrainMethod.staircase:
                phi = np.concatenate((np.zeros(len(self.z_computational_grid) - len(phi)), phi))

            if divmod(x_i, self.params.x_output_filter)[1] == 0:
                field.field[divmod(x_i, self.params.x_output_filter)[0], :] = phi[::self.params.z_output_filter]
                logging.debug('SSPade propagation x = ' + str(x) + "   " + str(np.linalg.norm(phi[::self.params.z_output_filter])))

        field.field *= np.tile(np.exp(1j * self.k0 * self.x_computational_grid[::self.params.x_output_filter]),
                               (len(self.z_computational_grid[::self.params.z_output_filter]), 1)).T
        if direction == 1:
            return field, reflected
        else:
            field.field = field.field[::-1, :]
            return field, reflected[::-1]

    def calculate(self, initial_func: types.FunctionType):
        start_time = time.time()
        initials_fw = [np.empty(0)] * self.n_x
        initials_fw[0] = np.array([initial_func(a) for a in self.z_computational_grid])
        field = HelmholtzField(x_grid_m=self.x_computational_grid[::self.params.x_output_filter],
                               z_grid_m=self.z_computational_grid[::self.params.z_output_filter])

        if self.params.two_way_iter_num in [None, 0]:
            self.params.two_way_iter_num = 10000000

        for i in range(0, self.params.two_way_iter_num):
            field_fw, initials_fw = self._propagate(initials=initials_fw, direction=1)

            prev_field = field.field.copy()
            field.field += field_fw.field
            if not self.params.two_way:
                break

            field_bw, initials_fw = self._propagate(initials=initials_fw, direction=-1)
            field.field += field_bw.field

            err = np.linalg.norm(field.field - prev_field) / np.linalg.norm(field.field)
            logging.debug("Iteration no " + str(i) + "relative error = " + str(err))
            if err < self.params.two_way_threshold:
                break

        logging.debug("Elapsed time: " + str(time.time() - start_time))

        return field


class PickleStorage(HelmholtzPropagatorStorage):

    def __init__(self, name='nlbc'):
        self.file_name = name
        import os
        if os.path.exists(self.file_name):
            with open(self.file_name, 'rb') as f:
                self.nlbc_dict = pickle.load(f)
        else:
            self.nlbc_dict = {}

    def get_lower_nlbc(self, *, k0, dx_wl, dz_wl, pade_order, z_order, sqrt_alpha, spe, beta, gamma, n_x) -> DiscreteNonLocalBC:
        q = 'lower', k0, dx_wl, dz_wl, pade_order, z_order, spe, beta, gamma
        if q not in self.nlbc_dict or self.nlbc_dict[q].coefs.shape[0] < n_x:
            return None
        else:
            return self.nlbc_dict[q]

    def set_lower_nlbc(self, *, k0, dx_wl, dz_wl, pade_order, z_order, sqrt_alpha, spe, beta, gamma, nlbc: DiscreteNonLocalBC):
        q = 'lower', k0, dx_wl, dz_wl, pade_order, z_order, spe, beta, gamma
        self.nlbc_dict[q] = nlbc
        with open(self.file_name, 'wb') as f:
            pickle.dump(self.nlbc_dict, f)

    def get_upper_nlbc(self, *, k0, dx_wl, dz_wl, pade_order, z_order, sqrt_alpha, spe, beta, gamma, n_x) -> DiscreteNonLocalBC:
        q = 'upper', k0, dx_wl, dz_wl, pade_order, z_order, spe, beta, gamma
        if q not in self.nlbc_dict or self.nlbc_dict[q].coefs.shape[0] < n_x:
            return None
        else:
            return self.nlbc_dict[q]

    def set_upper_nlbc(self, *, k0, dx_wl, dz_wl, pade_order, z_order, sqrt_alpha, spe, beta, gamma, nlbc: DiscreteNonLocalBC):
        q = 'upper', k0, dx_wl, dz_wl, pade_order, z_order, spe, beta, gamma
        self.nlbc_dict[q] = nlbc
        with open(self.file_name, 'wb') as f:
            pickle.dump(self.nlbc_dict, f)
