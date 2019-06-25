import logging
import math as fm
import pickle

import pyximport
import scipy.linalg as la

import time

from rwp.antennas import *
from rwp.environment import *
from rwp._utils import *
from fcc_fourier import FCCAdaptiveFourier
from rwp.field import Field

#pyximport.install(setup_args={"include_dirs": np.get_include()})
from contfrac import *
from _cn_utils import *

from scipy.sparse import *
from scipy.sparse.linalg import *


class BoundaryCondition:
    pass


class LocalBC(BoundaryCondition):

    def __init__(self, q1: complex, q2: complex):
        self.q1 = q1
        self.q2 = q2


class NonLocalBC(BoundaryCondition):

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


class SSPEEnvironment:

    def __init__(self, x_max_m, lower_bc, upper_bc):
        self.x_max_m = x_max_m
        self.lower_bc = lower_bc
        self.upper_bc = upper_bc

    def z_min(self, x_m):
        return 0

    def z_max(self, x_m):
        return 300


class HelmholtzPadeSolver:

    def __init__(self, env: Troposphere, n_x, wavelength=1.0, pade_order=(1, 2), z_order=4, spe=False, dx_wl=100, dz_wl=1,
                 terrain_method=TerrainMethod.pass_through, tol=1e-11):
        self.env = env
        self.k0 = (2 * cm.pi) / wavelength
        self.wavelength = wavelength
        self.freq_hz = LIGHT_SPEED / self.wavelength
        self.n_x = n_x
        self.n_z = fm.ceil(self.env.z_max / (dz_wl * wavelength)) + 1
        self.z_computational_grid, self.dz = np.linspace(0, self.env.z_max, self.n_z, retstep=True)
        self.dx = dx_wl * wavelength
        self.pade_order = pade_order
        self.spe = spe
        self.tol = tol
        self.z_order = z_order
        self.terrain_method = terrain_method

        if self.z_order == 2:
            self.alpha = 0
        else:
            self.alpha = 1 / 12

        if self.env.is_homogeneous() and self.terrain_method == TerrainMethod.staircase:
            def diff2(s):
                return mpmath.acosh(1 + (self.k0 * self.dz) ** 2 * s / 2) ** 2 / (self.k0 * self.dz) ** 2
        else:
            def diff2(s):
                return s

        self.pade_coefs = pade_propagator_coefs(pade_order=pade_order, diff2=diff2, k0=self.k0, dx=self.dx, spe=self.spe)

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
            return np.array(Crank_Nikolson_propagator((self.k0 * self.dz) ** 2, a, b, het, initial, lower_bound, upper_bound))
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
        c_a = alpha * (self.k0 * self.dz)**2 + a + alpha*a*(self.k0 * self.dz)**2 * het
        c_b = alpha * (self.k0 * self.dz)**2 + b + alpha*b*(self.k0 * self.dz)**2 * het
        d_a = (self.k0 * self.dz)**2 * (1 - 2*alpha) - 2*a + a*(self.k0 * self.dz)**2 * het - 2*a*alpha*(self.k0*self.dz)**2 * het
        d_b = (self.k0 * self.dz)**2 * (1 - 2*alpha) - 2*b + b*(self.k0 * self.dz)**2 * het - 2*b*alpha*(self.k0*self.dz)**2 * het

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

        c_a_left = 1 / (het_func(self.z_computational_grid[1:-1:] - self.dz / 2) + 1) * \
                   (alpha_m * (self.k0 * self.dz) ** 2 + a * (het_func(self.z_computational_grid[1:-1:]) + 1) +
                    a * alpha_m *(self.k0 * self.dz) ** 2 * het_func(self.z_computational_grid[:-2:]))

        c_a_right = 1 / (het_func(self.z_computational_grid[1:-1:] + self.dz / 2) + 1) * \
                   (alpha_m * (self.k0 * self.dz) ** 2 + a * (het_func(self.z_computational_grid[1:-1:]) + 1) +
                    a * alpha_m * (self.k0 * self.dz) ** 2 * het_func(self.z_computational_grid[2::]))

        c_b_left = 1 / (het_func(self.z_computational_grid[1:-1:] - self.dz / 2) + 1) * \
                   (alpha_m * (self.k0 * self.dz) ** 2 + b * (het_func(self.z_computational_grid[1:-1:]) + 1) +
                    b * alpha_m * (self.k0 * self.dz) ** 2 * het_func(self.z_computational_grid[:-2:]))

        c_b_right = 1 / (het_func(self.z_computational_grid[1:-1:] + self.dz / 2) + 1) * \
                    (alpha_m * (self.k0 * self.dz) ** 2 + b * (het_func(self.z_computational_grid[1:-1:]) + 1) +
                     b * alpha_m * (self.k0 * self.dz) ** 2 * het_func(self.z_computational_grid[2::]))

        het_mid2 = 1 / (het_func(self.z_computational_grid[1:-1] + self.dz / 2) + 1) + \
                   1 / (het_func(self.z_computational_grid[1:-1] - self.dz / 2) + 1)

        d_a = (self.k0 * self.dz) ** 2 * (1 - alpha_m * het_mid2) - a * het_mid2 * (
                    het_func(self.z_computational_grid[1:-1:]) + 1) + a * (self.k0 * self.dz) ** 2 * het_func(
            self.z_computational_grid[1:-1:]) - a * alpha_m * (self.k0 * self.dz) ** 2 * het_func(
            self.z_computational_grid[1:-1:]) * het_mid2

        d_b = (self.k0 * self.dz) ** 2 * (1 - alpha_m * het_mid2) - b * het_mid2 * (
                het_func(self.z_computational_grid[1:-1:]) + 1) + b * (self.k0 * self.dz) ** 2 * het_func(
            self.z_computational_grid[1:-1:]) - b * alpha_m * (self.k0 * self.dz) ** 2 * het_func(
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
        r0 = q2 * (self.k0 * self.dz) ** 2 * (1 + b * (self.env.n2m1_profile(x, z_min, self.freq_hz))) + 2 * b * (self.dz * q1 - q2)
        r1 = 2 * b * q2
        r2 = q2 * (self.k0 * self.dz) ** 2 * (1 + a * (self.env.n2m1_profile(x, z_min, self.freq_hz))) + 2 * a * (self.dz * q1 - q2)
        r3 = 2 * a * q2
        return r0, r1, r2 * phi[0] + r3 * phi[1]

    def _calc_nlbc(self, diff_eq_solution):
        num_roots, den_roots = list(zip(*self.pade_coefs))
        m_size = len(self.pade_coefs)
        tau = 1.001
        if max(self.pade_order) == 1:
            def nlbc_transformed(t):
                return diff_eq_solution(((1 - t) / (-num_roots[0] + den_roots[0] * t)))
        else:
            def nlbc_transformed(t):
                matrix_a = np.diag(den_roots, 0) - np.diag(num_roots[1:], -1)
                matrix_a[0, -1] = -num_roots[0]
                matrix_a[0, 0] *= t
                matrix_b = np.diag(-np.ones(m_size), 0) + np.diag(np.ones(m_size - 1), -1) + 0j
                matrix_b[0, -1] = 1
                matrix_b[0, 0] *= t
                w, vr = la.eig(matrix_b, matrix_a, right=True)
                r = np.diag([diff_eq_solution(a) for a in w])
                res = vr.dot(r).dot(la.inv(vr))
                return res.reshape(m_size**2)

        fcca = FCCAdaptiveFourier(2 * fm.pi, -np.arange(0, self.n_x), rtol=self.tol)

        coefs = (tau**np.repeat(np.arange(0, self.n_x)[:, np.newaxis], m_size ** 2, axis=1) / (2*fm.pi) *
                fcca.forward(lambda t: nlbc_transformed(tau * cm.exp(1j*t)), 0, 2*fm.pi)).reshape((self.n_x, m_size, m_size))

        return NonLocalBC(r0=1, r1=1, coefs=coefs)

    def calc_lower_nlbc(self, beta):
        logging.debug('Computing lower nonlocal boundary condition...')
        alpha = self.alpha

        def diff_eq_solution(s):
            a_m1 = 1 - alpha * (self.k0 * self.dz) ** 2 * (s - beta)
            a_1 = 1 - alpha * (self.k0 * self.dz) ** 2 * (s - beta)
            c = -2 + (2 * alpha - 1) * (self.k0 * self.dz) ** 2 * (s - beta)
            return 1 / sqr_eq(a_1, c, a_m1)

        return self._calc_nlbc(diff_eq_solution=diff_eq_solution)

    def calc_upper_nlbc(self, beta, gamma):
        logging.debug('Computing upper nonlocal boundary condition...')
        alpha = self.alpha
        if abs(gamma) < 10 * np.finfo(float).eps:

            def diff_eq_solution(s):
                a_m1 = 1 - alpha * (self.k0 * self.dz) ** 2 * (s - beta)
                a_1 = 1 - alpha * (self.k0 * self.dz) ** 2 * (s - beta)
                c = -2 + (2 * alpha - 1) * (self.k0 * self.dz) ** 2 * (s - beta)
                return 1 / sqr_eq(a_1, c, a_m1)

            return self._calc_nlbc(diff_eq_solution=diff_eq_solution)
        else:
            b = alpha * gamma * self.dz * (self.k0 * self.dz) ** 2
            d = gamma * self.dz * (self.k0 * self.dz) ** 2 - 2 * b

            def diff_eq_solution(s):
                a_m1 = 1 - alpha * (self.k0 * self.dz) ** 2 * (s - beta) - b
                a_1 = 1 - alpha * (self.k0 * self.dz) ** 2 * (s - beta) + b
                c = -2 + (2 * alpha - 1) * (self.k0 * self.dz) ** 2 * (s - beta)
                return bessel_ratio_4th_order(a_m1, a_1, b, c, d, len(self.z_computational_grid)-1, self.tol)

            return self._calc_nlbc(diff_eq_solution=diff_eq_solution)

    def propagate(self, initials: list, polarz, *, direction=1, lower_bc, upper_bc, n_dx_out=1, n_dz_out=1):
        x_computational_grid = np.arange(0, self.n_x) * self.dx
        field = Field(x_computational_grid[::n_dx_out], self.z_computational_grid[::n_dz_out],
                      freq_hz=self.freq_hz, precision=self.tol)
        reflected = [np.empty(0)] * self.n_x
        if direction == 1 and len(initials[0]) > 0:
            phi = initials[0]
        else:
            phi = self.z_computational_grid * 0j

        field.field[0, :] = phi[::n_dz_out]
        phi_0, phi_J = np.zeros((self.n_x, max(self.pade_order)), dtype=complex), np.zeros((self.n_x, max(self.pade_order)), dtype=complex)

        if direction == 1:
            iterator = enumerate(x_computational_grid[1:], start=1)
        else:
            iterator = enumerate(x_computational_grid[-2::-1], start=1)
            initials = initials[::-1]

        edges_dict = {}
        for edge in self.env.knife_edges:
            x_i = int(round(edge.range / self.dx))
            if direction == 1:
                edges_dict[x_i] = edge
            else:
                edges_dict[self.n_x - x_i - 1] = edge

        for x_i, x in iterator:
            terr_i = int(round(self.env.terrain(x) / self.dz))

            if self.terrain_method == TerrainMethod.pass_through:
                het = self.env.n2m1_profile(x, self.z_computational_grid, self.freq_hz) + 0j
                het[0:terr_i:] = self.env.ground_material.complex_permittivity(self.freq_hz) - 1
            elif self.terrain_method == TerrainMethod.staircase:
                phi = phi[terr_i::]
                het = self.env.n2m1_profile(x, self.z_computational_grid[terr_i::], self.freq_hz) + 0j

            # process boundary conditions
            if isinstance(lower_bc, NonLocalBC):
                lower_convolution = np.einsum('ijk,ik->j', lower_bc.coefs[1:x_i], phi_0[x_i-1:0:-1])
            if isinstance(upper_bc, NonLocalBC):
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
                    phi = self._Crank_Nikolson_propagate_4th_order_v_pol(a, b, lambda z: self.env.n2m1_profile(x, z, self.freq_hz), phi,
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
                logging.debug('SSPade propagation x = ' + str(x))

        field.field *= np.tile(np.exp(1j * self.k0 * x_computational_grid[::n_dx_out]), (len(self.z_computational_grid[::n_dz_out]), 1)).T
        if direction == 1:
            return field, reflected
        else:
            field.field = field.field[::-1, :]
            return field, reflected[::-1]


class TroposphericRadioWaveSSPadePropagator:

    def __init__(self, *, antenna: Source, env: Troposphere, two_way=False, iter_num=None, max_range_m, max_angle=None,
                 dx_wl=None, dz_wl=None, n_dx_out=None, n_dz_out=None, pade_order=None, z_order=4, spe=False,
                 terrain_method=None, nlbc_manager_path='nlbc', tol=1e-11):
        self.src = antenna
        self.env = env
        self.two_way = two_way
        if len(self.env.knife_edges) == 0:
            self.two_way = False
        self.iter_num = iter_num or 1
        self.max_range_m = max_range_m
        self.nlbc_manager_path = nlbc_manager_path

        max_angle = max_angle or self._optimal_angle()
        if dx_wl:
            dx_wl = [dx_wl]
        if dz_wl:
            dz_wl = [dz_wl]
        if pade_order:
            pade_order = [pade_order]

        logging.info("ground refractive index: " + str(self.env.ground_material.complex_permittivity(antenna.freq_hz)))
        if terrain_method is None:

            if abs(self.env.ground_material.complex_permittivity(antenna.freq_hz)) < 100 and max_angle > 10:
                self.terrain_method = TerrainMethod.pass_through
            else:
                self.terrain_method = TerrainMethod.staircase
        else:
            self.terrain_method = terrain_method

        logging.info("Terrain method: " + self.terrain_method.name)

        logging.info("max_angle = " + str(max_angle))
        logging.info("calculating optimal grid parameters...")

        if self.env.is_homogeneous() and self.terrain_method == TerrainMethod.staircase:
            z_order_p = float('inf')
            logging.info("using Pade approximation for diff2_z")
        else:
            z_order_p = z_order

        (opt_dx, opt_dz, opt_pade) = optimal_params(max_angle=max_angle, threshold=5e-3, dxs=dx_wl, dzs=dz_wl,
                                                    pade_orders=pade_order, z_order=z_order_p)

        x_approx_sampling = 2000
        z_approx_sampling = 1000
        wavelength = 3e8 / self.src.freq_hz

        opt_dx = min(opt_dx or dx_wl[0], max_range_m / wavelength / x_approx_sampling)
        opt_dz = min(opt_dz or dz_wl[0], env.z_max / wavelength / z_approx_sampling)
        opt_pade = opt_pade or pade_order[0]

        if self.terrain_method == TerrainMethod.pass_through:
            n_g = self.env.ground_material.complex_permittivity(antenna.freq_hz)
            opt_dx /= round(abs(cm.sqrt(n_g - 0.1)))
            opt_dz /= round(abs(cm.sqrt(n_g - 0.1)))

        self.n_x = fm.ceil(self.max_range_m / opt_dx / wavelength) + 1

        self.n_dx_out = n_dx_out or fm.ceil(max_range_m / antenna.wavelength / opt_dx / x_approx_sampling)
        self.n_dz_out = n_dz_out or fm.ceil(env.z_max / antenna.wavelength / opt_dz / z_approx_sampling)

        logging.info("dx = " + str(opt_dx))
        logging.info("dz = " + str(opt_dz))
        logging.info("Pade order = " + str(opt_pade))
        self.propagator = HelmholtzPadeSolver(env=self.env, n_x=self.n_x, wavelength=antenna.wavelength, z_order=z_order,
                                              pade_order=opt_pade, spe=spe, dx_wl=opt_dx, dz_wl=opt_dz,
                                              terrain_method=self.terrain_method, tol=tol)

    def _optimal_angle(self):
        if len(self.env.knife_edges) > 0:
            return 85
        else:
            res = 3
            step = 10
            for x in np.arange(step, self.max_range_m, step):
                angle = cm.atan((self.env.terrain(x) - self.env.terrain(x - step)) / step) * 180 / cm.pi
                res = max(res, abs(angle))
            res = max(self.src.max_angle(), fm.ceil(res))
            return res

    def _prepare_bc(self):
        upper_bc = NLBCManager(self.nlbc_manager_path).get_upper_nlbc(self.propagator, self.n_x)
        if self.terrain_method == TerrainMethod.pass_through:
            lower_bc = NLBCManager(self.nlbc_manager_path).get_lower_nlbc(self.propagator, self.n_x)
        else:
            if isinstance(self.env.ground_material, PerfectlyElectricConducting):
                if self.src.polarz == 'H':
                    q1, q2 = 1, 0
                else:
                    q1, q2 = 0, 1
            else:
                if self.src.polarz == 'H':
                    q1, q2 = 1j * self.propagator.k0 * self.env.ground_material.complex_permittivity(self.src.freq_hz) ** (1 / 2), 1
                else:
                    q1, q2 = 1j * self.propagator.k0 * self.env.ground_material.complex_permittivity(self.src.freq_hz) ** (-1 / 2), 1
            lower_bc = LocalBC(q1, q2)
        return lower_bc, upper_bc

    def calculate(self):
        start_time = time.time()
        lower_bc, upper_bc = self._prepare_bc()
        initials_fw = [np.empty(0)] * self.n_x
        initials_fw[0] = np.array([self.src.aperture(a) for a in self.propagator.z_computational_grid])
        reflected_bw = initials_fw
        x_computational_grid = np.arange(0, self.n_x) * self.propagator.dx
        field = Field(x_computational_grid[::self.n_dx_out], self.propagator.z_computational_grid[::self.n_dz_out],
                      freq_hz=self.propagator.freq_hz,
                      precision=self.propagator.tol)
        for i in range(0, self.iter_num):
            field_fw, reflected_fw = self.propagator.propagate(polarz=self.src.polarz, initials=reflected_bw,
                                                               direction=1, lower_bc=lower_bc, upper_bc=upper_bc,
                                                               n_dx_out=self.n_dx_out, n_dz_out=self.n_dz_out)

            field.field += field_fw.field
            #logging.debug(np.linalg.norm(field_fw.field))
            if not self.two_way:
                break

            field_bw, reflected_bw = self.propagator.propagate(polarz=self.src.polarz, initials=reflected_fw,
                                                               direction=-1, lower_bc=lower_bc, upper_bc=upper_bc,
                                                               n_dx_out=self.n_dx_out, n_dz_out=self.n_dz_out)
            field.field += field_bw.field
            #logging.debug(np.linalg.norm(field_bw.field))

        logging.debug("Elapsed time: " + str(time.time() - start_time))

        return field


class NLBCManager:

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
        q = 'lower', propagator.k0, propagator.dx, propagator.dz, propagator.pade_order, propagator.z_order, propagator.spe, beta, gamma
        if q not in self.nlbc_dict or self.nlbc_dict[q].coefs.shape[0] < n_x:
            self.nlbc_dict[q] = propagator.calc_lower_nlbc(beta)
        lower_nlbc = self.nlbc_dict[q]

        with open(self.file_name, 'wb') as f:
            pickle.dump(self.nlbc_dict, f)

        return lower_nlbc

    def get_upper_nlbc(self, propagator: HelmholtzPadeSolver, n_x):
        gamma = propagator.env.n2m1_profile(0, propagator.env.z_max+1, propagator.freq_hz) - propagator.env.n2m1_profile(0, propagator.env.z_max, propagator.freq_hz)
        beta = propagator.env.n2m1_profile(0, propagator.env.z_max, propagator.freq_hz) - gamma * propagator.env.z_max
        q = 'upper', propagator.k0, propagator.dx, propagator.dz, propagator.pade_order, propagator.z_order, propagator.spe, beta, gamma
        if q not in self.nlbc_dict or self.nlbc_dict[q].coefs.shape[0] < n_x:
            self.nlbc_dict[q] = propagator.calc_upper_nlbc(beta, gamma)
        upper_nlbc = self.nlbc_dict[q]

        with open(self.file_name, 'wb') as f:
            pickle.dump(self.nlbc_dict, f)

        return upper_nlbc


def d2a_n_eq_ba_n(b):
    c1 = (b+2-cm.sqrt(b**2+4*b))/2
    c2 = 1.0 / c1
    return [c1, c2][abs(c1) > abs(c2)]


def sqr_eq(a, b, c):
    c1 = (-b + cm.sqrt(b**2 - 4 * a * c)) / (2 * a)
    c2 = (-b - cm.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
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