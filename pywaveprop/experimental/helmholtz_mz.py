import numpy as np
from multiZ.mdual import *
import math as fm
import cmath as cm

from pywaveprop.experimental.grid_optimizer import get_optimal_grid
from pywaveprop.experimental.helmholtz_common import HelmholtzMeshParams2D
from pywaveprop.propagators._utils import pade_propagator_coefs
from pywaveprop.experimental.utils import sqr_eq, bessel_ratio_4th_order
import jax
import jax.numpy as jnp

from pywaveprop.transforms.fcc_fourier import FCCAdaptiveFourier


def tridiag_method(lower, diag, upper, rhs):
    """
    Solve a tridiagonal system of equations using the Thomas algorithm.

    Parameters:
    -----------
    lower : np.ndarray[complex, ndim=1]
        Lower diagonal of the tridiagonal matrix (length n-1)
    diag : np.ndarray[complex, ndim=1]
        Main diagonal of the tridiagonal matrix (length n)
    upper : np.ndarray[complex, ndim=1]
        Upper diagonal of the tridiagonal matrix (length n-1)
    rhs : np.ndarray[complex, ndim=1]
        Right-hand side vector (length n)
    res : np.ndarray[complex, ndim=1]
        Array to store the solution (length n)

    Note: This function modifies diag, rhs, and res in-place.
    """
    n = len(diag)

    # Forward elimination
    for i in range(1, n):
        w = lower[i - 1] / diag[i - 1]
        diag[i] = diag[i] - w * upper[i - 1]
        rhs[i] = rhs[i] - w * rhs[i - 1]

    # Backward substitution
    res = diag*0
    res[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        res[i] = (rhs[i] - upper[i] * res[i + 1]) / diag[i]
    return res


class RationalHelmholtzPropagator:

    @classmethod
    def create(cls, freq_hz: float, wave_speed, kz_max, k_bounds, precision, mesh_params: HelmholtzMeshParams2D,
               rho=None, lower_terrain=None, rational_approx_order=(7, 8)):
        dx_max = mesh_params.dx_output_m or mesh_params.x_size_m / (round(mesh_params.x_n_upper_bound / 2) - 1)
        dz_max = mesh_params.dz_output_m or mesh_params.z_size_m / (round(mesh_params.z_n_upper_bound / 2) - 1)

        if rational_approx_order == None:
            orders = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),]
        else:
            orders = [rational_approx_order]

        cur_best = fm.inf
        for order in orders:
            beta_t, dx_computational_m_t, dz_computational_m_t = get_optimal_grid(
                kz_max, k_bounds[0], k_bounds[1], precision / mesh_params.x_size_m,
                dx_max=dx_max,
                dz_max=dz_max,
                propagator_order=order
            )

            if fm.isnan(beta_t):
                continue

            if mesh_params.dx_output_m:
                x_output_step_t = fm.ceil(mesh_params.dx_output_m / dx_computational_m_t)
                dx_computational_m_t = mesh_params.dx_output_m / x_output_step_t
            if mesh_params.dz_output_m:
                z_output_step_t = fm.ceil(mesh_params.dz_output_m / dz_computational_m_t)
                dz_computational_m_t = mesh_params.dz_output_m / z_output_step_t
            if mesh_params.x_n_upper_bound:
                x_output_step_t = fm.floor(dx_max / dx_computational_m_t)
                dx_computational_m_t = dx_max / x_output_step_t
            if mesh_params.z_n_upper_bound:
                z_output_step_t = fm.floor(dz_max / dz_computational_m_t)
                dz_computational_m_t = dz_max / z_output_step_t

            x_n_t = fm.ceil(mesh_params.x_size_m / dx_computational_m_t) + 1
            z_n_t = fm.ceil(mesh_params.z_size_m / dz_computational_m_t) + 1

            if x_n_t*z_n_t*order[1] < cur_best:
                cur_best = x_n_t*z_n_t*order[1]
                x_n = x_n_t
                z_n = z_n_t
                x_output_step = x_output_step_t
                z_output_step = z_output_step_t
                beta = beta_t
                dx_computational_m = dx_computational_m_t
                dz_computational_m = dz_computational_m_t
                rational_approx_order = order

        print(f'rational_approx_order: {rational_approx_order}, '
              f'beta: {beta}, '
              f'dx: {dx_computational_m}, '
              f'dz: {dz_computational_m}')

        return cls(
            order=rational_approx_order,
            beta=beta,
            dx_m=dx_computational_m,
            dz_m=dz_computational_m,
            x_n=x_n,
            z_n=z_n,
            x_grid_scale=x_output_step,
            z_grid_scale=z_output_step,
            freq_hz=freq_hz,
            wave_speed=wave_speed,
            rho=rho,
            lower_terrain=lower_terrain
        )

    def __init__(self, order: tuple[int, int], beta: float, dx_m: float, dz_m: float, x_n: int, z_n: int,
                 x_grid_scale: int, z_grid_scale: int, freq_hz: float, wave_speed, lower_terrain_mask=None,
                 rho=None, lower_terrain=None, coefs=None, lower_nlbc_coefs=None, upper_nlbc_coefs=None):
        self.order = order
        self.beta = beta
        self.dx_m = dx_m
        self.dz_m = dz_m
        self.x_n = x_n
        self.z_n = z_n
        self.x_grid_scale = x_grid_scale
        self.z_grid_scale = z_grid_scale
        if coefs is not None:
            self.coefs_t = coefs
            self.coefs = np.array(coefs, dtype=complex)
        else:
            t = pade_propagator_coefs(pade_order=self.order, beta=self.beta, dx=self.dx_m)[0]
            a = [list(v) for v in t]
            self.coefs = np.array(a, dtype=complex)
            self.coefs_t = a
        self.freq_hz = freq_hz
        self.alpha = 1 / 12
        self.inv_z_transform_tau = 10 ** (3 / self.x_n)
        self.wave_speed = wave_speed
        self.rho = rho
        self.lower_terrain = lower_terrain
        if lower_terrain_mask is not None:
            self.lower_terrain_mask = lower_terrain_mask
        else:
            self.lower_terrain_mask = np.ones(shape=(self.x_n, self.z_n), dtype=complex)
        self._prepare_het_arrays()
        self.lower_nlbc_coefs = None  # lower_nlbc_coefs if lower_nlbc_coefs is not None else self._calc_nlbc(self.het[0])
        self.upper_nlbc_coefs = upper_nlbc_coefs if upper_nlbc_coefs is not None else self._calc_nlbc(self.het[-1], 0*(self.het[-1] - self.het[-2])/self.dz_m)


    def _prepare_het_arrays(self):
        self.het = (2 * np.pi * self.freq_hz / self.wave_speed(np.arange(0, self.z_n)*self.dz_m)) ** 2 / self.beta ** 2 - 1.0

        if self.rho is not None:
            self.rho_v = self.rho(np.arange(0, self.z_n)*self.dz_m)
            self.rho_v_min = self.rho(np.arange(0, self.z_n-2)*self.dz_m + self.dz_m / 2)
            self.rho_v_pls = self.rho(np.arange(0, self.z_n-2)*self.dz_m + self.dz_m * 3 / 2)
        else:
            self.rho_v = np.ones(self.z_n)
            self.rho_v_min = np.ones(self.z_n - 2)
            self.rho_v_pls = np.ones(self.z_n - 2)

        if self.lower_terrain is not None:
            terrain_heights = np.round(self.lower_terrain(np.arange(0, self.x_n)*self.dx_m) / self.dz_m)
            # Create a grid of z indices: shape (x_n, z_n)
            z_indices = np.arange(self.z_n)[np.newaxis, :]
            # Broadcast terrain heights to compare: shape (x_n, 1)
            terrain_heights_broadcast = terrain_heights[:, np.newaxis]
            # Mask is 0 where z_index < terrain_height, 1 otherwise
            self.lower_terrain_mask = np.where(z_indices < terrain_heights_broadcast, 0.0 + 0.0j, 1.0 + 0.0j)
        else:
            self.lower_terrain_mask = np.ones(shape=(self.x_n, self.z_n), dtype=complex)

    def x_computational_grid(self):
        return np.arange(0, self.x_n) * self.dx_m

    def x_output_grid(self):
        return self.x_computational_grid()[::self.x_grid_scale]

    def z_computational_grid(self):
        return np.arange(0, self.z_n) * self.dz_m

    def z_output_grid(self):
        return self.z_computational_grid()[::self.z_grid_scale]

    def _calc_nlbc(self, beta, gamma=0.0):

        if abs(gamma) < 10*jnp.finfo(complex).eps:
            inv_z_transform_rtol = 1e-7
            def diff_eq_solution_ratio(s):
                a_m1 = 1 - self.alpha * (self.beta * self.dz_m) ** 2 * (s - beta)
                a_1 = 1 - self.alpha * (self.beta * self.dz_m) ** 2 * (s - beta)
                c = -2 + (2 * self.alpha - 1) * (self.beta * self.dz_m) ** 2 * (s - beta)
                mu = sqr_eq(a_1, c, a_m1)
                return 1 / mu
        else:
            inv_z_transform_rtol = 1e-11
            b = self.alpha * gamma * self.dz_m * (self.beta * self.dz_m) ** 2
            d = gamma * self.dz_m * (self.beta * self.dz_m) ** 2 - 2 * b

            def diff_eq_solution_ratio(s):
                a_m1 = 1 - self.alpha * (self.beta * self.dz_m) ** 2 * (s - beta) - b
                a_1 = 1 - self.alpha * (self.beta * self.dz_m) ** 2 * (s - beta) + b
                c = -2 + (2 * self.alpha - 1) * (self.beta * self.dz_m) ** 2 * (s - beta)
                return bessel_ratio_4th_order(a_m1, a_1, b, c, d, self.z_n - 1, inv_z_transform_rtol)


        num_roots, den_roots = self.coefs[:, 0], self.coefs[:, 1]
        m_size = self.coefs.shape[0]
        tau = self.inv_z_transform_tau
        if max(self.order) == 1:
            @jax.jit
            def nlbc_transformed(t):
                t = tau * jnp.exp(1j * t)
                return diff_eq_solution_ratio(((1 - t) / (-num_roots[0] + den_roots[0] * t)))
        else:
            @jax.jit
            def nlbc_transformed(f):
                t = tau * jnp.exp(1j * f)
                matrix_a = jnp.diag(den_roots, 0) - jnp.diag(num_roots[1:], -1)
                matrix_a = matrix_a.at[0, -1].add(-num_roots[0])
                matrix_a = matrix_a.at[0, 0].set(matrix_a[0, 0] * t)
                matrix_b = jnp.diag(-jnp.ones(m_size), 0) + jnp.diag(jnp.ones(m_size - 1), -1) + 0j
                matrix_b = matrix_b.at[0, -1].set(1.0)
                matrix_b = matrix_b.at[0, 0].multiply(t)
                w, vr = jnp.linalg.eig(jnp.linalg.inv(matrix_a) @ matrix_b)
                r = jnp.diag(jnp.array([diff_eq_solution_ratio(a) for a in w]))
                res = vr.dot(r).dot(jnp.linalg.inv(vr))
                return res.reshape(m_size ** 2)

        fcca = FCCAdaptiveFourier(2 * cm.pi, -np.arange(0, self.x_n), rtol=inv_z_transform_rtol)

        coefs = (tau ** np.repeat(np.arange(0, self.x_n)[:, np.newaxis], m_size ** 2, axis=1) / (2 * cm.pi) *
                 fcca.forward(lambda t: np.array(nlbc_transformed(t), dtype=complex), 0, 2 * cm.pi)).reshape((self.x_n, m_size, m_size))

        return np.array(coefs)

    def _Crank_Nikolson_propagate_no_rho_4th_order(self, a, b, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        alpha = 1 / 12
        c_a = alpha * (self.beta * self.dz_m) ** 2 + a + alpha * a * (self.beta * self.dz_m) ** 2 * self.het
        c_b = alpha * (self.beta * self.dz_m) ** 2 + b + alpha * b * (self.beta * self.dz_m) ** 2 * self.het
        d_a = (self.beta * self.dz_m) ** 2 * (1 - 2 * alpha) - 2 * a + (
                    a * (self.beta * self.dz_m) ** 2 - 2 * a * alpha * (self.beta * self.dz_m) ** 2) * self.het
        d_b = (self.beta * self.dz_m) ** 2 * (1 - 2 * alpha) - 2 * b + (
                    b * (self.beta * self.dz_m) ** 2 - 2 * b * alpha * (self.beta * self.dz_m) ** 2) * self.het

        rhs = d_a * initial
        rhs[1::] += c_a[:-1:] * initial[:-1:]
        rhs[:-1:] += c_a[1::] * initial[1::]
        d_b[0] = lower_bound[0]
        d_b[-1] = upper_bound[1]
        diag_1 = np.copy(c_b[1::])
        diag_1[0] = lower_bound[1]
        diag_m1 = c_b[:-1:]
        diag_m1[-1] = upper_bound[0]
        rhs[0] = lower_bound[2]
        rhs[-1] = upper_bound[2]
        return tridiag_method(diag_m1, d_b, diag_1, rhs)

    def _Crank_Nikolson_propagate_rho_4th_order(self, a, b, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        pass

    def _Crank_Nikolson_propagate(self, a, b, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        if self.rho is None:
            return self._Crank_Nikolson_propagate_no_rho_4th_order(a, b, initial, lower_bound, upper_bound)
        else:
            return self._Crank_Nikolson_propagate_rho_4th_order(a, b, initial, lower_bound, upper_bound)

    def _step(self, initial, upper_convolution):
        upper_field = jnp.zeros(len(self.coefs), dtype=complex)

        def substep(i, val):
            y0, upper_field = val
            upper_bound = 1, -self.upper_nlbc_coefs[0, i, i], upper_convolution[i] + self.upper_nlbc_coefs[0, i] @ upper_field
            y1 = self._Crank_Nikolson_propagate(self.coefs[i][0], self.coefs[i][1], y0, upper_bound=upper_bound)
            upper_field = upper_field.at[i].set(y1[-1])
            return y1, upper_field

        v = (initial, upper_field)
        for i in range(len(self.coefs)):
            v = substep(i, v)

        return v

    def _convolution(self, a, b, i):

        def body_fun(ind, val):
            return val + a[ind] @ b[i-ind]

        v = np.zeros(max(self.order), dtype=complex)
        for i in range(1, len(a)):
            v = body_fun(i, v)

        return v

    @jax.jit
    def compute(self, initial):
        self._prepare_het_arrays()

        results = np.empty(shape=(self.x_n // self.x_grid_scale + 1, (self.z_n - 1) // self.z_grid_scale + 1), dtype=complex)
        results[0, :] = initial[::self.z_grid_scale]

        def body_fun(i, val):
            y0, res, upper_field = val
            upper_convolution = self._convolution(self.upper_nlbc_coefs, upper_field, i)
            y1, upper_field_i = self._step(y0, upper_convolution)
            y1 = y1 * self.lower_terrain_mask[i,:]
            if i % self.x_grid_scale == 0:
                res[i // self.x_grid_scale, :] = y1[::self.z_grid_scale]
            upper_field = upper_field.at[i].set(upper_field_i)
            return y1, res, upper_field

        v = (initial, results, jnp.zeros(shape=(round(self.x_n), max(self.order)), dtype=complex))
        for i in range(1, self.x_n):
            v = body_fun(i, v)

        _, results, _ = v
        return results
