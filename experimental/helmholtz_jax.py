from dataclasses import dataclass

from jax import tree_util
import jax
import lineax
from jax import numpy as jnp

from experimental.grid_optimizer import get_optimal_grid
from experimental.utils import bessel_ratio_4th_order
from propagators._utils import pade_propagator_coefs
from transforms.fcc_fourier import FCCAdaptiveFourier
import numpy as np
import cmath as cm

import math as fm


@dataclass
class RegularGrid:
    start: float
    dx: float
    n: int

    def interval_indexes(self, a: float, b: float):
        a_i = min(round(max(a - self.start, 0.0) // self.dx), self.n)
        b_i = self.n - min(round(max(self.start + (self.n - 1) * self.dx - b, 0.0) // self.dx), self.n)
        return a_i, b_i

    def array_grid(self, a_i, b_i):
        return jnp.arange(max(a_i, 0), min(b_i, self.n)) * self.dx

    def __eq__(self, other):
        return isinstance(other, type(self)) and ((self.start, self.dx, self.n) == (other.start, other.dx, other.n))

    def __hash__(self):
        return hash((self.start, self.dx, self.n))

    def _tree_flatten(self):
        dynamic = ()
        static = {
            'start': self.start,
            'dx': self.dx,
            'n': self.n
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(**static)


tree_util.register_pytree_node(RegularGrid,
                               RegularGrid._tree_flatten,
                               RegularGrid._tree_unflatten)


class AbstractWaveSpeedModel:

    def __call__(self, z):
        pass

    def on_regular_grid(self, z_grid: RegularGrid):
        return self(z_grid.array_grid(0, 100000000))

    def support(self):
        pass


class AbstractTerrainModel:

    def __call__(self, x):
        pass

    def on_regular_grid(self, x_grid: RegularGrid):
        return self(x_grid.array_grid(0, 100000000))


class PiecewiseLinearTerrainModel(AbstractTerrainModel):

    def __init__(self, x_grid_m: jax.Array, height: jax.Array):
        self.x_grid_m = x_grid_m
        self.height = height

    def __call__(self, x):
        return jnp.interp(x, self.x_grid_m, self.height,
                          left='extrapolate', right='extrapolate')

    def _tree_flatten(self):
        dynamic = (self.x_grid_m, self.height)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(x_grid_m=dynamic[0], height=dynamic[1])


tree_util.register_pytree_node(PiecewiseLinearTerrainModel,
                               PiecewiseLinearTerrainModel._tree_flatten,
                               PiecewiseLinearTerrainModel._tree_unflatten)


class ConstWaveSpeedModel(AbstractWaveSpeedModel):

    def __init__(self, c0):
        self.c0 = c0

    def __call__(self, z):
        return z*0.0 + self.c0

    def _tree_flatten(self):
        dynamic = (self.c0,)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(c0=dynamic[0])


tree_util.register_pytree_node(ConstWaveSpeedModel,
                               ConstWaveSpeedModel._tree_flatten,
                               ConstWaveSpeedModel._tree_unflatten)


class LinearSlopeWaveSpeedModel(AbstractWaveSpeedModel):

    def __init__(self, c0: float, slope_degrees: float):
        self.c0 = c0
        self.slope_degrees = slope_degrees

    def __call__(self, z):
        return self.c0 + z * jnp.sin(jnp.radians(self.slope_degrees)) / jnp.cos(jnp.radians(self.slope_degrees))#jnp.tan(jnp.radians(self.slope_degrees))

    def _tree_flatten(self):
        dynamic = (self.c0, self.slope_degrees)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(c0=dynamic[0], slope_degrees=dynamic[1])


class PiecewiseLinearWaveSpeedModel(AbstractWaveSpeedModel):

    def __init__(self, z_grid_m: jax.Array, sound_speed: jax.Array):
        self.z_grid_m = z_grid_m
        self.sound_speed = sound_speed

    def __call__(self, z):
        return jnp.interp(z, self.z_grid_m, self.sound_speed,
                          left='extrapolate', right='extrapolate')

    def support(self):
        return 0.0, self.z_grid_m[-1]

    def _tree_flatten(self):
        dynamic = (self.z_grid_m, self.sound_speed)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(z_grid_m=dynamic[0], sound_speed=dynamic[1])


tree_util.register_pytree_node(PiecewiseLinearWaveSpeedModel,
                               PiecewiseLinearWaveSpeedModel._tree_flatten,
                               PiecewiseLinearWaveSpeedModel._tree_unflatten)


class BasisWaveSpeedModel(AbstractWaveSpeedModel):

    def __init__(self, basis_z_grid_m: jax.Array, basis_vals: jax.Array, coefs: jax.Array):
        self.basis_z_grid_m = basis_z_grid_m
        self.basis_vals = basis_vals
        self.coefs = coefs

    def __call__(self, z: jax.Array):
        return jnp.interp(z, self.basis_z_grid_m, self.basis_vals @ self.coefs)

    def _tree_flatten(self):
        dynamic = (self.coefs,)
        static = {
            "basis_z_grid_m": self.basis_z_grid_m,
            "basis_vals": self.basis_vals
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(coefs=dynamic[0], **static)


tree_util.register_pytree_node(BasisWaveSpeedModel,
                               BasisWaveSpeedModel._tree_flatten,
                               BasisWaveSpeedModel._tree_unflatten)


class AbstractRhoModel:

    def __call__(self, z):
        pass


class StaircaseRhoModel(AbstractRhoModel):

    def __init__(self, heights, vals):
        self.heights = jnp.array(heights)
        self.vals = jnp.array(vals)

    def __call__(self, z):
        indexes = jnp.searchsorted(self.heights, z, side='right') - 1
        return self.vals[indexes]

    def _tree_flatten(self):
        dynamic = (self.heights, self.vals)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(heights=dynamic[0], vals=dynamic[1])



@jax.jit
def sqr_eq(a, b, c):
    c1 = (-b + jnp.sqrt(b**2 - 4 * a * c + 0j)) / (2 * a)
    c2 = (-b - jnp.sqrt(b ** 2 - 4 * a * c + 0j)) / (2 * a)
    return jax.lax.select(abs(c1) > abs(c2), c2, c1)


@dataclass
class HelmholtzMeshParams2D:
    x_size_m: float
    z_size_m: float
    dx_output_m: float = None
    x_n_upper_bound: int = None
    dz_output_m: float = None
    z_n_upper_bound: int = None

    def __post_init__(self):
        if self.x_n_upper_bound is None and self.dx_output_m is None:
            raise ValueError("one of x_n_upper_bound or dx_output_m should be specified")
        if self.x_n_upper_bound is not None and self.dx_output_m is not None:
            raise ValueError("Only one of x_n_upper_bound or dx_output_m should be specified, not both")
        if self.z_n_upper_bound is None and self.dz_output_m is None:
            raise ValueError("one of z_n_upper_bound or dz_output_m should be specified")
        if self.z_n_upper_bound is not None and self.dz_output_m is not None:
            raise ValueError("Only one of z_n_upper_bound or dz_output_m should be specified, not both")


class RationalHelmholtzPropagator:

    @classmethod
    def create(cls, freq_hz: float, wave_speed, kz_max, k_bounds, precision, mesh_params: HelmholtzMeshParams2D,
               rho=None, lower_terrain=None):
        dx_max = mesh_params.dx_output_m or mesh_params.x_size_m / (round(mesh_params.x_n_upper_bound / 2) - 1)
        dz_max = mesh_params.dz_output_m or mesh_params.z_size_m / (round(mesh_params.z_n_upper_bound / 2) - 1)

        beta, dx_computational_m, dz_computational_m = get_optimal_grid(
            kz_max, k_bounds[0], k_bounds[1], precision / mesh_params.x_size_m,
            dx_max=dx_max,
            dz_max=dz_max)

        if mesh_params.dx_output_m:
            x_output_step = fm.ceil(mesh_params.dx_output_m / dx_computational_m)
        if mesh_params.dz_output_m:
            z_output_step = fm.ceil(mesh_params.dz_output_m / dz_computational_m)
        if mesh_params.x_n_upper_bound:
            x_output_step = fm.floor(mesh_params.dx_output_m / dx_computational_m)
        if mesh_params.z_n_upper_bound:
            z_output_step = fm.floor(mesh_params.dz_output_m / dz_computational_m)

        dx_computational_m = mesh_params.dx_output_m / x_output_step
        dz_computational_m = mesh_params.dz_output_m / z_output_step

        x_n = fm.ceil(mesh_params.x_size_m / dx_computational_m) + 1
        z_n = fm.ceil(mesh_params.z_size_m / dz_computational_m) + 1
        x_c_grid = jnp.arange(0, x_n) * dx_computational_m

        lower_terrain_mask = np.ones(shape=(x_n, z_n), dtype=complex)
        if lower_terrain is not None:
            t = np.array(np.round((lower_terrain(x_c_grid)) / dz_computational_m), dtype=int)
            for i in range(x_n):
                lower_terrain_mask[i, 0:t[i]] = 0.0

        print(f'beta: {beta}, dx: {dx_computational_m}, dz: {dz_computational_m}')

        return cls(
            order=(7, 8),
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
            lower_terrain_mask=jnp.array(lower_terrain_mask)
        )

    def __init__(self, order: tuple[int, int], beta: float, dx_m: float, dz_m: float, x_n: int, z_n: int,
                 x_grid_scale: int, z_grid_scale: int, freq_hz: float, wave_speed, lower_terrain_mask,
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
            self.coefs = jnp.array(coefs, dtype=complex)
        else:
            t = pade_propagator_coefs(pade_order=self.order, beta=self.beta, dx=self.dx_m)[0]
            a = [list(v) for v in t]
            self.coefs = jnp.array(a, dtype=complex)
            self.coefs_t = a
        self.freq_hz = freq_hz
        self.alpha = 1 / 12
        self.inv_z_transform_tau = 10 ** (3 / self.x_n)
        self.wave_speed = wave_speed
        self.rho = rho
        self._prepare_het_arrays()
        self.lower_nlbc_coefs = None  # lower_nlbc_coefs if lower_nlbc_coefs is not None else self._calc_nlbc(self.het[0])
        self.upper_nlbc_coefs = upper_nlbc_coefs if upper_nlbc_coefs is not None else self._calc_nlbc(self.het[-1], 0*(self.het[-1] - self.het[-2])/self.dz_m)

        self.lower_terrain_mask = lower_terrain_mask

    def _prepare_het_arrays(self):
        self.het = jnp.array((2 * jnp.pi * self.freq_hz / self.wave_speed.on_regular_grid(
            RegularGrid(start=0, dx=self.dz_m, n=self.z_n))) ** 2 / self.beta ** 2 - 1.0, dtype=complex)
        if self.rho is not None:
            self.rho_v = self.rho.on_regular_grid(RegularGrid(start=0, dx=self.dz_m, n=self.z_n))
            self.rho_v_min = self.rho.on_regular_grid(RegularGrid(start=self.dz_m / 2, dx=self.dz_m, n=self.z_n - 2))
            self.rho_v_pls = self.rho.on_regular_grid(
                RegularGrid(start=self.dz_m * 3 / 2, dx=self.dz_m, n=self.z_n - 2))
        else:
            self.rho_v = jnp.ones(self.z_n)
            self.rho_v_min = jnp.ones(self.z_n - 2)
            self.rho_v_pls = jnp.ones(self.z_n - 2)

    def x_computational_grid(self):
        return jnp.arange(0, self.x_n) * self.dx_m

    def x_output_grid(self):
        return self.x_computational_grid()[::self.x_grid_scale]

    def z_computational_grid(self):
        return jnp.arange(0, self.z_n) * self.dz_m

    def z_output_grid(self):
        return self.z_computational_grid()[::self.z_grid_scale]

    def _tree_flatten(self):
        dynamic = (self.wave_speed, self.rho)
        static = {
            'order': self.order,
            'beta': self.beta,
            'dx_m': self.dx_m,
            'dz_m': self.dz_m,
            'x_n': self.x_n,
            'z_n': self.z_n,
            'coefs': self.coefs_t,
            'x_grid_scale': self.x_grid_scale,
            'z_grid_scale': self.z_grid_scale,
            'freq_hz': self.freq_hz,
            'lower_nlbc_coefs': self.lower_nlbc_coefs,
            'upper_nlbc_coefs': self.upper_nlbc_coefs,
            'lower_terrain_mask': self.lower_terrain_mask
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(wave_speed=dynamic[0], rho=dynamic[1], **static)
        return unf

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

        return jnp.array(coefs)

    @jax.jit
    def _Crank_Nikolson_propagate_no_rho_4th_order(self, a, b, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        alpha = self.alpha
        c_a = alpha * (self.beta * self.dz_m) ** 2 + a + alpha * a * (self.beta * self.dz_m) ** 2 * self.het
        c_b = alpha * (self.beta * self.dz_m) ** 2 + b + alpha * b * (self.beta * self.dz_m) ** 2 * self.het
        d_a = (self.beta * self.dz_m) ** 2 * (1 - 2 * alpha) - 2 * a + (a * (self.beta * self.dz_m) ** 2 - 2 * a * alpha * (self.beta * self.dz_m) ** 2) * self.het
        d_b = (self.beta * self.dz_m) ** 2 * (1 - 2 * alpha) - 2 * b + (b * (self.beta * self.dz_m) ** 2 - 2 * b * alpha * (self.beta * self.dz_m) ** 2) * self.het

        rhs = d_a * initial
        rhs = rhs.at[1::].set(rhs[1::] + c_a[:-1:] * initial[:-1:])
        rhs = rhs.at[:-1:].set(rhs[:-1:] + c_a[1::] * initial[1::])
        d_b = d_b.at[0].set(lower_bound[0])
        d_b = d_b.at[-1].set(upper_bound[1])
        diag_1 = c_b[1::]
        diag_1 = diag_1.at[0].set(lower_bound[1])
        diag_m1 = c_b[:-1:]
        diag_m1 = diag_m1.at[-1].set(upper_bound[0])
        rhs = rhs.at[0].set(lower_bound[2])
        rhs = rhs.at[-1].set(upper_bound[2])
        tridiag_op = lineax.TridiagonalLinearOperator(d_b, diag_m1, diag_1)
        res = lineax.linear_solve(tridiag_op, rhs)
        return res.value

    @jax.jit
    def _Crank_Nikolson_propagate_rho_4th_order(self, a, b, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        alpha_m = self.alpha * self.rho_v[1:-1]
        tau2 = (self.beta * self.dz_m) ** 2

        c_a_left = 1 / self.rho_v_min * (alpha_m * tau2 + a * self.rho_v[1:-1:] + a * alpha_m * tau2 * self.het[:-2:])
        c_a_right = 1 / self.rho_v_pls * (alpha_m * tau2 + a * self.rho_v[1:-1:] + a * alpha_m * tau2 * self.het[2::])
        c_b_left = 1 / self.rho_v_min * (alpha_m * tau2 + b * self.rho_v[1:-1:] + b * alpha_m * tau2 * self.het[:-2:])
        c_b_right = 1 / self.rho_v_pls * (alpha_m * tau2 + b * self.rho_v[1:-1:] + b * alpha_m * tau2 * self.het[2::])

        het_mid2 = 1 / self.rho_v_pls + 1 / self.rho_v_min

        d_a = tau2 * (1 - alpha_m * het_mid2) - a * het_mid2 * \
              self.rho_v[1:-1:] + a * tau2 * self.het[1:-1:] - a * alpha_m * tau2 * self.het[1:-1:] * het_mid2

        d_b = tau2 * (1 - alpha_m * het_mid2) - b * het_mid2 * \
              self.rho_v[1:-1:] + b * tau2 * self.het[1:-1:] - b * alpha_m * tau2 * self.het[1:-1:] * het_mid2

        rhs = jnp.concatenate((jnp.array([0.0]), d_a, jnp.array([0.0]))) * initial
        rhs = rhs.at[1::].add(jnp.concatenate((c_a_left, jnp.array([0.0]))) * initial[:-1:])
        rhs = rhs.at[:-1:].add(jnp.concatenate((jnp.array([0.0]), c_a_right)) * initial[1::])
        rhs = rhs.at[0].set(lower_bound[2])
        rhs = rhs.at[-1].set(upper_bound[2])

        d_b = jnp.concatenate((jnp.array([lower_bound[0]]), d_b, jnp.array([upper_bound[1]])))
        c_b_left = jnp.concatenate((c_b_left, jnp.array([upper_bound[0]])))
        c_b_right = jnp.concatenate((jnp.array([lower_bound[1]]), c_b_right))

        tridiag_op = lineax.TridiagonalLinearOperator(d_b, c_b_left, c_b_right)
        res = lineax.linear_solve(tridiag_op, rhs)
        return res.value

    @jax.jit
    def _Crank_Nikolson_propagate(self, a, b, initial, lower_bound=(1, 0, 0), upper_bound=(0, 1, 0)):
        return jax.lax.cond(
            self.rho is None,
            lambda: self._Crank_Nikolson_propagate_no_rho_4th_order(a, b, initial, lower_bound, upper_bound),
            lambda: self._Crank_Nikolson_propagate_rho_4th_order(a, b, initial, lower_bound, upper_bound)
        )

    @jax.jit
    def _step(self, initial, upper_convolution):
        upper_field = jnp.zeros(len(self.coefs), dtype=complex)

        def substep(i, val):
            y0, upper_field = val
            upper_bound = 1, -self.upper_nlbc_coefs[0, i, i], upper_convolution[i] + self.upper_nlbc_coefs[0, i] @ upper_field
            y1 = self._Crank_Nikolson_propagate(self.coefs[i][0], self.coefs[i][1], y0, upper_bound=upper_bound)
            upper_field = upper_field.at[i].set(y1[-1])
            return y1, upper_field

        return jax.lax.fori_loop(0, len(self.coefs), substep, (initial, upper_field))

    def _convolution(self, a, b, i):

        def body_fun(ind, val):
            return val + a[ind] @ b[i-ind]

        return jax.lax.fori_loop(1, len(a), body_fun, jnp.zeros(max(self.order), dtype=complex))

    @jax.jit
    def compute(self, initial):
        self._prepare_het_arrays()

        results = jnp.empty(shape=(self.x_n // self.x_grid_scale + 1, (self.z_n - 1) // self.z_grid_scale + 1), dtype=complex)
        results = results.at[0, :].set(initial[::self.z_grid_scale])

        def body_fun(i, val):
            y0, res, upper_field = val
            upper_convolution = self._convolution(self.upper_nlbc_coefs, upper_field, i)
            y1, upper_field_i = self._step(y0, upper_convolution)
            y1 = y1 * self.lower_terrain_mask[i]
            res = jax.lax.cond(i % self.x_grid_scale == 0, lambda: res.at[i // self.x_grid_scale, :].set(y1[::self.z_grid_scale]), lambda: res)
            upper_field = upper_field.at[i].set(upper_field_i)
            return y1, res, upper_field
        _, results, _ = jax.lax.fori_loop(1, self.x_n, body_fun,
                                          (initial, results, jnp.zeros(shape=(round(self.x_n), max(self.order)), dtype=complex)))

        return results


tree_util.register_pytree_node(RationalHelmholtzPropagator,
                               RationalHelmholtzPropagator._tree_flatten,
                               RationalHelmholtzPropagator._tree_unflatten)
tree_util.register_pytree_node(LinearSlopeWaveSpeedModel,
                               LinearSlopeWaveSpeedModel._tree_flatten,
                               LinearSlopeWaveSpeedModel._tree_unflatten)
tree_util.register_pytree_node(StaircaseRhoModel,
                               StaircaseRhoModel._tree_flatten,
                               StaircaseRhoModel._tree_unflatten)
