"""
Shift-map (Beilis--Tappert) finite-difference parabolic equations, used as a
baseline for the terrain-following rational propagator.

With ``zeta = z - T(x)`` and the gauge

    u(x, z) = v(x, zeta) exp(i k0 T'(x) zeta + (i k0 / 2) int_0^x T'(s)^2 ds),

the narrow-angle PE over the terrain ``T`` becomes the narrow-angle PE over flat
ground for ``v`` with the modified index ``n^2(zeta + T) - 2 zeta T''(x)``.  The
wide-angle shift-map PE applies the same transformation and replaces the
narrow-angle operator by a wide-angle one.  The classical finite-difference
schemes are single rational factors of the propagator, with ``X = L``:

* ``'na'``: ``u_x = (i k0 / 2) X u``, Crank--Nicolson: ``(1 + a X) / (1 + b X)`` with
  ``a = -b = i k0 dx / 4``;
* ``'wa'``: Claerbout ``(1 + X / 4) u_x = (i k0 / 2) X u``, Crank--Nicolson:
  ``a = 1/4 + i k0 dx / 4``, ``b = 1/4 - i k0 dx / 4``;

with second-order central differences in height (``alpha = 0``).  ``'pade'``
gives the higher-order rational propagator in shift-map coordinates instead.
The domain is truncated from above either by the discrete nonlocal boundary
condition or by an absorbing layer (complex index ``i A ((zeta - zeta_a) / D)^2``
over the thickness ``D`` and a zero field at the top), as in the literature.

All marching is done by :class:`pywaveprop.helmholtz_tf_jax.TerrainFollowingHelmholtzPropagator`
with an identity map, so that the schemes differ only in the approximation.
"""
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np

from pywaveprop.helmholtz_tf_jax import TerrainFollowingHelmholtzPropagator, TFMedium
from pywaveprop.terrain_transform import OrthogonalTerrainMap


@dataclass
class ShiftMapTerrain:
    """Terrain ``T(x)`` with its first two derivatives (vectorised callables)."""
    T: Callable
    dT: Callable
    d2T: Callable

    @staticmethod
    def flat():
        z = lambda x: np.zeros_like(np.asarray(x, dtype=float))  # noqa: E731
        return ShiftMapTerrain(z, z, z)

    @staticmethod
    def linear(slope):
        return ShiftMapTerrain(lambda x: slope * np.asarray(x, dtype=float),
                               lambda x: np.full_like(np.asarray(x, dtype=float), slope),
                               lambda x: np.zeros_like(np.asarray(x, dtype=float)))


class ShiftMapPropagator:
    """Shift-map PE on the grid ``x_i = i dx``, ``zeta_j = j dz``.

    ``k2(z)``: squared wavenumber as a function of the physical height (complex
    allowed).  ``top``: ``'nlbc'`` or ``'absorber'`` (then ``absorber_m`` is the
    thickness, included in ``z_size_m``, and ``absorber_strength`` the value of
    ``A``).  ``scheme``: ``'na'``, ``'wa'`` or ``('pade', (m, n))``.
    """

    def __init__(self, freq_hz, k0, k2: Callable, terrain: ShiftMapTerrain, x_size_m, z_size_m, dx_m, dz_m,
                 scheme='wa', alpha=None, top='nlbc', absorber_m=0.0, absorber_strength=0.0,
                 x_grid_scale=1, z_grid_scale=1, lower_bc='dirichlet', coarse_dxi_m=None, beta=None):
        self.k0 = float(k0)
        # reference wavenumber of the envelope u = psi exp(-i beta x); the classical
        # NA / WA schemes are written for beta = k0
        self.beta = self.k0 if (beta is None or not isinstance(scheme, tuple)) else float(beta)
        self.terrain = terrain
        x_n = int(round(x_size_m / dx_m)) + 1
        z_n = int(round(z_size_m / dz_m)) + 1
        self.dx, self.dz = dx_m, dz_m
        if isinstance(scheme, tuple):
            order, coefs = scheme[1], None
            alpha = 1 / 12 if alpha is None else alpha
        else:
            c = 1j * self.k0 * dx_m / 4
            coefs = [[c, -c]] if scheme == 'na' else [[0.25 + c, 0.25 - c]]
            order = (1, 1)
            alpha = 0.0 if alpha is None else alpha
        z_top = (z_n - 1) * dz_m
        zeta_a = z_top - absorber_m

        def k2_eff(x, zeta):
            x = np.asarray(x, dtype=float)
            zeta = np.asarray(zeta, dtype=float)
            val = np.asarray(k2(zeta + terrain.T(x)), dtype=complex) - 2 * self.k0 ** 2 * zeta * terrain.d2T(x)
            if top == 'absorber' and absorber_m > 0:
                t = np.clip((zeta - zeta_a) / absorber_m, 0.0, None)
                val = val + 1j * self.k0 ** 2 * absorber_strength * t ** 2
            return val

        tm = OrthogonalTerrainMap([0.0, x_size_m + 10 * dx_m], [0.0, 0.0], x_max_m=x_size_m,
                                  transition_m=(z_n - 7) * dz_m, z_ref_m=0.0, smoothing_m=0.0)
        tables = TerrainFollowingHelmholtzPropagator.build_tables(
            tm, TFMedium(k2_above=k2_eff), beta=self.beta, dxi=dx_m, deta=dz_m, x_n=x_n, z_n=z_n,
            coarse_dxi_m=coarse_dxi_m or dx_m)
        self.prop = TerrainFollowingHelmholtzPropagator(
            order=order, beta=self.beta, dx_m=dx_m, dz_m=dz_m, x_n=x_n, z_n=z_n, x_grid_scale=x_grid_scale,
            z_grid_scale=z_grid_scale, freq_hz=freq_hz, tables=tables, coefs=coefs, terrain_map=tm,
            flux_order=0, alpha=alpha, lower_bc=lower_bc, upper_bc='nlbc' if top == 'nlbc' else 'dirichlet')

    def zeta_grid(self):
        return np.arange(self.prop.z_n) * self.dz

    def gauge(self, x, zeta):
        """``u = v * gauge``; ``x`` (n,), ``zeta`` (m,) -> (n, m)."""
        x = np.asarray(x, dtype=float)
        xs = np.linspace(0.0, max(float(x.max()), 1e-9), 4001)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (self.terrain.dT(xs[1:]) ** 2 + self.terrain.dT(xs[:-1]) ** 2)
                                               * np.diff(xs))])
        integral = np.interp(x, xs, cum)
        return np.exp(1j * self.k0 * (self.terrain.dT(x)[:, None] * zeta[None, :] + 0.5 * integral[:, None]))

    def compute(self, u0):
        """``u0``: physical field on the starting line ``x = 0`` at the heights
        ``zeta + T(0)``.  Returns ``(x, z, u)`` on the output grid, ``z`` physical,
        ``u`` the envelope ``psi exp(-i beta x)``."""
        zeta = self.zeta_grid()
        v0 = np.asarray(u0) / self.gauge(np.zeros(1), zeta)[0]
        v = np.asarray(self.prop.compute(jnp.asarray(v0)))
        x = np.asarray(self.prop.x_output_grid())
        zo = zeta[::self.prop.z_grid_scale]
        u = v * self.gauge(x, zo)
        return x, zo[None, :] + self.terrain.T(x)[:, None], u
