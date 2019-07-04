from dataclasses import dataclass

import numpy as np
import cmath as cm


class ThinBody2d:

    def __init__(self, x1_m: float, x2_m: float, eps_r: complex):
        self.x1_m = x1_m
        self.x2_m = x2_m
        self.eps_r = eps_r

    def get_intersecting_intervals(self, x_m):
        pass


class Plate(ThinBody2d):

    def __init__(self, x0_m, z1_m, z2_m, width_m, eps_r):
        self.x1_m = x0_m - width_m / 2
        self.x2_m = x0_m + width_m / 2
        self.z1_m = z1_m
        self.z2_m = z2_m
        self.eps_r = eps_r

    def get_intersecting_intervals(self, x_m):
        if self.x1_m <= x_m <= self.x2_m:
            return [(self.z1_m, self.z2_m)]
        else:
            return []


@dataclass
class ThinScatteringComputationalParams:
    max_p_k0: float
    p_grid_size: int
    x_min_m: float
    x_max_m: float
    z_min_m: float
    z_max_m: float
    quadrature_points: int
    alpha: float


class ThinScattering:

    def __init__(self, src, bodies, params: ThinScatteringComputationalParams):
        self.bodies = bodies
        self.params = params
        self.k0 = 2 * cm.pi / src.wavelength
        self.max_p = self.params.max_p_k0 * self.k0
        self.p_computational_grid, self.d_p = np.linspace(-self.max_p, self.max_p, self.params.p_grid_size, retstep=True)

        if self.params.quadrature_points == 1:
            self.quad_x_grid = np.array([(self.bodies[0].x1 + self.bodies[0].x2) / 2])
            self.quad_weights = np.array([self.bodies[0].x2 - self.bodies[0].x1])
        else:
            self.quad_x_grid, dx = np.linspace(self.bodies[0].x1, self.bodies[0].x2, self.params.quadrature_points)
            self.quad_weights = np.concatenate(([dx / 2], np.repeat(dx, self.params.quadrature_points - 2), [dx / 2]))

    def green_function(self, x, xsh, p):
        xv, xshv, pv = np.meshgrid(x, xsh, p)
        gv = -1 / (2 * self._gamma(pv)) * np.exp(-self._gamma(pv) * np.abs(xv - xshv))
        return np.squeeze(gv)

    def _gamma(self, p):
        alpha = self.params.alpha
        a = np.sqrt((np.sqrt((p**2 - self.k0**2)**2 + alpha**2) - (self.k0**2 - p**2)) / 2)
        d = -np.sqrt((np.sqrt((p ** 2 - self.k0 ** 2) ** 2 + alpha ** 2) + (self.k0 ** 2 - p ** 2)) / 2)
        return a + d

    def _ker(self, body_number, i, j):
        pv, psh = np.meshgrid(self.p_computational_grid, self.p_computational_grid)

    def _body_z_fourier(self, body_number, x_m, p):
        intervals = self.bodies[body_number].get_intersecting_intervals(x_m)
        res = p*0
        for (a, b) in intervals:
            f = 1j / p * (np.exp(-1j * p * b) - np.exp(-1j * p * a))
            f[abs(p) < 0.0000001] = b - a
            res += f

        return res