from dataclasses import dataclass
import logging

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
        self.p_grid_is_regular = True
        self.x_computational_grid, self.dx = np.linspace(self.params.x_min_m, self.params.x_max_m, retstep=True)
        self.z_computational_grid, self.dz = np.linspace(self.params.z_min_m, self.params.z_max_m, retstep=True)

        self.quad_x_grid = np.empty((len(self.bodies), self.params.quadrature_points))
        self.quad_weights = np.empty((len(self.bodies), self.params.quadrature_points))
        for body_index, body in enumerate(self.bodies):
            if self.params.quadrature_points == 1:
                self.quad_x_grid[body_index] = np.array([(body.x1 + body.x2) / 2])
                self.quad_weights[body_index] = np.array([body.x2 - body.x1])
            else:
                self.quad_x_grid[body_index], dx = np.linspace(body.x1, body.x2, self.params.quadrature_points)
                self.quad_weights[body_index] = np.concatenate(([dx / 2], np.repeat(dx, self.params.quadrature_points - 2), [dx / 2]))

        self.qrc_q = self.p_computational_grid * 0 + 1 / cm.sqrt(2*cm.pi)

        self.super_ker_size = self.params.p_grid_size * self.params.quadrature_points * len(bodies)
        gms = self.super_ker_size**2 * 16 /1024/1024
        logging.debug("matrix %d x %d, size = %d mb", self.super_ker_size, self.super_ker_size, gms)

    def green_function(self, x, xsh, p):
        if len(p.shape) == 2:
            xv, xshv, pv = x, xsh, p
        else:
            xv, xshv, pv = np.meshgrid(x, xsh, p, indexing='ij')
        gv = -1 / (2 * self._gamma(pv)) * np.exp(-self._gamma(pv) * np.abs(xv - xshv))
        return np.squeeze(gv)

    def _gamma(self, p):
        alpha = self.params.alpha
        a = np.sqrt((np.sqrt((p**2 - self.k0**2)**2 + alpha**2) - (self.k0**2 - p**2)) / 2)
        d = -np.sqrt((np.sqrt((p ** 2 - self.k0 ** 2) ** 2 + alpha ** 2) + (self.k0 ** 2 - p ** 2)) / 2)
        return a + d

    def _ker(self, body_number, i, j):
        pv, pshv = np.meshgrid(self.p_computational_grid, self.p_computational_grid, indexing='ij')
        return self.k0**2 / cm.sqrt(2*cm.pi) * self._body_z_fourier(body_number, self.quad_x_grid[body_number][i], pv - pshv) * \
               self.green_function(self.quad_x_grid[body_number][i], self.quad_x_grid[body_number][j], pv) * self.quad_weights[body_number][i]

    def _rhs(self, body_number, i):
        pv, pshv = np.meshgrid(self.p_computational_grid, self.p_computational_grid, indexing='ij')
        m = self._body_z_fourier(body_number, self.quad_x_grid[body_number][i], pv - pshv) * \
            self.green_function(self.quad_x_grid[body_number][i], 0, pv) * self.quad_weights[body_number][i]
        if self.p_grid_is_regular:
            return np.sum(m, axis=1) * self.d_p    #TODO improve integral approximation??
        else:
            raise Exception("not yet supported")

    def _body_z_fourier(self, body_number, x_m, p):
            intervals = self.bodies[body_number].get_intersecting_intervals(x_m)
            res = p*0
            for (a, b) in intervals:
                f = 1j / p * (np.exp(-1j * p * b) - np.exp(-1j * p * a))
                f[abs(p) < 0.0000001] = b - a
                res += f

            res *= 1 / cm.sqrt(2*cm.pi) * self.bodies[body_number].eps_r - 1
            return res

    def _super_ker(self):
        sk = np.empty((self.super_ker_size, self.super_ker_size))
        ks = self.params.p_grid_size
        for body_i in range(0, len(self.bodies)):
            for body_j in range(0, len(self.bodies)):
                for x_i in range(0, len(self.quad_x_grid[body_i])):
                    for x_j in range(0, len(self.quad_x_grid[body_j])):
                        sk[ks*x_i:ks*(x_i + 1):, ks*x_j:ks*(x_j + 1):] = self._ker(body_i, x_i, x_j)
        return sk

    def _super_rhs(self):
        rs = np.empty(self.super_ker_size)
        ks = self.params.p_grid_size
        for body_i in range(0, len(self.bodies)):
            for x_i in range(0, len(self.quad_x_grid[body_i])):
                rs[ks*x_i:ks*(x_i + 1):] = self._rhs(body_i, x_i)
        return rs

    def calculate(self):
        ker = self._super_ker()
        rhs = self._super_rhs()
        super_phi = np.linalg.solve(ker*self.d_p, rhs)

        ks = self.params.p_grid_size
        psi = self.green_function(self.x_computational_grid, 0, self.p_computational_grid) * 1 / cm.sqrt(2*cm.pi)
        for body_i in range(0, len(self.bodies)):
            for x_i in range(0, len(self.quad_x_grid[body_i])):
                phi = super_phi[ks*x_i:ks*(x_i + 1):]
                psi += -self.k0**2/cm.sqrt(2*cm.pi) * self.green_function(
                    self.x_computational_grid, self.quad_x_grid[x_i], self.p_computational_grid) * phi