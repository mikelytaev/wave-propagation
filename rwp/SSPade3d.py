import logging
import math as fm

from scipy import sparse as sparse, linalg as la

from rwp.field import Field3d
from propagators._utils import pade_propagator_coefs
from rwp.environment import EMEnvironment3d, LIGHT_SPEED
from rwp.propagators import Crank_Nikolson_propagator2, Crank_Nikolson_propagator2_4th_order

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class Pade3dPropagatorComputationParameters:

    def __init__(self, dx_wl, dy_wl, dz_wl, n_dx_out=1, n_dy_out=1, n_dz_out=1, pade_order=(1, 2), spe=False,
                 abs_layer_scale=0, yz_order=2, adi_iters=1, tol=1e-11):
        self.dx_wl = dx_wl
        self.dy_wl = dy_wl
        self.dz_wl = dz_wl
        self.n_dx_out = n_dx_out
        self.n_dy_out = n_dy_out
        self.n_dz_out = n_dz_out
        self.pade_order = pade_order
        self.abs_layer_scale = abs_layer_scale
        self.spe = spe
        self.yz_order = yz_order
        self.adi_iters = adi_iters
        self.tol = tol


class Pade3dPropagator:

    def __init__(self, env: EMEnvironment3d, freq_hz, comp_params: Pade3dPropagatorComputationParameters):
        self.env = env
        self.freq_hz = freq_hz
        self.comp_params = comp_params
        self.k0 = 2*fm.pi*self.freq_hz / LIGHT_SPEED
        wavelength = LIGHT_SPEED / self.freq_hz
        self.dx = wavelength * self.comp_params.dx_wl

        abs_y_length = (self.env.y_max - self.env.y_min) * self.comp_params.abs_layer_scale / 2
        y_min = self.env.y_min - abs_y_length
        y_max = self.env.y_max + abs_y_length
        self.n_y = fm.ceil((y_max - y_min) / (self.comp_params.dy_wl * wavelength)) + 1
        self.y_computational_grid, self.dy = np.linspace(y_min, y_max, self.n_y, retstep=True)

        abs_z_length = (self.env.z_max - self.env.z_min) * self.comp_params.abs_layer_scale / 2
        z_min = self.env.z_min - abs_z_length
        z_max = self.env.z_max + abs_z_length
        self.n_z = fm.ceil((z_max - z_min) / (self.comp_params.dz_wl * wavelength)) + 1
        self.z_computational_grid, self.dz = np.linspace(z_min, z_max, self.n_z, retstep=True)

        if self.comp_params.abs_layer_scale > 0:
            abs_mult_y = np.full((self.n_y, self.n_z), 1.0, dtype=float)
            left_grid = self.y_computational_grid[self.y_computational_grid < self.env.y_min]
            left_matrix_grid = np.tile(left_grid, (self.n_z, 1)).transpose()
            abs_mult_y[self.y_computational_grid < self.env.y_min, :] = np.cos(
                fm.pi / 2 * (left_matrix_grid - left_grid[-1]) / (left_grid[0] - left_grid[-1]))

            right_grid = self.y_computational_grid[self.y_computational_grid > self.env.y_max]
            right_matrix_grid = np.tile(right_grid, (self.n_z, 1)).transpose()
            abs_mult_y[self.y_computational_grid > self.env.y_max, :] = np.cos(
                fm.pi / 2 * (right_matrix_grid - right_grid[0]) / (right_grid[-1] - right_grid[0]))

            abs_mult_z = np.full((self.n_y, self.n_z), 1.0, dtype=float)
            left_grid = self.z_computational_grid[self.z_computational_grid < self.env.z_min]
            left_matrix_grid = np.tile(left_grid, (self.n_y, 1))
            abs_mult_z[:, self.z_computational_grid < self.env.z_min] = np.cos(
                fm.pi / 2 * (left_matrix_grid - left_grid[-1]) / (left_grid[0] - left_grid[-1]))

            right_grid = self.z_computational_grid[self.z_computational_grid > self.env.z_max]
            right_matrix_grid = np.tile(right_grid, (self.n_y, 1))
            abs_mult_z[:, self.z_computational_grid > self.env.z_max] = np.cos(
                fm.pi / 2 * (right_matrix_grid - right_grid[0]) / (right_grid[-1] - right_grid[0]))

            self.abs_mult = abs_mult_y * abs_mult_z
        else:
            self.abs_mult = np.ones((self.n_y, self.n_z))

        self.pade_coefs = pade_propagator_coefs(pade_order=self.comp_params.pade_order, diff2=lambda s: s,
                                                k0=self.k0, dx=self.dx, spe=self.comp_params.spe)

    def _ADI_2d_propagate(self, a, b, het, phi, y_left_bound, y_right_bound, z_left_bound, z_right_bound, iter_num=1):
        dz_2 = 1 / (self.k0 * self.dz) ** 2 * \
               sparse.diags([np.ones(self.n_z - 1), -2 * np.ones(self.n_z), np.ones(self.n_z - 1)], [-1, 0, 1])
        right_z_matrix = sparse.eye(self.n_z) + a * dz_2
        dy_2 = 1 / (self.k0 * self.dy) ** 2 * \
               sparse.diags([np.ones(self.n_y - 1), -2 * np.ones(self.n_y), np.ones(self.n_y - 1)], [-1, 0, 1])
        right_y_matrix = sparse.eye(self.n_y) + a * dy_2

        res2 = np.zeros(phi.shape)*0j
        phi_k = (a/b)**2 * phi
        for i in range(0, iter_num):
            res = (right_z_matrix * phi.T).T + dy_2 * (dz_2 * (b**2 * phi_k - a**2 * phi).T).T / 2
            for i_z in range(0, res.shape[1]):
                res2[:, i_z] = np.array(Crank_Nikolson_propagator2(self.k0*self.dy, b, np.zeros(self.n_y, dtype=complex), res[:, i_z],
                                                                  (1, 0, 0), (0, 1, 0)))

            res2 = right_y_matrix * res2 + dy_2 * (dz_2 * (b**2 * phi_k - a**2 * phi).T).T / 2
            for i_y in range(0, res.shape[0]):
                res[i_y, :] = np.array(Crank_Nikolson_propagator2(self.k0*self.dz, b, np.zeros(self.n_z, dtype=complex), res2[i_y, :],
                                                                  (1, 0, 0), (0, 1, 0)))
            phi_k = res

        return phi_k

    def _ADI_2d_propagate_4th_order(self, a, b, het, phi, y_left_bound, y_right_bound, z_left_bound, z_right_bound, iter_num=1):
        delta_z_2 = 1 / self.dz ** 2 * \
               sparse.diags([np.ones(self.n_z - 1), -2 * np.ones(self.n_z), np.ones(self.n_z - 1)], [-1, 0, 1])
        right_z_matrix = self.k0**2 * sparse.eye(self.n_z) + (self.k0**2 * self.dz**2 / 12 + a) * delta_z_2
        delta_y_2 = 1 / self.dy ** 2 * \
                    sparse.diags([np.ones(self.n_y - 1), -2 * np.ones(self.n_y), np.ones(self.n_y - 1)], [-1, 0, 1])
        right_y_matrix = self.k0 ** 2 * sparse.eye(self.n_y) + (self.k0 ** 2 * self.dy**2 / 12 + a) * delta_y_2

        res2 = np.zeros(phi.shape) * 0j
        phi_k = phi*0
        for i in range(0, iter_num):
            rem = delta_y_2 * (delta_z_2 * (b**2 * phi_k - a**2 * phi).T).T
            logging.debug('rem=' + str(np.linalg.norm(rem)))
            res = right_y_matrix*(right_z_matrix * phi.T).T + rem
            logging.debug('res=' + str(np.linalg.norm(res)))
            for i_z in range(0, res.shape[1]):
                res2[:, i_z] = np.array(Crank_Nikolson_propagator2_4th_order(self.k0, self.dy, b, np.zeros(self.n_y, dtype=complex), res[:, i_z],
                                                                  (1, 0, 0), (0, 1, 0)))

            #res2 = right_y_matrix * res2 + delta_y_2 * (delta_z_2 * (b**2 * phi_k - a**2 * phi).T).T / 2
            for i_y in range(0, res.shape[0]):
                res[i_y, :] = np.array(Crank_Nikolson_propagator2_4th_order(self.k0, self.dz, b, np.zeros(self.n_z, dtype=complex), res2[i_y, :],
                                                                  (1, 0, 0), (0, 1, 0)))
            phi_k = res

        return phi_k

    def sylvester_propagate(self, a, b, het, phi, y_left_bound, y_right_bound, z_left_bound, z_right_bound, iter_num=1):
        matrix_z = 1 / (self.k0*self.dz)**2 * (
                np.diag(np.ones(self.n_z - 1), -1) - 2*np.diag(np.ones(self.n_z)) + np.diag(np.ones(self.n_z - 1), 1))
        matrix_y = 1 / (self.k0 * self.dy) ** 2 * (
                    np.diag(np.ones(self.n_y - 1), -1) - 2 * np.diag(np.ones(self.n_y)) + np.diag(np.ones(self.n_y - 1), 1))

        matrix_c = a * phi @ matrix_z + a * matrix_y @ phi + phi

        res = la.solve_sylvester(b * matrix_y, b * matrix_z + np.eye(self.n_z), matrix_c)

        return res

    def _intersection_mask(self, x_i):
        mask = np.full((self.n_y, self.n_z), False, dtype=bool)
        for kn in self.env.knife_edges:
            if kn.x1 != kn.x2:
                logging.warning("knife-edge geometry is not supported")
                continue
            if divmod(kn.x1, self.dx)[0] == x_i:
                mg = np.meshgrid(np.logical_and(self.y_computational_grid >= kn.y1, self.y_computational_grid <= kn.y2),
                                 np.logical_and(self.z_computational_grid >= 0, self.z_computational_grid <= kn.height))
                mask[mg[0].T * mg[1].T == 1] = True

        return mask

    def propagate(self, src, polarz, x_max):
        n_x = fm.ceil(x_max / self.dx)
        x_computational_grid = np.arange(0, n_x) * self.dx

        y_res_mask = np.full(self.y_computational_grid.size, False, dtype=bool)
        y_res_mask[::self.comp_params.n_dy_out] = True
        y_res_mask = np.logical_and(np.logical_and(self.y_computational_grid >= self.env.y_min,
                                    self.y_computational_grid <= self.env.y_max), y_res_mask)

        z_res_mask = np.full(self.z_computational_grid.size, False, dtype=bool)
        z_res_mask[::self.comp_params.n_dz_out] = True
        z_res_mask = np.logical_and(np.logical_and(self.z_computational_grid >= self.env.z_min,
                                    self.z_computational_grid <= self.env.z_max), z_res_mask)

        field = Field3d(x_computational_grid[::self.comp_params.n_dx_out],
                        self.y_computational_grid[y_res_mask],
                        self.z_computational_grid[z_res_mask])

        phi = src
        field.field[0, :] = (phi[y_res_mask, :])[:, z_res_mask]

        for x_i, x in enumerate(x_computational_grid[1:], start=1):
            for pc_i, (a, b) in enumerate(self.pade_coefs):
                mask = self._intersection_mask(x_i - 1)
                phi[mask] = 0.0
                if self.comp_params.yz_order == 4:
                    phi = self.sylvester_propagate(a, b, None, phi, iter_num=self.comp_params.adi_iters,
                                                            y_left_bound=None, y_right_bound=None,
                                                            z_left_bound=None, z_right_bound=None)
                else:
                    phi = self._ADI_2d_propagate(a, b, None, phi, iter_num=self.comp_params.adi_iters,
                                                           y_left_bound=None, y_right_bound=None,
                                                           z_left_bound=None, z_right_bound=None)
            phi *= self.abs_mult
            if divmod(x_i, self.comp_params.n_dx_out)[1] == 0:
                field.field[divmod(x_i, self.comp_params.n_dx_out)[0], :] = (phi[y_res_mask, :])[:, z_res_mask]
                logging.debug('Pade 3D propagation x = ' + str(x))

        return field


class Pade3DPropagationTask:

    def __init__(self, *, src, env: EMEnvironment3d, comp_params):
        self.src = src
        self.env = env
        self.comp_params = comp_params
        self.propagator_fw = Pade3dPropagator(env=env, freq_hz=src.freq_hz, comp_params=comp_params)

    def calculate(self):
        return self.propagator_fw.propagate(
            self.src.aperture(self.propagator_fw.y_computational_grid, self.propagator_fw.z_computational_grid),
                                     polarz=self.src.polarz, x_max=self.env.x_max)