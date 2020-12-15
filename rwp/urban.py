from rwp.environment import *
import math as fm
from propagators._utils import *
import scipy.linalg as la
from rwp.antennas import *
import logging
from scipy import fft

from rwp.field import Field3d


class Facets:

    def __init__(self, facets):
        self.x_indexes = []
        self.masks = []
        self.fields = []
        for facet in facets:
            self.x_indexes += [facet[0]]
            self.masks += [facet[1]]
            self.fields += [np.zeros(facet[1].shape, dtype=complex)]

    def mask(self, x_index):
        internal_index = self.x_indexes.index(x_index)
        return self.masks[internal_index]

    def field(self, x_index):
        internal_index = self.x_indexes.index(x_index)
        return self.fields[internal_index]

    def add_facet(self, x_index, mask, field):
        self.x_indexes += [x_index]
        self.masks += [mask]
        self.fields += [field]



class FDUrbanPropagatorComputationParameters:

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


class FDUrbanPropagator:

    def __init__(self, env: Manhattan3D, freq_hz, comp_params: FDUrbanPropagatorComputationParameters):
        self.env = env
        self.freq_hz = freq_hz
        self.comp_params = comp_params
        self.k0 = 2*fm.pi*self.freq_hz / LIGHT_SPEED
        wavelength = LIGHT_SPEED / self.freq_hz
        self.dx = wavelength * self.comp_params.dx_wl

        # prepare an absorption layer
        abs_y_length = self.env.domain_width * self.comp_params.abs_layer_scale / 2
        y_min = -self.env.domain_width / 2 - abs_y_length
        y_max = self.env.domain_width / 2 + abs_y_length
        self.n_y = fm.ceil((y_max - y_min) / (self.comp_params.dy_wl * wavelength)) + 1
        self.y_computational_grid, self.dy = np.linspace(y_min, y_max, self.n_y, retstep=True)

        abs_z_length = self.env.domain_height * self.comp_params.abs_layer_scale / 2
        z_min = 0
        z_max = self.env.domain_height + abs_z_length
        self.n_z = fm.ceil((z_max - z_min) / (self.comp_params.dz_wl * wavelength)) + 1
        self.z_computational_grid, self.dz = np.linspace(z_min, z_max, self.n_z, retstep=True)

        if self.comp_params.abs_layer_scale > 0:
            abs_mult_y = np.full((self.n_y, self.n_z), 1.0, dtype=float)
            left_grid = self.y_computational_grid[self.y_computational_grid < -self.env.domain_width/2]
            left_matrix_grid = np.tile(left_grid, (self.n_z, 1)).transpose()
            abs_mult_y[self.y_computational_grid < -self.env.domain_width/2, :] = np.cos(
                fm.pi / 2 * (left_matrix_grid - left_grid[-1]) / (left_grid[0] - left_grid[-1]))

            right_grid = self.y_computational_grid[self.y_computational_grid > self.env.domain_width/2]
            right_matrix_grid = np.tile(right_grid, (self.n_z, 1)).transpose()
            abs_mult_y[self.y_computational_grid > self.env.domain_width/2, :] = np.cos(
                fm.pi / 2 * (right_matrix_grid - right_grid[0]) / (right_grid[-1] - right_grid[0]))

            abs_mult_z = np.full((self.n_y, self.n_z), 1.0, dtype=float)
            right_grid = self.z_computational_grid[self.z_computational_grid > self.env.domain_height]
            right_matrix_grid = np.tile(right_grid, (self.n_y, 1))
            abs_mult_z[:, self.z_computational_grid > self.env.domain_height] = np.cos(
                fm.pi / 2 * (right_matrix_grid - right_grid[0]) / (right_grid[-1] - right_grid[0]))

            self.abs_mult = abs_mult_y * abs_mult_z
        else:
            self.abs_mult = np.ones((self.n_y, self.n_z))

        self.pade_coefs = pade_propagator_coefs(pade_order=self.comp_params.pade_order, diff2=lambda s: s,
                                                k0=self.k0, dx=self.dx, spe=self.comp_params.spe)

        self.matrix_z = 1 / (self.k0 * self.dz) ** 2 * (
                np.diag(np.ones(self.n_z - 1), -1) - 2 * np.diag(np.ones(self.n_z)) + np.diag(np.ones(self.n_z - 1), 1))
        self.matrix_y = 1 / (self.k0 * self.dy) ** 2 * (
                np.diag(np.ones(self.n_y - 1), -1) - 2 * np.diag(np.ones(self.n_y)) + np.diag(np.ones(self.n_y - 1), 1))

        self.k_y_grid = 2 * fm.pi * (np.arange(0, self.n_y) - self.n_y / 2) / (self.n_y * self.dy)
        self.dk_y = self.k_y_grid[1] - self.k_y_grid[0]
        self.k_z_grid = 2 * fm.pi * (np.arange(0, self.n_z) - self.n_z / 2) / (self.n_z * self.dz)
        self.dk_z = self.k_z_grid[1] - self.k_z_grid[0]

        self.k_y_matrix_grid = np.tile(self.k_y_grid, (self.n_z, 1)).transpose()
        self.k_z_matrix_grid = np.tile(self.k_z_grid, (self.n_y, 1))

    def sylvester_propagate(self, a, b, het, phi, y_left_bound, y_right_bound, z_left_bound, z_right_bound):
        matrix_c = a * phi @ self.matrix_z + a * self.matrix_y @ phi + phi
        res = la.solve_sylvester(b * self.matrix_y, b * self.matrix_z + np.eye(self.n_z), matrix_c)
        return res

    def _transform(self, phi):
        y_indexes = np.tile(np.arange(0, self.n_y), (self.n_z, 1)).transpose()
        z_indexes = np.tile(np.arange(0, self.n_z), (self.n_y, 1))
        x = phi * np.power(-1, y_indexes + z_indexes, dtype=float)
        fx = fft.fftn(x, axes=1, overwrite_x=True, workers=4)
        ffx = fft.fftn(fx, axes=0, overwrite_x=True, workers=4)
        return self.dy * self.dz / (2 * fm.pi) * \
               np.exp(-1j * (2 * cm.pi * (z_indexes - self.n_z / 2) / (self.n_z * self.dz)) * self.z_computational_grid[
                   0]) * \
               np.exp(-1j * (2 * cm.pi * (y_indexes - self.n_y / 2) / (self.n_y * self.dy)) * self.y_computational_grid[
                   0]) * ffx

    def _inv_transform(self, phi):
        y_indexes = np.tile(np.arange(0, self.n_y), (self.n_z, 1)).transpose()
        z_indexes = np.tile(np.arange(0, self.n_z), (self.n_y, 1))
        return self.dk_y * self.dk_z / (2 * fm.pi) * \
               np.exp(1j * self.k_z_grid[0] * (
                       self.z_computational_grid[0] + 2 * cm.pi * z_indexes / (self.n_z * self.dk_z))) * \
               np.exp(1j * self.k_y_grid[0] * (
                           self.y_computational_grid[0] + 2 * cm.pi * y_indexes / (self.n_y * self.dk_y))) * \
               fft.ifft2(phi * np.exp(1j * y_indexes * self.dk_y * self.y_computational_grid[0] +
                                         1j * z_indexes * self.dk_z * self.z_computational_grid[0]), overwrite_x=True, workers=4) * self.n_y * self.n_z

    def ssf_propagate(self, phi):
        return self.abs_mult * self._inv_transform(
            np.exp(1j * np.sqrt(self.k0 ** 2 - self.k_y_matrix_grid ** 2 - self.k_z_matrix_grid ** 2, dtype=complex) * self.dx) *
            self._transform(phi))

    def _intersection_mask(self, x_i):
        # mask = np.full((self.n_y, self.n_z), False, dtype=bool)
        # mg = np.meshgrid(np.logical_or(self.y_computational_grid <= -self.env.street_width / 2,
        #                                 self.y_computational_grid >= self.env.street_width / 2),
        #                  self.z_computational_grid <= self.env.building_height)
        # mask[mg[0].T * mg[1].T == 1] = True
        #
        # return mask
        return self.env.intersection_mask_x(x_i*self.dx, self.y_computational_grid, self.z_computational_grid)

    def _propagate(self, x_computational_grid, fwd_facets: Facets, bwd_facets: Facets, fwd=True):
        y_res_mask = np.full(self.y_computational_grid.size, False, dtype=bool)
        y_res_mask[::self.comp_params.n_dy_out] = True
        y_res_mask = np.logical_and(np.logical_and(self.y_computational_grid >= -self.env.domain_width / 2,
                                    self.y_computational_grid <= self.env.domain_width / 2), y_res_mask)

        z_res_mask = np.full(self.z_computational_grid.size, False, dtype=bool)
        z_res_mask[::self.comp_params.n_dz_out] = True
        z_res_mask = np.logical_and(np.logical_and(self.z_computational_grid >= 0,
                                    self.z_computational_grid <= self.env.domain_height), z_res_mask)

        field = Field3d(x_computational_grid[::self.comp_params.n_dx_out],
                        self.y_computational_grid[y_res_mask],
                        self.z_computational_grid[z_res_mask])

        if fwd:
            iterator = enumerate(x_computational_grid[1:], start=1)
        else:
            iterator = enumerate(x_computational_grid[-2::-1], start=1)

        phi = np.zeros((len(self.y_computational_grid), len(self.z_computational_grid)), dtype=complex)
        if fwd:
            if 0 in fwd_facets.x_indexes:
                phi_reshaped = phi.reshape(phi.shape[0]*phi.shape[1])
                phi_reshaped[fwd_facets.mask(0).reshape(phi.shape[0]*phi.shape[1])] = fwd_facets.field(0).reshape(phi.shape[0]*phi.shape[1])
                phi = phi_reshaped.reshape(phi.shape)

                mask = self._intersection_mask(0)
                phi[mask] = 0.0

        field.field[0, :] = (phi[y_res_mask, :])[:, z_res_mask]

        for x_i, x in iterator:
            phi = self.ssf_propagate(phi)
            if fwd:
                mask = self._intersection_mask(x_i - 1)
            else:
                mask = self._intersection_mask(0)#what??

            phi[mask] = 0.0
            phi *= self.abs_mult
            if x_i in fwd_facets.x_indexes:
                phi_reshaped = phi.reshape(phi.shape[0] * phi.shape[1])
                mask = fwd_facets.mask(x_i)
                f = fwd_facets.field(x_i)
                phi_reshaped[mask.reshape(mask.shape[0] * mask.shape[1])] = f.reshape(
                    f.shape[0] * f.shape[1])[mask.reshape(mask.shape[0] * mask.shape[1])]
                phi = phi_reshaped.reshape(phi.shape)
            if x_i in bwd_facets.x_indexes:
                bwd_facets.field(x_i)[:] = -phi
            # for pc_i, (a, b) in enumerate(self.pade_coefs):
            #     mask = self._intersection_mask(x_i - 1)
            #     phi[mask] = 0.0
            #     phi = self.sylvester_propagate(a, b, None, phi,
            #                                    y_left_bound=None, y_right_bound=None,
            #                                    z_left_bound=None, z_right_bound=None)
            if divmod(x_i, self.comp_params.n_dx_out)[1] == 0:
                field.field[divmod(x_i, self.comp_params.n_dx_out)[0], :] = (phi[y_res_mask, :])[:, z_res_mask]
                logging.debug('Pade 3D propagation x = ' + str(x))

        return field

    def calculate(self, src, two_way=False):
        n_x = fm.ceil(self.env.x_max / self.dx)
        x_computational_grid = np.arange(0, n_x) * self.dx

        fwd_facets = Facets(self.env.facets(
            x_computational_grid, self.y_computational_grid, self.z_computational_grid, forward=True))
        fwd_facets.add_facet(0, np.full((len(self.y_computational_grid), len(self.z_computational_grid)), True, dtype=bool),
                             src.aperture(self.y_computational_grid, self.z_computational_grid))
        bwd_facets = Facets(self.env.facets(
            x_computational_grid, self.y_computational_grid, self.z_computational_grid, forward=False))

        fwd_field = self._propagate(x_computational_grid, fwd_facets, bwd_facets, fwd=True)
        fwd_facets.x_indexes = [n_x - a - 1 for a in fwd_facets.x_indexes]
        bwd_facets.x_indexes = [n_x - a - 1 for a in bwd_facets.x_indexes]
        #bwd_field = self._propagate(x_computational_grid, bwd_facets, fwd_facets, fwd=False)
        #fwd_field.field += bwd_field.field[::-1, :]
        return fwd_field