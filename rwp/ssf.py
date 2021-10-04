import logging
from rwp.environment import *
from rwp.antennas import *
import math as fm

from rwp.field import Field3d, Field

from dataclasses import dataclass, field


@dataclass
class SSF2DComputationalParams:
    max_range_m: float = None
    max_height_m: float = None
    dx_wl: float = None
    dz_wl: float = 0.5
    abs_layer_scale: float = 0.5


class SSF2DPropagator:

    def __init__(self, *, antenna: Source, env: Troposphere, max_range_m: float, comp_params: SSF2DComputationalParams):
        self.antenna = antenna
        self.env = env
        self.max_range_m = max_range_m
        self.comp_params = comp_params

    def calculate(self):
        h_field = self.propagator.calculate(lambda z: self.src.aperture(z))
        res = Field(x_grid=h_field.x_grid_m, z_grid=h_field.z_grid_m, freq_hz=self.src.freq_hz)
        res.field = h_field.field
        return res


class SSF3DPropagatorComputationParameters:

    def __init__(self, dx_wl, dy_wl, dz_wl, n_dx_out=1, n_dy_out=1, n_dz_out=1, abs_layer_scale=0.5):
        self.dx_wl = dx_wl
        self.dy_wl = dy_wl
        self.dz_wl = dz_wl
        self.n_dx_out = n_dx_out
        self.n_dy_out = n_dy_out
        self.n_dz_out = n_dz_out
        self.abs_layer_scale = abs_layer_scale


class SSF3DPropagator:

    def __init__(self, env: EMEnvironment3d, freq_hz, comp_params: SSF3DPropagatorComputationParameters):
        self.env = env
        self.freq_hz = freq_hz
        self.comp_params = comp_params
        self.k0 = 2 * fm.pi * self.freq_hz / LIGHT_SPEED
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

        self.k_y_grid = 2 * fm.pi * (np.arange(0, self.n_y) - self.n_y / 2) / (self.n_y * self.dy)
        self.dk_y = self.k_y_grid[1] - self.k_y_grid[0]
        self.k_z_grid = 2 * fm.pi * (np.arange(0, self.n_z) - self.n_z / 2) / (self.n_z * self.dz)
        self.dk_z = self.k_z_grid[1] - self.k_z_grid[0]

    def _transform(self, phi):
        y_indexes = np.tile(np.arange(0, self.n_y), (self.n_z, 1)).transpose()
        z_indexes = np.tile(np.arange(0, self.n_z), (self.n_y, 1))
        return self.dy * self.dz / (2 * fm.pi) * \
               np.exp(-1j * (2 * cm.pi * (z_indexes - self.n_z / 2) / (self.n_z * self.dz)) * self.z_computational_grid[
                   0]) * \
               np.exp(-1j * (2 * cm.pi * (y_indexes - self.n_y / 2) / (self.n_y * self.dy)) * self.y_computational_grid[
                   0]) * \
               np.fft.fft2(phi * np.power(-1, y_indexes + z_indexes, dtype=float))

    def _inv_transform(self, phi):
        y_indexes = np.tile(np.arange(0, self.n_y), (self.n_z, 1)).transpose()
        z_indexes = np.tile(np.arange(0, self.n_z), (self.n_y, 1))
        return self.dk_y * self.dk_z / (2 * fm.pi) * \
               np.exp(1j * self.k_z_grid[0] * (
                       self.z_computational_grid[0] + 2 * cm.pi * z_indexes / (self.n_z * self.dk_z))) * \
               np.exp(1j * self.k_y_grid[0] * (
                           self.y_computational_grid[0] + 2 * cm.pi * y_indexes / (self.n_y * self.dk_y))) * \
               np.fft.ifft2(phi * np.exp(1j * y_indexes * self.dk_y * self.y_computational_grid[0] +
                                         1j * z_indexes * self.dk_z * self.z_computational_grid[0])) * self.n_y * self.n_z

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

        k_y_matrix_grid = np.tile(self.k_y_grid, (self.n_z, 1)).transpose()
        k_z_matrix_grid = np.tile(self.k_z_grid, (self.n_y, 1))
        for x_i, x in enumerate(x_computational_grid[1:], start=1):
            mask = self._intersection_mask(x_i - 1)
            phi[mask] = 0.0
            phi = self.abs_mult * self._inv_transform(
                np.exp(1j * np.sqrt(self.k0 ** 2 - k_y_matrix_grid ** 2 - k_z_matrix_grid ** 2, dtype=complex) * self.dx) *
                self._transform(phi))

            if divmod(x_i, self.comp_params.n_dx_out)[1] == 0:
                field.field[divmod(x_i, self.comp_params.n_dx_out)[0], :] = (phi[y_res_mask, :])[:, z_res_mask]
                logging.debug('SSFourier 3D propagation x = ' + str(x))
                #logging.debug(np.linalg.norm(phi))

        return field


class SSF3DPropagationTask:

    def __init__(self, *, src, env: EMEnvironment3d, comp_params):
        self.src = src
        self.env = env
        self.comp_params = comp_params
        self.propagator_fw = SSF3DPropagator(env=env, freq_hz=src.freq_hz, comp_params=comp_params)

    def calculate(self):
        return self.propagator_fw.propagate(
            self.src.aperture(self.propagator_fw.y_computational_grid, self.propagator_fw.z_computational_grid),
                                     polarz=self.src.polarz, x_max=self.env.x_max)