import logging
import math as fm

import matlab.engine

from rwp.antennas import *
from rwp.environment import *
from rwp.field import Field

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class PETOOLPropagator:
    """
    Python wrapper for PETOOL.
    Ozgun, O., Apaydin, G., Kuzuoglu, M., Sevgi, L. (2011). PETOOL: MATLAB-based one-way and two-way
     split-step parabolic equation tool for radiowave propagation over variable terrain.
     Computer Physics Communications, 182(12), 2638-2654.
    """

    def __init__(self, env: Troposphere, wavelength=1.0, dx_wl=1000, dz_wl=1):
        self.env = env
        self.k0 = (2 * cm.pi) / wavelength
        self.wavelength = wavelength
        self.freq_hz = LIGHT_SPEED / self.wavelength
        self.dx = dx_wl * wavelength
        self.dz = dz_wl * wavelength
        self.z_computational_grid = np.linspace(0, self.env.z_max, fm.ceil(self.env.z_max / self.dz) + 1)

    def propagate(self, src: GaussAntenna, n_x, *, n_dx_out=1, n_dz_out=1, two_way=False):
        x_computational_grid = np.arange(0, n_x) * self.dx
        terrain_type = 1
        interp_type = 1
        if len(self.env.knife_edges) > 0:
            terrain_type = 2
            interp_type = 1
            edge_range = matlab.double([ke.range * 1e-3 for ke in self.env.knife_edges], is_complex=True)
            edge_height = matlab.double([ke.height for ke in self.env.knife_edges], is_complex=True)
        else:
            terrain_type = 1 if self.env.terrain.is_homogeneous else 2
            interp_type = 2
            edge_range = matlab.double([a * 1e-3 for a in x_computational_grid], is_complex=True)
            edge_height = matlab.double([self.env.terrain.elevation(a)+0.0 for a in x_computational_grid], is_complex=True)

        polarz_n = 1 if src.polarz.upper() == 'H' else 2
        backward_n = 2 if two_way and terrain_type == 2 else 1
        het = matlab.double([a * 1e6 / 4 for a in self.env.n2m1_profile(0, self.z_computational_grid, self.freq_hz)],
                            is_complex=True)
        if isinstance(self.env.terrain.ground_material(0), PerfectlyElectricConducting):
            ground_type = 1
            epsilon = 0
            sigma = 0
        else:
            ground_type = 2
            epsilon = self.env.terrain.ground_material(0).permittivity(self.freq_hz)
            sigma = self.env.terrain.ground_material(0).conductivity_sm_m(self.freq_hz)

        logging.debug('Starting Matlab engine...')
        eng = matlab.engine.start_matlab()
        logging.debug('PETOOL propagating...')
        path_loss, prop_fact, free_space_loss, range_vec, z_user, z, stopflag = \
            eng.SSPE_function(float(3 * 1e8 / self.wavelength * 1e-6),  # freq, MHz
                          float(src.beam_width),  # thetabw, degrees
                          float(src.elevation_angle),  # thetae, degrees
                          polarz_n,  # polrz (1=hor, 2=ver)
                          float(src.height_m),  # tx_height, m
                          float(self.dx * (n_x - 1) * 1e-3),  # range, km
                          float(self.env.z_max),  # zmax_user, m
                          edge_range,  # edge_range, km
                          edge_height,  # edge_height, m
                          6,  # duct_type
                          het,  # duct_M
                          matlab.double([a for a in self.z_computational_grid], is_complex=True),  # duct_height
                          0.0,  # duct_range
                          terrain_type,  # terrain_type, 1=no terrain, 2=terrain
                          interp_type,  # interp_type, 1=none(knife-edges), 2=linear, 3=cubic spline
                          backward_n,  # backward, 1=one-way, 2=two-way
                          ground_type,  # ground_type, 1=PEC, 2=impedance ground surface
                          float(epsilon.real),
                              float(sigma.real),
                              float(self.dx),
                              float(self.dz),
                              nargout=7)
        eng.quit()

        x_computational_grid = np.array(range_vec._data)
        z_petool_grid = np.array(z_user._data)
        field = Field(x_computational_grid[1::n_dx_out], z_petool_grid[::n_dz_out], self.freq_hz)
        #field.field[:, :] = np.array(path_loss._data).reshape(path_loss.size[::-1]).T[::n_dz_out, 1::n_dx_out].T
        field.field[:, :] = np.array(prop_fact._data).reshape(prop_fact.size[::-1]).T[::n_dz_out, 1::n_dx_out].T
        field.field -= np.tile(10 * np.log10(x_computational_grid[1::n_dx_out]), (z_petool_grid[::n_dz_out].shape[0], 1)).T
        field.field -= 10*fm.log10(self.wavelength)
        field.field /= 2
        field.field = np.nan_to_num(field.field)

        return field


class PETOOLPropagationTask:

    def __init__(self, *, antenna: Source, env: Troposphere, two_way=False, max_range_m=100000, dx_wl=100, dz_wl=1,
                 n_dx_out=1, n_dz_out=1):
        self.src = antenna
        self.env = env
        self.two_way = two_way
        self.max_range_m = max_range_m
        self.n_dx_out = n_dx_out
        self.n_dz_out = n_dz_out
        self.propagator = PETOOLPropagator(env=self.env, wavelength=self.src.wavelength, dx_wl=dx_wl, dz_wl=dz_wl)

    def calculate(self):
        n_x = fm.ceil(self.max_range_m / self.propagator.dx) + 1
        field = self.propagator.propagate(src=self.src, n_x=n_x, n_dx_out=self.n_dx_out, n_dz_out=self.n_dz_out, two_way=self.two_way)
        return field
