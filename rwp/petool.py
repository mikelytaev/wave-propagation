import logging
import math as fm

import matlab.engine

from rwp.WPDefs import *
from rwp.environment import *

__author__ = 'Lytaev Mikhail (mikelytaev@gmail.com)'


class PETOOLPropagator:
    """
    Python wrapper for PETOOL.
    Ozgun, O., Apaydin, G., Kuzuoglu, M., Sevgi, L. (2011). PETOOL: MATLAB-based one-way and two-way
     split-step parabolic equation tool for radiowave propagation over variable terrain.
     Computer Physics Communications, 182(12), 2638-2654.
    """

    def __init__(self, env: EMEnvironment, wavelength=1.0, dx_wl=1000, dz_wl=1):
        self.env = env
        self.k0 = (2 * cm.pi) / wavelength
        self.wavelength = wavelength
        self.dx = dx_wl * wavelength
        self.dz = dz_wl * wavelength
        self.z_computational_grid = np.linspace(0, self.env.z_max, fm.ceil(self.env.z_max / self.dz) + 1)

    def propagate(self, src: Source, n_x, *, n_dx_out=1, n_dz_out=1, two_way=False):
        edge_range = matlab.double([a * 1e-3 for a in self.env.terrain.edge_range], is_complex=True)
        edge_height = matlab.double(list(self.env.terrain.edge_height), is_complex=True)
        terrain_type = 1
        interp_type = 1
        if isinstance(self.env.terrain, KnifeEdges):
            terrain_type = 2
            interp_type = 1
        elif isinstance(self.env.terrain, LinearTerrain):
            terrain_type = 2
            interp_type = 2

        polarz_n = 1 if src.polarz.upper == 'H' else 2
        backward_n = 2 if two_way and terrain_type == 2 else 1
        het = matlab.double([a * 1e6 / 4 for a in self.env.n2m1_profile(0, self.z_computational_grid)], is_complex=True)
        if isinstance(self.env.lower_boundary, EarthSurfaceBC):
            ground_type = 2
            self.env.lower_boundary(self.wavelength, src.polarz)
            epsilon = self.env.lower_boundary.permittivity
            sigma = self.env.lower_boundary.conductivity
        else:
            ground_type = 1
            epsilon = 0
            sigma = 0

        logging.debug('Starting Matlab engine...')
        eng = matlab.engine.start_matlab()
        logging.debug('PETOOL propagating...')
        path_loss, prop_fact, free_space_loss, range_vec, z_user, z, stopflag = \
            eng.SSPE_function(float(3 * 1e8 / self.wavelength * 1e-6),  # freq, MHz
                          float(src.beam_width),  # thetabw, degrees
                          float(src.eval_angle),  # thetae, degrees
                          polarz_n,  # polrz (1=hor, 2=ver)
                          float(src.height),  # tx_height, m
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
                          float(epsilon),
                              float(sigma),
                              float(self.dx),
                              float(self.dz),
                              nargout=7)
        eng.quit()

        x_computational_grid = np.array(range_vec._data)
        field = Field(x_computational_grid[1::n_dx_out], self.z_computational_grid[::n_dz_out])
        field.field[:, :] = np.array(prop_fact._data).reshape(prop_fact.size[::-1]).T[::n_dz_out, 1::n_dx_out].T
        field.field -= np.tile(10 * np.log10(x_computational_grid[1::n_dx_out]), (self.z_computational_grid.shape[0], 1)).T
        field.field -= 10*fm.log10(self.wavelength)
        field.field /= 2

        return field


class PETOOLPropagationTask:

    def __init__(self, *, src: Source, env: EMEnvironment, two_way=False, max_range_m=100000, dx_wl=100, dz_wl=1,
                 n_dx_out=1, n_dz_out=1):
        self.src = src
        self.env = env
        self.two_way = two_way
        self.max_range_m = max_range_m
        self.n_dx_out = n_dx_out
        self.n_dz_out = n_dz_out
        self.propagator = PETOOLPropagator(env=self.env, wavelength=src.wavelength, dx_wl=dx_wl, dz_wl=dz_wl)

    def calculate(self):
        n_x = fm.ceil(self.max_range_m / self.propagator.dx) + 1
        field = self.propagator.propagate(src=self.src, n_x=n_x, n_dx_out=self.n_dx_out, n_dz_out=self.n_dz_out, two_way=self.two_way)
        return field
