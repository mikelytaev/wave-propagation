import logging
import math as fm
import pickle

import pyximport

import time

from rwp.antennas import *
from rwp.environment import *
from propagators._utils import *
from rwp.field import Field
from propagators.sspade import LocalBC, TerrainMethod, HelmholtzPadeSolver

pyximport.install(setup_args={"include_dirs": np.get_include()})
from propagators._cn_utils import *


class TroposphericRadioWaveSSPadePropagator:

    def __init__(self, *, antenna: Source, env: Troposphere, two_way=False, iter_num=None, max_range_m, max_angle=None,
                 dx_wl=None, dz_wl=None, n_dx_out=None, n_dz_out=None, pade_order=None, z_order=4, spe=False,
                 terrain_method=None, nlbc_manager_path='nlbc', tol=1e-11):
        self.src = antenna
        self.env = env
        self.two_way = two_way
        if len(self.env.knife_edges) == 0:
            self.two_way = False
        self.iter_num = iter_num or 1
        self.max_range_m = max_range_m
        self.nlbc_manager_path = nlbc_manager_path

        max_angle = max_angle or self._optimal_angle()
        if dx_wl:
            dx_wl = [dx_wl]
        if dz_wl:
            dz_wl = [dz_wl]
        if pade_order:
            pade_order = [pade_order]

        logging.info("ground refractive index: " + str(self.env.ground_material.complex_permittivity(antenna.freq_hz)))
        if terrain_method is None:

            if abs(self.env.ground_material.complex_permittivity(antenna.freq_hz)) < 100 and max_angle > 10:
                self.terrain_method = TerrainMethod.pass_through
            else:
                self.terrain_method = TerrainMethod.staircase
        else:
            self.terrain_method = terrain_method

        logging.info("Terrain method: " + self.terrain_method.name)

        logging.info("max_angle = " + str(max_angle))
        logging.info("calculating optimal grid parameters...")

        if self.env.is_homogeneous() and self.terrain_method == TerrainMethod.staircase:
            z_order_p = float('inf')
            logging.info("using Pade approximation for diff2_z")
        else:
            z_order_p = z_order

        (opt_dx, opt_dz, opt_pade) = optimal_params(max_angle=max_angle, threshold=5e-3, dxs=dx_wl, dzs=dz_wl,
                                                    pade_orders=pade_order, z_order=z_order_p)

        x_approx_sampling = 2000
        z_approx_sampling = 1000
        wavelength = 3e8 / self.src.freq_hz

        opt_dx = min(opt_dx or dx_wl[0], max_range_m / wavelength / x_approx_sampling)
        opt_dz = min(opt_dz or dz_wl[0], env.z_max / wavelength / z_approx_sampling)
        opt_pade = opt_pade or pade_order[0]

        if self.terrain_method == TerrainMethod.pass_through:
            n_g = self.env.ground_material.complex_permittivity(antenna.freq_hz)
            opt_dx /= round(abs(cm.sqrt(n_g - 0.1)))
            opt_dz /= round(abs(cm.sqrt(n_g - 0.1)))

        self.n_x = fm.ceil(self.max_range_m / opt_dx / wavelength) + 1

        self.n_dx_out = n_dx_out or fm.ceil(max_range_m / antenna.wavelength / opt_dx / x_approx_sampling)
        self.n_dz_out = n_dz_out or fm.ceil(env.z_max / antenna.wavelength / opt_dz / z_approx_sampling)

        logging.info("dx = " + str(opt_dx))
        logging.info("dz = " + str(opt_dz))
        logging.info("Pade order = " + str(opt_pade))
        self.propagator = HelmholtzPadeSolver(env=self.env, n_x=self.n_x, wavelength=antenna.wavelength, z_order=z_order,
                                              pade_order=opt_pade, spe=spe, dx_wl=opt_dx, dz_wl=opt_dz,
                                              terrain_method=self.terrain_method, tol=tol)

    def _optimal_angle(self):
        if len(self.env.knife_edges) > 0:
            return 85
        else:
            res = 3
            step = 10
            for x in np.arange(step, self.max_range_m, step):
                angle = cm.atan((self.env.terrain(x) - self.env.terrain(x - step)) / step) * 180 / cm.pi
                res = max(res, abs(angle))
            res = max(self.src.max_angle(), fm.ceil(res))
            return res

    def _prepare_bc(self):
        upper_bc = NLBCManager(self.nlbc_manager_path).get_upper_nlbc(self.propagator, self.n_x)
        if self.terrain_method == TerrainMethod.pass_through:
            lower_bc = NLBCManager(self.nlbc_manager_path).get_lower_nlbc(self.propagator, self.n_x)
        else:
            if isinstance(self.env.ground_material, PerfectlyElectricConducting):
                if self.src.polarz == 'H':
                    q1, q2 = 1, 0
                else:
                    q1, q2 = 0, 1
            else:
                if self.src.polarz == 'H':
                    q1, q2 = 1j * self.propagator.k0 * self.env.ground_material.complex_permittivity(self.src.freq_hz) ** (1 / 2), 1
                else:
                    q1, q2 = 1j * self.propagator.k0 * self.env.ground_material.complex_permittivity(self.src.freq_hz) ** (-1 / 2), 1
            lower_bc = LocalBC(q1, q2)
        return lower_bc, upper_bc

    def calculate(self):
        start_time = time.time()
        lower_bc, upper_bc = self._prepare_bc()
        initials_fw = [np.empty(0)] * self.n_x
        initials_fw[0] = np.array([self.src.aperture(a) for a in self.propagator.z_computational_grid])
        reflected_bw = initials_fw
        x_computational_grid = np.arange(0, self.n_x) * self.propagator.dx
        field = Field(x_computational_grid[::self.n_dx_out], self.propagator.z_computational_grid[::self.n_dz_out],
                      freq_hz=self.propagator.freq_hz,
                      precision=self.propagator.tol)
        for i in range(0, self.iter_num):
            field_fw, reflected_fw = self.propagator.propagate(polarz=self.src.polarz, initials=reflected_bw,
                                                               direction=1, lower_bc=lower_bc, upper_bc=upper_bc,
                                                               n_dx_out=self.n_dx_out, n_dz_out=self.n_dz_out)

            field.field += field_fw.field
            #logging.debug(np.linalg.norm(field_fw.field))
            if not self.two_way:
                break

            field_bw, reflected_bw = self.propagator.propagate(polarz=self.src.polarz, initials=reflected_fw,
                                                               direction=-1, lower_bc=lower_bc, upper_bc=upper_bc,
                                                               n_dx_out=self.n_dx_out, n_dz_out=self.n_dz_out)
            field.field += field_bw.field
            #logging.debug(np.linalg.norm(field_bw.field))

        logging.debug("Elapsed time: " + str(time.time() - start_time))

        return field


class NLBCManager:

    def __init__(self, name='nlbc'):
        self.file_name = name
        import os
        if os.path.exists(self.file_name):
            with open(self.file_name, 'rb') as f:
                self.nlbc_dict = pickle.load(f)
        else:
            self.nlbc_dict = {}

    def get_lower_nlbc(self, propagator: HelmholtzPadeSolver, n_x):
        beta = propagator.env.ground_material.complex_permittivity(propagator.freq_hz) - 1
        gamma = 0
        q = 'lower', propagator.k0, propagator.dx, propagator.dz, propagator.pade_order, propagator.z_order, propagator.spe, beta, gamma
        if q not in self.nlbc_dict or self.nlbc_dict[q].coefs.shape[0] < n_x:
            self.nlbc_dict[q] = propagator.calc_lower_nlbc(beta)
        lower_nlbc = self.nlbc_dict[q]

        with open(self.file_name, 'wb') as f:
            pickle.dump(self.nlbc_dict, f)

        return lower_nlbc

    def get_upper_nlbc(self, propagator: HelmholtzPadeSolver, n_x):
        gamma = propagator.env.n2m1_profile(0, propagator.env.z_max+1, propagator.freq_hz) - propagator.env.n2m1_profile(0, propagator.env.z_max, propagator.freq_hz)
        beta = propagator.env.n2m1_profile(0, propagator.env.z_max, propagator.freq_hz) - gamma * propagator.env.z_max
        q = 'upper', propagator.k0, propagator.dx, propagator.dz, propagator.pade_order, propagator.z_order, propagator.spe, beta, gamma
        if q not in self.nlbc_dict or self.nlbc_dict[q].coefs.shape[0] < n_x:
            self.nlbc_dict[q] = propagator.calc_upper_nlbc(beta, gamma)
        upper_nlbc = self.nlbc_dict[q]

        with open(self.file_name, 'wb') as f:
            pickle.dump(self.nlbc_dict, f)

        return upper_nlbc


def d2a_n_eq_ba_n(b):
    c1 = (b+2-cm.sqrt(b**2+4*b))/2
    c2 = 1.0 / c1
    return [c1, c2][abs(c1) > abs(c2)]


def sqr_eq(a, b, c):
    c1 = (-b + cm.sqrt(b**2 - 4 * a * c)) / (2 * a)
    c2 = (-b - cm.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return [c1, c2][abs(c1) > abs(c2)]

# def bessel_ratio(c, d, j, tol):
#     return lentz(lambda n: (-1)**(n+1) * 2.0 * (j + (2.0 + d) / c + n - 1) * (c / 2.0))


def lentz(cont_frac_seq, tol=1e-20):
    """
    Lentz W. J. Generating Bessel functions in Mie scattering calculations using continued fractions
    //Applied Optics. – 1976. – 15. – №. 3. – P. 668-671.
    :param cont_frac_seq: continued fraction sequence
    :param tol: absolute tolerance
    """
    num = cont_frac_seq(2) + 1.0 / cont_frac_seq(1)
    den = cont_frac_seq(2)
    y = cont_frac_seq(1) * num / den
    i = 3
    while abs(num / den - 1) > tol:
        num = cont_frac_seq(i) + 1.0 / num
        den = cont_frac_seq(i) + 1.0 / den
        y = y * num / den
        i += 1

    return y