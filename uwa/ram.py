from uwa.field import *
from uwa.source import *
from uwa.environment import *
from propagators.sspade import *

import numpy as np
import matlab.engine

from dataclasses import dataclass


@dataclass
class RAMComputationalParams:
    output_ranges: np.ndarray
    dr: float
    dz: float
    zmax: float = None
    dz_decimate: int = 1
    c0: float = None
    pade_coefs_num: int = 4
    ns: int = 1 # # of stability terms
    rs: float = 10000.0 # stability range


class RAMMatlabPropagator:
    """
    Pyton wrapper on Matlab version of RAM http://staff.washington.edu/dushaw/AcousticsCode/RamMatlabCode.html
    """

    def __init__(self, src: Source, env: UnderwaterEnvironment, comp_params: RAMComputationalParams):
        self.src = src
        self.env = env
        self.comp_params = comp_params

    def calculate(self) -> AcousticPressureField:

        z_grid_m = np.arange(0, self.env.bottom_profile.max_depth, self.comp_params.dz)

        dim = 2
        frq = float(self.src.freq_hz)
        zsrc = float(self.src.depth_m)
        rg = matlab.double(self.comp_params.output_ranges.tolist())
        dr = float(self.comp_params.dr)
        if self.comp_params.zmax is None:
            zmax = float(self.env.bottom_profile.max_depth)+1000
        else:
            zmax = self.comp_params.zmax
        dz = float(self.comp_params.dz)
        dzm = self.comp_params.dz_decimate
        np_ = float(self.comp_params.pade_coefs_num)
        ns = float(self.comp_params.ns)
        rs = float(self.comp_params.rs)
        rb = matlab.double(self.env.bottom_profile.ranges()) # bathymetry range
        zb = matlab.double(self.env.bottom_profile.depths()) # bathymetry
        rp = matlab.double([0.0])
        zw = matlab.double(z_grid_m.tolist()) # sound speed grid depth(nzw)
        cw = matlab.double(self.env.sound_speed_profile_m_s(0, z_grid_m).tolist()) # sound speed(nr,nzw)
        zs = zmax# sediment speed grid depth(nzs)
        cs = float(self.env.bottom_sound_speed_m_s) # sediment speed(nr, nzs)
        zr = zmax# density depth grid(nzr)
        rho = float(self.env.bottom_density_g_cm) # density(nr,nzr)
        za = zmax # attenuation depth grid(nza)
        attn = float(self.env.bottom_attenuation_dm_lambda) # attenuation(nr,nza)

        if self.comp_params.c0 is None:
            self.comp_params.c0 = np.mean(cw)

        c0 = float(self.comp_params.c0)

        logging.debug('Starting Matlab engine...')
        eng = matlab.engine.start_matlab()
        logging.debug('PETOOL propagating...')
        fld, zg, rout = eng.ram(frq, zsrc, dim, rg, dr, zmax, dz, dzm, c0, np_, ns, rs, rb, zb, rp, zw, cw, zs, cs, zr, rho, za, attn, nargout=3)
        eng.quit()

        x_grid_out = np.array(rout._data)
        z_grid_out = np.array(zg._data)
        f = np.array(fld._data).reshape(fld.size[::-1])
        field = AcousticPressureField(x_grid=x_grid_out, z_grid=z_grid_out, freq_hz=frq, field=f)
        return field
