from rwp.antennas import *
from rwp.environment import *
from copy import deepcopy
from propagators._utils import *


class TwoRayModel:

    def __init__(self, src: Source, env: Troposphere):
        self.src = deepcopy(src)
        self.env = deepcopy(env)
        self.k0 = 2*cm.pi / self.src.wavelength

        if isinstance(self.src, GaussAntenna):
            def fur_q_func(z_spectral_point):
                z_spectral_point += self.src.k0 * cm.sin(self.src.elevation_angle * cm.pi / 180)
                ww = cm.exp(-z_spectral_point**2 * cm.log(2) / (2 * self.k0**2 * cm.sin(self.src.beam_width * fm.pi / 180 / 2)**2))
                return ww
        else:
            def fur_q_func(z_spectral_points):
                return 1 / cm.sqrt(2*cm.pi)

        self.fur_q_func = fur_q_func

    def _reflection_coefficient(self, theta):
        if isinstance(self.env.terrain.ground_material(0), PerfectlyElectricConducting):
            if self.src.polarz == 'H':
                return -1
            else:
                return 1
        else:
            return reflection_coef(1, self.env.terrain.ground_material(0).complex_permittivity(self.src.freq_hz), 90 - theta, self.src.polarz)

    def _val(self, x_m, z_m):
        z0 = self.src.height_m
        r_los = fm.sqrt(x_m**2 + (z_m - z0)**2)
        r_refl = fm.sqrt(x_m**2 + (z_m + z0)**2)
        theta_los = fm.atan2(z_m - z0, x_m) * 180 / fm.pi
        k_z_los = self.k0 * fm.sin(theta_los * fm.pi / 180)
        theta_r = fm.atan2(z_m + z0, x_m) * 180 / fm.pi
        k_z_r = self.k0 * fm.sin(theta_r * fm.pi / 180)
        return self.fur_q_func(k_z_los) * cm.exp(1j*self.k0*r_los) / r_los + \
               self.fur_q_func(-k_z_r) * self._reflection_coefficient(theta_r) * cm.exp(1j*self.k0*r_refl) / r_refl

    def calculate(self, x_grid_m, z_grid_m):
        res = np.empty((len(x_grid_m), len(z_grid_m)), dtype=complex)
        for x_i, x in enumerate(x_grid_m):
            for z_i, z in enumerate(z_grid_m):
                res[x_i, z_i] = self._val(x, z)
        return res
