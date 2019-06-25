from uwa.source import *
from uwa.environment import *
from propagators.sspade import *


class UnderwaterAcousticsSSPadePropagator:

    def __init__(self, src: Source, env: UnderwaterEnvironment, max_range_m, comp_params=HelmholtzPropagatorComputationalParams()):
        self.uwa_env = env
        self.comp_params = comp_params

        # prepare Helmholtz environment
        self.helmholtz_env = HelmholtzEnvironment(x_max_m=max_range_m, lower_bc=TransparentConstBC(), upper_bc=RobinBC(q1=1, q2=0, q3=0))
        self.helmholtz_env.z_max = 0
        self.helmholtz_env.z_min = min([self.uwa_env.bottom_profile(x) for x in range(0, max_range_m, 1)]) - 1
        c0 = min([self.uwa_env.sound_speed_profile_m_s(0, z) for z in range(self.helmholtz_env.z_min, self.helmholtz_env.z_max, 1)]) - 1
        self.helmholtz_env.n2minus1 = lambda x, z: (c0 / self.uwa_env.sound_speed_profile_m_s(x, z))**2 - 1
        self.helmholtz_env.rho = lambda x, z: self.uwa_env.density_profile_g_cm(x, z) #???
        self.helmholtz_env.terrain = lambda x: self.uwa_env.bottom_profile

        wavelength = c0 / src.freq_hz
        self.propagator = HelmholtzPadeSolver(env=self.helmholtz_env, wavelength=wavelength, freq_hz=src.freq_hz, params=comp_params)