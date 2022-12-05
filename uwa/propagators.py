from uwa.field import *
from uwa.source import *
from uwa.environment import *
from propagators.sspade import *
from copy import deepcopy


class UnderwaterAcousticsSSPadePropagator:

    def __init__(self, src: Source, env: UnderwaterEnvironment, max_range_m,
                 comp_params=HelmholtzPropagatorComputationalParams(), max_depth_m=None, c0=None, lower_bc=None):
        self.uwa_env = deepcopy(env)
        self.comp_params = deepcopy(comp_params)
        self.src = deepcopy(src)
        c0 = c0 or min([self.uwa_env.sound_speed_profile_m_s(0, z) for z in range(0, self.uwa_env.bottom_profile.max_depth, 1)])
        self.k0 = 2*cm.pi*self.src.freq_hz / c0

        # prepare Helmholtz environment
        m2_ground = (c0 / env.bottom_sound_speed_m_s) ** 2
        lower_bc = lower_bc or TransparentBC(m2_ground)
        self.helmholtz_env = HelmholtzEnvironment(x_max_m=max_range_m, lower_bc=lower_bc, upper_bc=RobinBC(q1=1, q2=0, q3=0))
        #self.helmholtz_env = HelmholtzEnvironment(x_max_m=max_range_m, lower_bc=RobinBC(q1=0, q2=1, q3=0), upper_bc=RobinBC(q1=1, q2=0, q3=0))
        self.helmholtz_env.z_max = 0
        self.helmholtz_env.z_min = -max_depth_m if max_depth_m else -(self.uwa_env.bottom_profile.max_depth + 300)

        eta = 1 / (40*cm.pi*cm.log10(cm.exp(1)))
        def n2minus1(x, z, freq_hz):
            depth = -self.uwa_env.bottom_profile(x)
            if isinstance(z, float):
                if z > depth:
                    return (c0 / self.uwa_env.sound_speed_profile_m_s(x, -z))**2 - 1
                else:
                    return (c0 / self.uwa_env.bottom_sound_speed_m_s)**2 * (1 + 1j*eta*self.uwa_env.bottom_attenuation_dm_lambda) ** 2 - 1
            else:
                res = z * 0j + (c0 / self.uwa_env.sound_speed_profile_m_s(x, -z))**2 - 1
                ind = z <= depth
                res[ind] = (c0 / self.uwa_env.bottom_sound_speed_m_s)**2 * (1 + 1j*eta*self.uwa_env.bottom_attenuation_dm_lambda) ** 2 - 1
                return res

        self.helmholtz_env.n2minus1 = n2minus1

        def rho(x, z):
            depth = -self.uwa_env.bottom_profile(x)
            if isinstance(z, float):
                if z > depth:
                    return 1.0
                else:
                    return self.uwa_env.bottom_density_g_cm
            else:
                res = z*0 + 1
                ind = z <= depth
                res[ind] = self.uwa_env.bottom_density_g_cm
                return res

        self.helmholtz_env.rho = rho
        self.helmholtz_env.use_rho = True
        self.helmholtz_env.lower_z = lambda x: self.uwa_env.bottom_profile(x)

        wavelength = c0 / src.freq_hz
        self.comp_params.terrain_method = TerrainMethod.pass_through

        self.propagator = HelmholtzPadeSolver(env=self.helmholtz_env, wavelength=wavelength, freq_hz=self.src.freq_hz, params=self.comp_params)

    def calculate(self):
        h_field = self.propagator.calculate(lambda z: self.src.aperture(self.k0, -z))
        res = AcousticPressureField(x_grid=h_field.x_grid_m, z_grid=-h_field.z_grid_m[::-1], freq_hz=self.src.freq_hz, field=h_field.field[:,::-1])
        return res
