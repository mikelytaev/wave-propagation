from uwa.field import *
from uwa.source import *
from uwa.environment import *
from propagators.sspade import *
from copy import deepcopy
from uwa._optimization.utils import get_optimal


@dataclass
class UWASSpadeComputationalParams:
    max_range_m: float = None
    max_depth_m: float = None
    c0: float = None,
    rational_approx_order = (7, 8)
    dx_m: float = None
    dz_m: float = None
    precision: float = 0.01


def uwa_ss_pade(src: Source, env: UnderwaterEnvironment, params: UWASSpadeComputationalParams) -> AcousticPressureField:
    propagator = UnderwaterAcousticsSSPadePropagator(
        src=src,
        env=env,
        max_range_m=params.max_range_m,
        max_depth_m=params.max_depth_m,
        comp_params=HelmholtzPropagatorComputationalParams(
            exp_pade_order=params.rational_approx_order,
            grid_optimizator_abs_threshold=params.precision
        )
    )
    return propagator.calculate()

class UnderwaterAcousticsSSPadePropagator:

    def __init__(self, src: Source, env: UnderwaterEnvironment, max_range_m,
                 comp_params=HelmholtzPropagatorComputationalParams(), max_depth_m=None, c0=None, lower_bc=None):
        self.uwa_env = deepcopy(env)
        self.comp_params = deepcopy(comp_params)
        self.src = deepcopy(src)

        if self.comp_params.max_propagation_angle is None:
            self.comp_params.max_propagation_angle = self.src.max_angle_deg()

        c_min = np.min(self.uwa_env.sound_speed_profile_m_s(0, np.linspace(0, self.uwa_env.bottom_profile.max_depth(), 5000)))
        c_max = np.max(self.uwa_env.sound_speed_profile_m_s(0, np.linspace(0, self.uwa_env.bottom_profile.max_depth(), 5000)))
        dx, dz, c0, _ = get_optimal(
            freq_hz=src.freq_hz,
            x_max_m=max_range_m,
            prec=comp_params.grid_optimizator_abs_threshold,
            theta_max_degrees=self.comp_params.max_propagation_angle,
            pade_order=comp_params.exp_pade_order,
            z_order=4,
            c_bounds=[c_min, c_max],
            return_meta=True
        )
        wavelength = c0 / src.freq_hz
        self.comp_params.dx_wl = dx / wavelength
        self.comp_params.dz_wl = dz / wavelength
        self.comp_params.modify_grid = False

        c0 = c0 or min([self.uwa_env.sound_speed_profile_m_s(0, z) for z in range(0, self.uwa_env.bottom_profile.max_depth(), 1)])
        self.k0 = 2*cm.pi*self.src.freq_hz / c0
        self.c0 = c0

        # prepare Helmholtz environment
        m2_ground = (c0 / env.bottom_sound_speed_m_s) ** 2
        lower_bc = lower_bc or TransparentBC(m2_ground)
        self.helmholtz_env = HelmholtzEnvironment(x_max_m=max_range_m, lower_bc=lower_bc, upper_bc=RobinBC(q1=1, q2=0, q3=0))
        self.helmholtz_env.z_max = 0
        self.helmholtz_env.z_min = -max_depth_m if max_depth_m else -(self.uwa_env.bottom_profile.max_depth() + 300)

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

        self.comp_params.terrain_method = TerrainMethod.pass_through

        self.propagator = HelmholtzPadeSolver(
            env=self.helmholtz_env,
            wavelength=wavelength,
            freq_hz=self.src.freq_hz,
            params=self.comp_params
        )

    def calculate(self) -> AcousticPressureField:
        h_field = self.propagator.calculate(lambda z: self.src.aperture(self.k0, -z, self.c0 / self.uwa_env.sound_speed_profile_m_s(self.src.depth_m, 0)))
        res = AcousticPressureField(x_grid=h_field.x_grid_m, z_grid=-h_field.z_grid_m[::-1], freq_hz=self.src.freq_hz, field=h_field.field[:,::-1])
        return res
