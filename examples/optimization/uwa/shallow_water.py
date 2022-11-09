from uwa.field import *
from uwa.source import *
from uwa.environment import *
from uwa.propagators import *
from uwa.utils import *

from uwa.vis import AcousticPressureFieldVisualiser2d


logging.basicConfig(level=logging.DEBUG)

src = GaussSource(freq_hz=400, depth=20, beam_width=45, eval_angle=0)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: z*0 + 1500
env.bottom_profile = Bathymetry(ranges_m=[0, 50000], depths_m=[100, 100])
env.bottom_sound_speed_m_s = 1700
env.bottom_density_g_cm = 1.5
env.bottom_attenuation_dm_lambda = 0.5

max_range = 100000

wavelength = 1500 / src.freq_hz
dr_wl = 10
dz_wl = 0.1

sspe_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr_wl,
    dz_wl=dz_wl,
    exp_pade_order=(4, 4),
    sqrt_alpha=0
)

sspe_propagator = UnderwaterAcousticsSSPadePropagator(src=src, env=env, max_range_m=max_range, max_depth_m=200, comp_params=sspe_comp_params)
sspe_field = sspe_propagator.calculate()
sspe_field.field *= 5.50 #normalization
sspe_vis = AcousticPressureFieldVisualiser2d(field=sspe_field, label='WPF')
sspe_vis.plot2d(-50, -5).show()

