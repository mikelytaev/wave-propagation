from uwa.field import *
from uwa.source import *
from uwa.environment import *
from uwa.propagators import *
from uwa.ram import *
from uwa.utils import *

from uwa.vis import AcousticPressureFieldVisualiser2d


logging.basicConfig(level=logging.DEBUG)

src = GaussSource(freq_hz=50, depth=100, beam_width=20, eval_angle=0)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: munk_profile(z)
env.bottom_profile = Bathymetry(ranges_m=[0], depths_m=[5000])
env.bottom_sound_speed_m_s = 1700
env.bottom_density_g_cm = 1.5
env.bottom_attenuation_dm_lambda = 0.5

max_ramge = 150000
sspe_propagator = UnderwaterAcousticsSSPadePropagator(src=src, env=env, max_range_m=max_ramge)
sspe_field = sspe_propagator.calculate()
sspe_field.field *= 5.88
sspe_vis = AcousticPressureFieldVisualiser2d(field=sspe_field, label='WPF')
sspe_vis.plot2d(-70, 0).show()

comp_params = RAMComputationalParams(
    output_ranges=np.arange(0, max_ramge, 250),
    dr=250,
    dz=0.5
)
ram_propagator = RAMMatlabPropagator(src=src, env=env, comp_params=comp_params)
ram_field = ram_propagator.calculate()

ram_vis = AcousticPressureFieldVisualiser2d(field=ram_field, label='RAM')
ram_vis.plot2d(-70, 0).show()
ram_vis.plot_hor(src.depth, sspe_vis)

