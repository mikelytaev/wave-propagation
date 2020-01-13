from uwa.field import *
from uwa.source import *
from uwa.environment import *
from uwa.propagators import *
from uwa.ram import *
from uwa.utils import *

from uwa.vis import AcousticPressureFieldVisualiser2d


logging.basicConfig(level=logging.DEBUG)

src = GaussSource(freq_hz=50, depth=50, beam_width=45, eval_angle=0)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: z*0 + 1500
env.bottom_profile = Bathymetry(ranges_m=[0, 50000, 100000, 150000], depths_m=[500, 500, 1000, 1000])
env.bottom_sound_speed_m_s = 1700
env.bottom_density_g_cm = 1.5
env.bottom_attenuation_dm_lambda = 0.5

max_range = 150000

wavelength = 1500 / src.freq_hz
dr_wl = 250 / wavelength
dz_wl = 1 / wavelength

sspe_comp_params = HelmholtzPropagatorComputationalParams(z_order=4, dx_wl=dr_wl, dz_wl=dz_wl, exp_pade_order=(4, 4), sqrt_alpha=0)
sspe_propagator = UnderwaterAcousticsSSPadePropagator(src=src, env=env, max_range_m=max_range, comp_params=sspe_comp_params)
sspe_field = sspe_propagator.calculate()
sspe_field.field *= 5.50 #normalization
sspe_vis = AcousticPressureFieldVisualiser2d(field=sspe_field, label='WPF')
sspe_vis.plot2d(-70, 0).show()

comp_params = RAMComputationalParams(
    output_ranges=np.arange(0, max_range, 250),
    dr=dr_wl * wavelength,
    dz=dz_wl * wavelength
)
ram_propagator = RAMMatlabPropagator(src=src, env=env, comp_params=comp_params)
ram_field = ram_propagator.calculate()

ram_vis = AcousticPressureFieldVisualiser2d(field=ram_field, label='RAM')
ram_vis.plot2d(-70, 0).show()
ram_vis.plot_hor(src.depth, sspe_vis).show()

