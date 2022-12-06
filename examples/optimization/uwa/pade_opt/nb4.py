import numpy as np

from uwa.propagators import *
from uwa.vis import AcousticPressureFieldVisualiser2d
from examples.optimization.uwa.pade_opt.utils import get_optimal
import propagators._utils as utils

logging.basicConfig(level=logging.DEBUG)

src = GaussSource(freq_hz=1000, depth=100, beam_width=1, eval_angle=-30)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: 1500 + z
env.bottom_profile = Bathymetry(ranges_m=[0, 5000], depths_m=[500, 500])
env.bottom_sound_speed_m_s = 2000
#env.bottom_density_g_cm = 1.0
env.bottom_attenuation_dm_lambda = 0.0

max_range = 3000
prec = 1e-1

wavelength = 1500 / src.freq_hz
max_range_wl = max_range / wavelength

pade_order = (7, 8)

k0 = 2*fm.pi
theta_max_degrees = 30
k_z_max = k0*fm.sin(theta_max_degrees*fm.pi/180)
xi_bounds = [-k_z_max**2/k0**2+((1500/2000)**2-1)*0, 0]

dr_wl, dz_wl = get_optimal(max_range_wl, prec, xi_bounds, k_z_max, pade_order=pade_order, shift_pade=False)
print(f"Pade: dx = {dr_wl}; dz = {dz_wl}")

sspe_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr_wl,
    dz_wl=dz_wl,
    exp_pade_order=pade_order,
    sqrt_alpha=0,
    modify_grid=False
)

sspe_propagator = UnderwaterAcousticsSSPadePropagator(
    src=src,
    env=env,
    max_range_m=max_range,
    max_depth_m=600,
    comp_params=sspe_comp_params
)
sspe_field = sspe_propagator.calculate()
sspe_field.field *= 5.50 #normalization
sspe_vis = AcousticPressureFieldVisualiser2d(field=sspe_field, label='WPF')
sspe_vis.plot2d(-90, -5).show()
