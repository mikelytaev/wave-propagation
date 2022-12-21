from uwa.propagators import *
from uwa.vis import AcousticPressureFieldVisualiser2d
from examples.optimization.uwa.pade_opt.utils import get_optimal
import propagators._utils as utils

#logging.basicConfig(level=logging.DEBUG)

src = GaussSource(freq_hz=1000, depth=100, beam_width=1, eval_angle=-30)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: 1500 + z*0
env.bottom_profile = Bathymetry(ranges_m=[0, 5000], depths_m=[300, 300])
env.bottom_sound_speed_m_s = 1500
#env.bottom_density_g_cm = 1.0
env.bottom_attenuation_dm_lambda = 0.0

max_range = 3000
prec = 1e-3

wavelength = 1500 / src.freq_hz
max_range_wl = max_range / wavelength

pade_order = (7, 8)

k0 = 2*fm.pi
theta_max_degrees = 31
k_z_max = k0*fm.sin(theta_max_degrees*fm.pi/180)
xi_bounds = [-k_z_max**2/k0**2+((1590/1500)**2-1), ((1590/1500)**2-1)]
dr_wl_s, dz_wl_s = get_optimal(max_range_wl, prec, xi_bounds, k_z_max, pade_order=pade_order)