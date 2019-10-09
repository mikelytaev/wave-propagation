from uwa.field import *
from uwa.source import *
from uwa.environment import *
from uwa.ram import *
from uwa.utils import *


src = Source(freq_hz=50, depth=100)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: munk_profile(z)
env.bottom_profile = Bathymetry(ranges_m=[0], depths_m=[5000])
env.bottom_sound_speed_m_s = 1700
env.bottom_density_g_cm = 1.5
env.bottom_attenuation_dm_lambda = 0.5

comp_params = RAMComputationalParams(
    output_ranges = np.arange(0, 150000, 250),
    dr = 250,
    dz = 0.5
)
ram_propagator = RAMMatlabPropagator(src=src, env=env, comp_params=comp_params)
field = ram_propagator.calculate()
