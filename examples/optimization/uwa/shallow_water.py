from uwa.field import *
from uwa.source import *
from uwa.environment import *
from uwa.propagators import *
from uwa.utils import *

from uwa.vis import AcousticPressureFieldVisualiser2d
import propagators._utils as utils
from utils import approx_error, approx_exp
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

src = GaussSource(freq_hz=500, depth=50, beam_width=5, eval_angle=-30)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: 1500 + z/2*0
env.bottom_profile = Bathymetry(ranges_m=[0, 5000], depths_m=[200, 200])
env.bottom_sound_speed_m_s = 1700
env.bottom_density_g_cm = 1.5
env.bottom_attenuation_dm_lambda = 0.5

max_range = 3000

wavelength = 1500 / src.freq_hz
max_range_wl = max_range / wavelength

dr_wl = 5
dz_wl = 0.1
pade_order = (8, 8)

xi_bound = 0.5
grid_re = np.linspace(-xi_bound, 0.1, 500)
grid_im = np.linspace(-0.1, xi_bound, 500)
i_grid, j_grid = np.meshgrid(grid_re, grid_im)
xi_grid_2d = i_grid + 1j*j_grid
shape = xi_grid_2d.shape
pade_coefs, c0 = utils.pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, k0=2 * cm.pi, dx=dr_wl)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])
errors_pade = approx_error(pade_coefs_num, pade_coefs_den, xi_grid_2d.flatten(), dr_wl).reshape(shape) * (max_range_wl / dr_wl)
plt.imshow(
    np.log10(abs(errors_pade)) < -1,
    extent=[grid_re[0], grid_re[-1], grid_im[-1], grid_im[0]],
    cmap=plt.get_cmap('binary')
)
plt.colorbar()
plt.grid(True)
plt.show()

approx_exp_vals = approx_exp(pade_coefs_num, pade_coefs_den, xi_grid_2d.flatten()).reshape(shape)
plt.imshow(abs(approx_exp_vals) > 1, extent=[grid_re[0], grid_re[-1], grid_im[-1], grid_im[0]], cmap=plt.get_cmap('binary'))
plt.colorbar()
plt.grid(True)
plt.show()

sspe_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr_wl,
    dz_wl=dz_wl,
    exp_pade_order=pade_order,
    sqrt_alpha=0,
    modify_grid=False
)

sspe_propagator = UnderwaterAcousticsSSPadePropagator(src=src, env=env, max_range_m=max_range, max_depth_m=300, comp_params=sspe_comp_params)
sspe_field = sspe_propagator.calculate()
sspe_field.field *= 5.50 #normalization
sspe_vis = AcousticPressureFieldVisualiser2d(field=sspe_field, label='WPF')
sspe_vis.plot2d(-60, -5).show()
