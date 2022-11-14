import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
from scipy.optimize import differential_evolution
import math as fm

from examples.optimization.uwa.utils import fit_func
from itertools import zip_longest
import cmath as cm
from examples.optimization.evol import opt_utils as opt_utils

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from utils import approx_error

import propagators._utils as utils

from uwa.field import *
from uwa.source import *
from uwa.environment import *
from uwa.propagators import *
from uwa.utils import *

from uwa.vis import AcousticPressureFieldVisualiser2d


dr_wl = 20
dz_wl = 0.1
pade_order = (7, 8)
xi_bound = 0.2
xi_bounds = (-xi_bound, 0)
bounds_ga = [(-60, 60)] * (pade_order[0] + pade_order[1]) * 2

result_ga = differential_evolution(
    func=opt_utils.fit_func_exp_rational_approx_ga,
    args=(dr_wl, pade_order, xi_bounds),
    bounds=bounds_ga,
    popsize=20,
    disp=True,
    mutation=(0.5, 1.0),
    recombination=1.0,
    strategy='currenttobest1exp',
    tol=1e-19,
    maxiter=5000,
    polish=False,
    workers=-1
)

print(result_ga)
print(min(result_ga.x))
print(max(result_ga.x))


num_coefs, den_coefs = opt_utils.opt_coefs_to_coefs_ga(result_ga.x, pade_order)
xi_grid = np.linspace(xi_bounds[0]*2, -xi_bounds[0]*2, 100)

grid_re = np.linspace(-xi_bound*2, 0.1, 500)
grid_im = np.linspace(-0.1, xi_bound*2, 500)
i_grid, j_grid = np.meshgrid(grid_re, grid_im)
xi_grid_2d = i_grid + 1j*j_grid
shape = xi_grid_2d.shape

errors_de = approx_error(num_coefs, den_coefs, xi_grid_2d.flatten(), dr_wl).reshape(shape)

plt.imshow(
    np.log10(abs(errors_de)) < -3,
    extent=[grid_re[0], grid_re[-1], grid_im[-1], grid_im[0]],
    cmap=plt.get_cmap('binary'),
)
plt.colorbar()
plt.grid(True)
plt.show()

pade_coefs = utils.pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, k0=2 * cm.pi, dx=dr_wl)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])
errors_pade = approx_error(pade_coefs_num, pade_coefs_den, xi_grid_2d.flatten(), dr_wl).reshape(shape)
plt.imshow(
    np.log10(abs(errors_pade)) < -3,
    extent=[grid_re[0], grid_re[-1], grid_im[-1], grid_im[0]],
    cmap=plt.get_cmap('binary')
)
plt.colorbar()
plt.grid(True)
plt.show()



src = GaussSource(freq_hz=500, depth=30, beam_width=45, eval_angle=0)
env = UnderwaterEnvironment()
env.sound_speed_profile_m_s = lambda x, z: 1500 - z/2
env.bottom_profile = Bathymetry(ranges_m=[0, 50000], depths_m=[100, 100])
env.bottom_sound_speed_m_s = 1700
env.bottom_density_g_cm = 1.5
env.bottom_attenuation_dm_lambda = 0.5

max_range_m = 15000

sspe_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr_wl,
    dz_wl=dz_wl,
    exp_pade_order=pade_order,
    sqrt_alpha=0,
    modify_grid=False,
    exp_pade_coefs=list(zip_longest(num_coefs, den_coefs, fillvalue=0.0j)),
)

sspe_propagator = UnderwaterAcousticsSSPadePropagator(src=src, env=env, max_range_m=max_range_m, max_depth_m=150, comp_params=sspe_comp_params)
sspe_field = sspe_propagator.calculate()
sspe_field.field *= 5.50 #normalization
sspe_vis = AcousticPressureFieldVisualiser2d(field=sspe_field, label='WPF')
sspe_vis.plot2d(-50, -5).show()


de_comp_params = HelmholtzPropagatorComputationalParams(
    z_order=4,
    dx_wl=dr_wl,
    dz_wl=dz_wl,
    exp_pade_order=pade_order,
    sqrt_alpha=0,
    modify_grid=False
)

de_propagator = UnderwaterAcousticsSSPadePropagator(src=src, env=env, max_range_m=max_range_m, max_depth_m=150, comp_params=de_comp_params)
de_field = de_propagator.calculate()
de_field.field *= 5.50 #normalization
de_vis = AcousticPressureFieldVisualiser2d(field=de_field, label='WPF')
de_vis.plot2d(-50, -5).show()