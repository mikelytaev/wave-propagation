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
from utils import approx_error, approx_exp

import propagators._utils as utils


dx = 25
order = (8, 8)
#theta_max_degrees = 20
xi_bound = 0.3
xi_bounds = (-xi_bound, 0)
bounds_ga = [(-60, 60)] * (order[0] + order[1]) * 2

result_ga = differential_evolution(
    func=opt_utils.fit_func_exp_rational_approx_ga,
    args=(dx, order, xi_bounds),
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


num_coefs, den_coefs = opt_utils.opt_coefs_to_coefs_ga(result_ga.x, order)
xi_grid = np.linspace(xi_bounds[0]*2, -xi_bounds[0]*2, 100)
error = approx_error(num_coefs, den_coefs, xi_grid, dx)

plt.figure(figsize=(6, 3.2))
plt.plot(xi_grid, error)
plt.grid(True)
plt.tight_layout()
plt.show()


grid_re = np.linspace(-xi_bound*2, 0.1, 500)
grid_im = np.linspace(-0.1, xi_bound*2, 500)
i_grid, j_grid = np.meshgrid(grid_re, grid_im)
xi_grid_2d = i_grid + 1j*j_grid
shape = xi_grid_2d.shape
approx_exp_vals = approx_exp(num_coefs, den_coefs, xi_grid_2d.flatten()).reshape(shape)
# plt.imshow(abs(approx_exp_vals) > 1, extent=[grid_re[0], grid_re[-1], grid_im[0], grid_im[-1]], cmap=plt.get_cmap('binary'))
# plt.title("DE > 1")
# plt.colorbar()
# plt.grid(True)
# plt.show()

errors_de = approx_error(num_coefs, den_coefs, xi_grid_2d.flatten(), dx).reshape(shape)
# plt.imshow(
#     10*np.log10(abs(errors_de)),
#     extent=[grid_re[0], grid_re[-1], grid_im[0], grid_im[-1]],
#     cmap=plt.get_cmap('binary'),
#     norm=Normalize(-90, -40)
# )
# plt.colorbar()
# plt.grid(True)
# plt.show()

plt.imshow(
    np.log10(abs(errors_de)) < -3,
    extent=[grid_re[0], grid_re[-1], grid_im[-1], grid_im[0]],
    cmap=plt.get_cmap('binary'),
    #norm=Normalize(-9, -5)
)
plt.colorbar()
plt.grid(True)
plt.show()

pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2 * cm.pi, dx=dx)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])
errors_pade = approx_error(pade_coefs_num, pade_coefs_den, xi_grid_2d.flatten(), dx).reshape(shape)
plt.imshow(
    np.log10(abs(errors_pade)) < -3,
    extent=[grid_re[0], grid_re[-1], grid_im[-1], grid_im[0]],
    cmap=plt.get_cmap('binary')
)
plt.colorbar()
plt.grid(True)
plt.show()

# pade_exp = approx_exp(pade_coefs_num, pade_coefs_den, xi_grid_2d.flatten()).reshape(shape)
# plt.imshow(abs(pade_exp) > 1, extent=[grid_re[0], grid_re[-1], grid_im[0], grid_im[-1]], cmap=plt.get_cmap('binary'))
# plt.title("Pade > 1")
# plt.colorbar()
# plt.grid(True)
# plt.show()

xi_grid = np.linspace(-xi_bound*2, 0, 100)
error_de = approx_error(num_coefs, den_coefs, xi_grid)
error_pade = approx_error(pade_coefs_num, pade_coefs_den, xi_grid)

# plt.figure(figsize=(6, 3.2))
# plt.plot(xi_grid, 10*np.log10(error_de), label="de")
# plt.plot(xi_grid, 10*np.log10(error_pade), label="pade")
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
# plt.show()