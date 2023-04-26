import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
from scipy.optimize import differential_evolution
import math as fm

from examples.optimization.uwa.utils import approx_error, approx_exp
from itertools import zip_longest
import cmath as cm
from examples.optimization.evol import opt_utils as opt_utils

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import propagators._utils as utils


dx = 50
order = (7, 8)
xi_bound = 0.2
xi_bounds = (-xi_bound, 0)


xi_grid = np.linspace(-xi_bound*2, xi_bound*2, 100)

grid_re = np.linspace(-xi_bound*2, xi_bound*2, 500)
grid_im = np.linspace(-xi_bound*2, xi_bound*2, 500)
i_grid, j_grid = np.meshgrid(grid_re, grid_im)
xi_grid_2d = i_grid + 1j*j_grid
shape = xi_grid_2d.shape


pade_coefs, a0 = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2 * cm.pi, dx=dx)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])
errors_pade = approx_error(pade_coefs_num, pade_coefs_den, xi_grid_2d.flatten(), dx).reshape(shape)
plt.figure(figsize=(4, 3.2))
plt.imshow(
    10*np.log10(abs(errors_pade)),
    extent=[grid_re[0], grid_re[-1], grid_im[-1], grid_im[0]],
    cmap=plt.get_cmap('binary'),
    #aspect='auto',
    norm=Normalize(-100, 20)
)
plt.colorbar()
plt.grid(True)
plt.xlabel("Re(error)")
plt.ylabel("Im(error)")
plt.tight_layout()
#plt.show()
plt.savefig("approx_error_2d.eps")



xi_grid = np.linspace(-xi_bound*2, xi_bound*2, 1000)
error_pade = approx_error(pade_coefs_num, pade_coefs_den, xi_grid, dx)

plt.figure(figsize=(4, 3.2))
plt.plot(xi_grid, 10*np.log10(error_pade))
plt.grid(True)
plt.xlim([xi_grid[0], xi_grid[-1]])
plt.xlabel("xi")
plt.ylabel("10log(error)")
plt.tight_layout()
#plt.show()
plt.savefig("approx_error_real.eps")
