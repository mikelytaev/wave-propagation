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

import propagators._utils as utils


dx = 50
order = (6, 7)
theta_max_degrees = 20
xi_bound = fm.sin(theta_max_degrees*fm.pi/180)**2 * 0.3
xi_bounds = (-xi_bound, xi_bound)
bounds_ga = [(-25, 25)] * (order[0] + order[1]) * 2

result_ga = differential_evolution(
    func=fit_func,
    args=(dx, order, xi_bounds),
    bounds=bounds_ga,
    popsize=50,
    disp=True,
    mutation=(0.5, 1.0),
    recombination=1.0,
    strategy='randtobest1exp',
    tol=1e-9,
    maxiter=2000,
    polish=False,
    workers=-1
)

print(result_ga)

def approx_error(num_coefs, den_coefs, xi_grid):
    k0 = 2 * fm.pi
    res = xi_grid * 0.0
    for xi_i, xi in enumerate(xi_grid):
        p = 1.0
        for a, b in zip_longest(num_coefs, den_coefs, fillvalue=0.0j):
            p *= (1 + xi * a) / (1 + xi * b)

        ev = cm.exp(1j * k0 * dx * (cm.sqrt(1 + xi) - 1))
        res[xi_i] = abs(p - ev)
    return res


num_coefs, den_coefs = opt_utils.opt_coefs_to_coefs_ga(result_ga.x, order)
xi_grid = np.linspace(xi_bounds[0]*2, -xi_bounds[0]*2, 100)
error = approx_error(num_coefs, den_coefs, xi_grid)

plt.figure(figsize=(6, 3.2))
plt.plot(xi_grid, error)
plt.grid(True)
plt.tight_layout()
plt.show()


def approx_exp(num_coefs, den_coefs, xi_grid):
    k0 = 2 * fm.pi
    res = xi_grid * 0.0
    for xi_i, xi in enumerate(xi_grid):
        p = 1.0
        for a, b in zip_longest(num_coefs, den_coefs, fillvalue=0.0j):
            p *= (1 + xi * a) / (1 + xi * b)
        res[xi_i] = p
    return res


grid = np.linspace(-0.2, 0.2, 500)
i_grid, j_grid = np.meshgrid(grid, grid)
xi_grid_2d = i_grid + 1j*j_grid
shape = xi_grid_2d.shape
approx_exp_vals = approx_exp(num_coefs, den_coefs, xi_grid_2d.flatten()).reshape(shape)
plt.imshow(abs(approx_exp_vals) > 1, extent=[grid[0], grid[-1], grid[0], grid[-1]], cmap=plt.get_cmap('binary'))
plt.colorbar()
plt.grid(True)
plt.show()

errors_de = approx_error(num_coefs, den_coefs, xi_grid_2d.flatten()).reshape(shape)
plt.imshow(
    10*np.log10(abs(errors_de)),
    extent=[grid[0], grid[-1], grid[0], grid[-1]],
    cmap=plt.get_cmap('binary'),
    norm=Normalize(-90, -40)
)
plt.colorbar()
plt.grid(True)
plt.show()

plt.imshow(
    np.log10(abs(errors_de)) < -5,
    extent=[grid[0], grid[-1], grid[0], grid[-1]],
    cmap=plt.get_cmap('binary'),
    #norm=Normalize(-9, -5)
)
plt.colorbar()
plt.grid(True)
plt.show()

pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2 * cm.pi, dx=dx)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])
errors_pade = approx_error(pade_coefs_num, pade_coefs_den, xi_grid_2d.flatten()).reshape(shape)
plt.imshow(
    np.log10(abs(errors_pade)) < -5,
    extent=[grid[0], grid[-1], grid[0], grid[-1]],
    cmap=plt.get_cmap('binary')
)
plt.colorbar()
plt.grid(True)
plt.show()

xi_grid = np.linspace(xi_bounds[0]*2, -xi_bounds[0]*2, 100)
error_de = approx_error(num_coefs, den_coefs, xi_grid)
error_pade = approx_error(pade_coefs_num, pade_coefs_den, xi_grid)

plt.figure(figsize=(6, 3.2))
plt.plot(xi_grid, 10*np.log10(error_de), label="de")
plt.plot(xi_grid, 10*np.log10(error_pade), label="pade")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()