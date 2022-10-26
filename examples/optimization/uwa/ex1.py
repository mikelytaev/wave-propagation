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


dx = 10
order = (6, 7)
theta_max_degrees = 20
xi_bounds = (-fm.sin(theta_max_degrees*fm.pi/180)**2, 0)
bounds_ga = [(-10, 10)] * (order[0] + order[1]) * 2

result_ga = differential_evolution(
    func=fit_func,
    args=(dx, order, xi_bounds),
    bounds=bounds_ga,
    popsize=50,
    disp=True,
    mutation=(0.5, 1),
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