import numpy as np

import pyximport
import cmath as cm
import math as fm
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
import propagators._utils as utils
import propagators.dispersion_relations as disp_rels
from scipy.optimize import differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt
import mpmath
import examples.optimization.evol.opt_utils as opt_utils


k0 = 2*cm.pi
theta_max_degrees = 10
order = (7, 7)

print(k0*fm.sin(theta_max_degrees * fm.pi / 180))

eps = 1e-3
eps_x_max = 1e3


dx_wl = 200

bounds_ga = [(-100, 100)] * (order[0] + order[1]) * 2

result_ga = differential_evolution(
    func=opt_utils.fit_func_exp_rational_approx_ga,
    args=(dx_wl, order, theta_max_degrees),
    bounds=bounds_ga,
    popsize=15,
    disp=True,
    recombination=1.0,
    strategy='currenttobest1exp',
    tol=1e-9,
    maxiter=100000,
    polish=False,
    workers=-1,
    #callback=lambda xk, convergence: print(xk)
)
print(result_ga)

num_coefs_ga, den_coefs_ga = opt_utils.opt_coefs_to_coefs_ga(result_ga.x, order)


def k_x_angle(dx, num_coefs, den_coefs, kz_arr):
    return np.array([disp_rels.exp_rational_approx_abs_error_point(k0, dx, num_coefs, den_coefs, -kz**2) for kz in kz_arr/k0])


kz_arr = np.linspace(0, k0*fm.sin(theta_max_degrees*fm.pi/180), 10000)
k_x_1 = k_x_angle(dx_wl, num_coefs_ga, den_coefs_ga, kz_arr)

plt.figure(figsize=(6, 3.2))
plt.plot(kz_arr, (np.real(k_x_1)), label='opt')
plt.xlabel('k_z')
plt.ylabel('k_x abs. error')
plt.xlim([kz_arr[0], kz_arr[-1]])
plt.ylim([1e-8, 1e-2])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
