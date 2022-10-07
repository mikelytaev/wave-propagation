import numpy as np

import pyximport
import cmath as cm
import math as fm
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
import propagators._utils as utils
import propagators.dispersion_relations as disp_rels
from scipy.optimize import differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt


def opt_coefs_to_coefs(coefs_arr, order):
    n = order[0]
    m = order[1]
    num_coefs = np.array([coefs_arr[2 + 2 * i] + 1j * coefs_arr[2 + 2 * i + 1] for i in range(0, n)])
    den_coefs = np.array([coefs_arr[2+ 2 * n + 2 * i] + 1j * coefs_arr[2 + 2 * n + 2 * i + 1] for i in range(0, m)])
    return num_coefs, den_coefs



def opt_coefs_to_grids(coefs_arr):
    dx = coefs_arr[0]
    dz = coefs_arr[1]
    return dx, dz


def coefs_to_opt_coefs(coefs):
    co = []
    for c in coefs:
        co += [c[0].real, c[0].imag]
    for c in coefs:
        co += [c[1].real, c[1].imag]
    return co


k0 = 2*cm.pi
theta_max_degrees = 22
order = (6, 7)

print(k0*fm.sin(theta_max_degrees * fm.pi / 180))


def fit_func(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)
    return 1 / (dx * dz)


eps = 3e-4
eps_x_max = 3e3


def constraint_ga(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)
    num_coefs, den_coefs = opt_coefs_to_coefs(coefs_arr, order)
    err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs, k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                        round(theta_max_degrees) * 5) / dx / k0
    return err


def constraint_pade_2nd_order(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)
    pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2*cm.pi, dx=dx)
    num_coefs = np.array([a[0] for a in pade_coefs])
    den_coefs = np.array([a[1] for a in pade_coefs])
    err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs, k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                        round(theta_max_degrees) * 5) / dx / k0
    return err


bounds_pade = [(0.1, 100), (0.0001, 1)]

result_pade = differential_evolution(
    fit_func,
    bounds_pade,
    constraints=(NonlinearConstraint(constraint_pade_2nd_order, 0, eps/eps_x_max)),
    popsize=15,
    disp=True,
    recombination=1,
    strategy='randtobest1exp',
    tol=1e-9,
    polish=False,
    maxiter=2000,
    workers=-1,
    callback=lambda xk, convergence: print(xk)
)
print(result_pade)
dx_pade, dz_pade = opt_coefs_to_grids(result_pade.x)
pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2*cm.pi, dx=dx_pade)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])

bounds_ga = bounds_pade + [(-50, 50)] * (order[0] + order[1]) * 2

result_ga = differential_evolution(
    fit_func,
    bounds_ga,
    constraints=(NonlinearConstraint(constraint_ga, 0, eps/eps_x_max)),
    popsize=15,
    disp=True,
    recombination=1.0,
    strategy='randtobest1exp',
    tol=1e-9,
    maxiter=4000,
    polish=False,
    workers=-1,
    callback=lambda xk, convergence: print(str(constraint_ga(xk)) + " " + str(opt_coefs_to_grids(xk)))
)
print(result_ga)
dx_ga, dz_ga = opt_coefs_to_grids(result_ga.x)
ga_coefs_num, ga_coefs_den = opt_coefs_to_coefs(result_ga.x, order)