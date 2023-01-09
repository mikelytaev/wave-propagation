import numpy as np

import pyximport
import cmath as cm
import math as fm

from examples.optimization.uwa.pade_opt.utils import get_optimal

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


def fit_func(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)
    return 1 / (dx * dz)


eps = 3e-4
eps_x_max = 3e3


def get_pade_opt_grid(*, theta_max_degrees, order, grid_bounds=None):
    def constraint_pade_2nd_order(coefs_arr):
        dx, dz = opt_coefs_to_grids(coefs_arr)
        pade_coefs, _ = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2 * cm.pi, dx=dx)
        num_coefs = np.array([a[0] for a in pade_coefs])
        den_coefs = np.array([a[1] for a in pade_coefs])
        err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs,
                                            k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                            round(theta_max_degrees) * 5) / dx / k0
        return err

    grid_bounds = grid_bounds or [(0.1, 100), (0.0001, 1)]
    result_pade = differential_evolution(
        fit_func,
        grid_bounds,
        constraints=(NonlinearConstraint(constraint_pade_2nd_order, 0, eps / eps_x_max)),
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
    return dx_pade, dz_pade


def get_de_opt_grid(*, theta_max_degrees, order, coef_bounds, grid_bounds=None):
    grid_bounds = grid_bounds or [(0.1, 100), (0.0001, 1)]
    bounds_ga = grid_bounds + [coef_bounds] * (order[0] + order[1]) * 2

    def constraint_ga(coefs_arr):
        dx, dz = opt_coefs_to_grids(coefs_arr)
        num_coefs, den_coefs = opt_coefs_to_coefs(coefs_arr, order)
        err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs,
                                            k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                            round(theta_max_degrees) * 5) / dx / k0
        return err

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
    dx_ga, dz_ga = opt_coefs_to_grids(result_ga.x)
    return dx_ga, dz_ga


# theta_max = 3
# dx_pade, dz_pade = get_pade_opt_grid(theta_max_degrees=theta_max, order=(6, 7), grid_bounds=[(0, 2500), (0, 5)])
# print(f"Pade: dx = {dx_pade}; dz = {dz_pade}")
# dx_de, dz_de = get_de_opt_grid(theta_max_degrees=theta_max, order=(6, 7), coef_bounds=(-500, 500), grid_bounds=[(dx_pade, 5000), (dz_pade, 10)])
# print(f"DE: dx = {dx_de}; dz = {dz_de}")

# theta_max = 10
# dx_pade, dz_pade = get_pade_opt_grid(theta_max_degrees=theta_max, order=(6, 7), grid_bounds=[(0, 150), (0, 0.5)])
# print(f"Pade: dx = {dx_pade}; dz = {dz_pade}")
# dx_de, dz_de = get_de_opt_grid(theta_max_degrees=theta_max, order=(6, 7), coef_bounds=(-200, 200), grid_bounds=[(dx_pade, 500), (dz_pade, 3.0)])
# print(f"DE: dx = {dx_de}; dz = {dz_de}")

# theta_max = 22
# dx_pade, dz_pade = get_pade_opt_grid(theta_max_degrees=theta_max, order=(6, 7), grid_bounds=[(0, 70), (0, 0.06)])
# print(f"Pade: dx = {dx_pade}; dz = {dz_pade}")
# dx_de, dz_de = get_de_opt_grid(theta_max_degrees=theta_max, order=(6, 7), coef_bounds=(-50, 50), grid_bounds=[(dx_pade, 300), (dz_pade, 3.0)])
# print(f"DE: dx = {dx_de}; dz = {dz_de}")

theta_max = 30
dx_pade, dz_pade = get_pade_opt_grid(theta_max_degrees=theta_max, order=(6, 7), grid_bounds=[(0, 10.8), (0, 0.005)])
print(f"Pade: dx = {dx_pade}; dz = {dz_pade}")
dx_de, dz_de = get_de_opt_grid(theta_max_degrees=theta_max, order=(6, 7), coef_bounds=(-20, 20), grid_bounds=[(dx_pade, 50), (dz_pade, 0.7)])
print(f"DE: dx = {dx_de}; dz = {dz_de}")


# dx_pade, dz_pade = get_pade_opt_grid(theta_max_degrees=22, order=(6, 7))
# print(f"Pade: dx = {dx_pade}; dz = {dz_pade}")
# dx_de, dz_de = get_de_opt_grid(theta_max_degrees=22, order=(6, 7), coef_bounds=(-50, 50), grid_bounds=[(dx_pade, 100), (dz_pade, 1)])
# print(f"DE: dx = {dx_de}; dz = {dz_de}")