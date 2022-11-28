import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

import utils as utils

from examples.jcs.d2_error import *
import math as fm
from propagators._utils import *


import matplotlib.pyplot as plt
# dz = 20
# errors_vs_kz = [fourth_order_error_kz(kz, dz) for kz in np.linspace(-fm.pi*3, fm.pi*3, 2000)]
# plt.plot(errors_vs_kz)
# plt.grid(True)
#plt.show()

# kz = 0.25#*fm.pi
# z_grid = np.linspace(0, 40, 2000)
# errors_vs_dz = [fourth_order_error_kz(kz, dz) for dz in z_grid]
# plt.plot(z_grid, errors_vs_dz)
# plt.grid(True)
#plt.show()


def h_error(dz_wl, k_z_max):
    return fourth_order_error_kz(k_z_max, dz_wl)


def tau_error(xi1, xi2, dx_wl, pade_coefs):
    k0 = 2 * fm.pi
    return abs(cm.exp(1j*k0*dx_wl*(cm.sqrt(1 + xi1) - 1)) - fm.prod([(1 + a*xi2) / (1 + b*xi2) for a, b in pade_coefs]))


def tau_error_sup_h(xi, xi_bounds, h, n, dx_wl, pade_coefs_num, pade_coefs_den, c0):
    xi_grid = np.linspace(xi-h, xi+h, n)
    xi_grid = xi_grid[xi_grid >= xi_bounds[0]]
    xi_grid = xi_grid[xi_grid <= xi_bounds[1]]
    errors = [utils.tau_error(xi, xi2, dx_wl, pade_coefs_num, pade_coefs_den, c0) for xi2 in xi_grid]
    return max(errors)


def precision_step(xi_bounds, k_z_max, dxs_wl: np.array, dzs_wl: np.array, pade_order, shift_pade=False):
    k0 = 2 * fm.pi
    res = np.zeros((len(dxs_wl), len(dzs_wl)))
    xi_grid = np.linspace(xi_bounds[0], xi_bounds[1], 100)
    for dx_i, dx_wl in enumerate(dxs_wl):
        coefs, c0 = pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, k0=k0, dx=dx_wl, a0=((xi_bounds[0]+xi_bounds[1])/2 if shift_pade else 0))
        pade_coefs_num = np.array([a[0] for a in coefs])
        pade_coefs_den = np.array([a[1] for a in coefs])
        for dz_i, dz_wl in enumerate(dzs_wl):
            h = h_error(dz_wl, k_z_max) / k0**2
            error = max([tau_error_sup_h(xi, xi_bounds, h, 100, dx_wl, pade_coefs_num, pade_coefs_den, c0) for xi in xi_grid])
            res[dx_i, dz_i] = error
    return res


def get_optimal(x_max_wl, prec, xi_min, k_z_max, shift_pade=False):
    pade_order = (8, 8)
    dxs_wl = np.concatenate((
        #[0.0005],
        #[0.001],
        [0.01],
        np.linspace(0.1, 1, 10),
        np.linspace(2, 10, 9),
        np.linspace(20, 100, 9),
    ))
    dzs_wl = np.concatenate((
        #[0.00001],
        [0.0001],
        np.linspace(0.001, 0.01, 10),
        np.linspace(0.02, 0.1, 9),
        np.linspace(0.2, 1, 9),
    ))
    errors = precision_step([xi_min, 0], k_z_max, dxs_wl, dzs_wl, pade_order, shift_pade)
    cur_best_dx = 1e-16
    cur_best_dz = 1e-16
    for dx_i, dx in enumerate(dxs_wl):
        for dz_i, dz in enumerate(dzs_wl):
            err = errors[dx_i, dz_i]
            if x_max_wl / dx * err < prec:
                if dx * dz > cur_best_dx * cur_best_dz:
                    cur_best_dx, cur_best_dz = dx, dz

    return cur_best_dx, cur_best_dz


k0 = 2*fm.pi
theta_max_degrees = 70
k_z_max = k0*fm.sin(theta_max_degrees*fm.pi/180)
xi_bounds = [-k_z_max**2/k0**2, 0]
# z_grid = np.linspace(0.01, 5, 100)
# errors = precision_step(xi_bounds, k_z_max, [10], z_grid, (7, 8))
#
# plt.plot(z_grid, 10*np.log10(errors[0]))
# plt.grid(True)
# plt.show()

dx, dz = get_optimal(10000, 1e-2, xi_bounds[0], k_z_max, shift_pade=False)
print(f"dx = {dx}; dz = {dz}")

dx, dz = get_optimal(10000, 1e-2, xi_bounds[0], k_z_max, shift_pade=True)
print(f"dx = {dx}; dz = {dz}")
