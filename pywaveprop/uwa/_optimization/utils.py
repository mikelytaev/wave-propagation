import cmath as cm

import math as fm
import numpy as np

from pywaveprop.propagators._utils import pade_propagator_coefs

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
from pywaveprop.uwa._optimization import utils_pyx as utils_pyx


def second_difference_disp_rel(k_z: complex, dz: float, z=0):
    return cm.exp(1j*k_z*z) * (cm.exp(-1j*k_z*dz) - 2 + cm.exp(1j*k_z*dz))


def fourth_difference_disp_rel(k_z: complex, dz: float, z=0):
    return cm.exp(1j*k_z*z) * (cm.exp(-1j*k_z*dz) - 2 + cm.exp(1j*k_z*dz))**2


def second_order_error_kz(k_z: float, dz: float):
    d = 1 / dz**2 * second_difference_disp_rel(k_z, dz)
    return abs(d - (-k_z**2))


def fourth_order_error_kz(k_z: float, dz: float):
    d = 1 / dz**2 * (second_difference_disp_rel(k_z, dz) - 1/12 * fourth_difference_disp_rel(k_z, dz))
    return abs(d - (-k_z**2))


def h_error(dz, k_z_max, z_order=4):
    if z_order == 2:
        return second_order_error_kz(k_z_max, dz)
    else:
        return fourth_order_error_kz(k_z_max, dz)


def tau_error(xi1, xi2, dx_m, pade_coefs):
    k0 = 2 * fm.pi
    return abs(cm.exp(1j*k0*dx_m*(cm.sqrt(1 + xi1) - 1)) - fm.prod([(1 + a*xi2) / (1 + b*xi2) for a, b in pade_coefs]))


def tau_error_sup_h(k0, xi, xi_bounds, h, n, dx_m, pade_coefs_num, pade_coefs_den, c0):
    xi_grid = np.linspace(xi-h, xi+h, n)
    xi_grid = xi_grid[xi_grid >= xi_bounds[0]]
    xi_grid = xi_grid[xi_grid <= xi_bounds[1]]
    errors = [utils_pyx.tau_error(k0, xi, xi2, dx_m, pade_coefs_num, pade_coefs_den, c0) for xi2 in xi_grid]
    return max(errors)


def precision_step(k0, xi_bounds, k_z_max, dxs_m: np.array, dzs_wl: np.array, pade_order, z_order, shift_pade=False):
    res = np.zeros((len(dxs_m), len(dzs_wl)))
    xi_grid = np.linspace(xi_bounds[0], xi_bounds[1], 100)
    for dx_i, dx_m in enumerate(dxs_m):
        coefs, c0 = pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, beta=k0, dx=dx_m, a0=((xi_bounds[0] + xi_bounds[1]) / 2 if shift_pade else 0))
        pade_coefs_num = np.array([a[0] for a in coefs])
        pade_coefs_den = np.array([a[1] for a in coefs])
        for dz_i, dz_m in enumerate(dzs_wl):
            h = h_error(dz_m, k_z_max, z_order) / k0**2
            error = max([tau_error_sup_h(k0, xi, xi_bounds, h, 100, dx_m, pade_coefs_num, pade_coefs_den, c0) for xi in xi_grid])
            res[dx_i, dz_i] = error
    return res


def get_optimal(*, freq_hz, x_max_m, prec, theta_max_degrees, pade_order, z_order=4, c_bounds, c0=None, return_meta=False):
    c0 = c0 or fm.sqrt((2 + fm.sin(theta_max_degrees * fm.pi / 180)**2) / (1 / c_bounds[0]**2 + 1/c_bounds[1]**2))
    k0 = 2 * fm.pi * freq_hz / c0
    k_z_max = k0 * fm.sin(theta_max_degrees * fm.pi / 180)
    xi_bounds = [-(k_z_max / k0) ** 2 + ((c0 / c_bounds[1]) ** 2 - 1), ((c0 / c_bounds[0]) ** 2 - 1)]
    dxs_m = np.concatenate((
        #[0.0005],
        #[0.001],
        [0.01],
        np.linspace(0.1, 1, 10),
        np.linspace(2, 10, 9),
        np.linspace(20, 100, 9),
    ))
    dzs_m = np.concatenate((
        #[0.00001],
        [0.0001],
        np.linspace(0.001, 0.01, 10),
        np.linspace(0.02, 0.1, 9),
        np.linspace(0.2, 1, 9),
    ))
    errors = precision_step(k0, xi_bounds, k_z_max, dxs_m, dzs_m, pade_order, z_order, False)
    cur_best_dx = 1e-16
    cur_best_dz = 1e-16
    for dx_i, dx in enumerate(dxs_m):
        for dz_i, dz in enumerate(dzs_m):
            err = errors[dx_i, dz_i]
            if x_max_m / dx * err < prec:
                if dx * dz > cur_best_dx * cur_best_dz:
                    cur_best_dx, cur_best_dz = dx, dz

    if return_meta:
        return cur_best_dx, cur_best_dz, c0, xi_bounds
    else:
        return cur_best_dx, cur_best_dz
