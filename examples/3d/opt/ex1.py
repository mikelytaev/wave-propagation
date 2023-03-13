import cmath as cm

import math as fm
import numpy as np

from examples.jcs.d2_error import second_order_error_kz, fourth_order_error_kz
from propagators._utils import pade_propagator_coefs
import propagators._utils as utils


def precision_step(k0, theta_max_degrees, dxs_m, dzs_m, pade_order, alpha=0):
    res = np.zeros((len(dxs_m), len(dzs_m)))
    n = 100
    t_grid = np.linspace(0, 2*fm.pi, n)
    r_grid = np.linspace(0, k0*fm.sin(fm.radians(theta_max_degrees)), n)
    k_y_grid = np.sin(t_grid) * r_grid
    k_z_grid = np.cos(t_grid) * r_grid
    xi_grid = -(k_y_grid/k0)**2 - (k_z_grid/k0)**2 + alpha
    print(f"{np.min(xi_grid)} {np.max(xi_grid)}")
    for dx_i, dx_m in enumerate(dxs_m):
        pade_coefs, c0 = utils.pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, k0=k0, dx=dx_m, a0=0)
        for dz_i, dz_m in enumerate(dzs_m):
            #xi_grid = -(2*np.sin(k_y_grid*dz_m/2) / (k0*dz_m))**2 - (2*np.sin(k_z_grid*dz_m/2) / (k0*dz_m))**2 + alpha
            t = c0
            for (a, b) in pade_coefs:
                t *= (1 + a * xi_grid) / (1 + b * xi_grid)
            discrete_k_x = k0*fm.sqrt(1-alpha) - 1j/dx_m * np.log(t)
            k_x = np.sqrt(k0**2 - k_y_grid**2 - k_z_grid**2)
            res[dx_i, dz_i] = np.max(np.abs(k_x - discrete_k_x))
    return res


def get_optimal(*, freq_hz, x_max_m, prec, theta_max_degrees, pade_order, z_order=4, shift=False, return_meta=False):
    alpha = 0.016 if shift else 0
    k0 = 2 * fm.pi * freq_hz / 3E8
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
    errors = precision_step(k0, theta_max_degrees, dxs_m, dzs_m, pade_order, alpha=alpha)
    print(f"{np.min(errors)}")
    cur_best_dx = 1e-16
    cur_best_dz = 1e-16
    for dx_i, dx in enumerate(dxs_m):
        for dz_i, dz in enumerate(dzs_m):
            err = errors[dx_i, dz_i]
            if x_max_m / dx * err < prec:
                if dx * dz > cur_best_dx * cur_best_dz:
                    cur_best_dx, cur_best_dz = dx, dz

    if return_meta:
        return cur_best_dx, cur_best_dz, xi_bounds
    else:
        return cur_best_dx, cur_best_dz


dx, dz = get_optimal(
    freq_hz=300e6,
    x_max_m=100,
    prec=5e-2,
    theta_max_degrees=30,
    pade_order=(6, 7),
)

print(f'dx={dx}; dz={dz}')

dx, dz = get_optimal(
    freq_hz=300e6,
    x_max_m=100,
    prec=5e-2,
    theta_max_degrees=30,
    pade_order=(6, 7),
    shift=True
)

print(f'dx={dx}; dz={dz}')