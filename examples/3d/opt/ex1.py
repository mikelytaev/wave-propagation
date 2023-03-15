import cmath as cm

import math as fm
import numpy as np

from examples.jcs.d2_error import second_order_error_kz, fourth_order_error_kz
from propagators._utils import pade_propagator_coefs
import propagators._utils as utils

from examples.chebyshev_pade.cheb_pade_coefs import *


def precision_step(k0, k0sh, theta_max_degrees, dxs_m, dzs_m, pade_order, z_order=4, adi=False, ratinterp=False):
    res = np.zeros((len(dxs_m), len(dzs_m)))
    n = 100
    alpha = (k0/k0sh)**2-1
    z = 1 if z_order == 4 else 0
    t_grid = np.linspace(0, 2*fm.pi, n)
    r_grid = np.linspace(0, k0*fm.sin(fm.radians(theta_max_degrees)), n)
    k_y_grid = np.sin(t_grid) * r_grid
    k_z_grid = np.cos(t_grid) * r_grid
    xis = -(k_y_grid/k0sh)**2 - (k_z_grid/k0sh)**2 + alpha
    for dx_i, dx_m in enumerate(dxs_m):
        if ratinterp:
            pade_coefs, c0 = cheb_pade_coefs(k0sh, dx_m, pade_order, (np.min(xis), np.max(xis)), "ratinterp")
        else:
            pade_coefs, c0 = utils.pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, k0=k0sh, dx=dx_m, a0=0)
        for dz_i, dz_m in enumerate(dzs_m):
            xi_y = -(2*np.sin(k_y_grid*dz_m/2) / (k0sh*dz_m))**2 - z*4/3*np.sin(k_y_grid*dz_m/2)**4 / (k0sh*dz_m)**2 + alpha/2
            xi_z = -(2*np.sin(k_z_grid*dz_m/2) / (k0sh*dz_m))**2 - z*4/3*np.sin(k_z_grid*dz_m/2)**4 / (k0sh*dz_m)**2 + alpha/2
            xi_grid = xi_y + xi_z
            t = c0
            for (a, b) in pade_coefs:
                if adi:
                    t *= ((1 + a * xi_y) * (1 + a * xi_z)) / ((1 + b * xi_y) * (1 + b * xi_z))
                else:
                    t *= (1 + a * xi_grid) / (1 + b * xi_grid)
            discrete_k_x = k0sh - 1j/dx_m * np.log(t)
            k_x = np.sqrt(k0**2 - k_y_grid**2 - k_z_grid**2)
            res[dx_i, dz_i] = np.max(np.abs(k_x - discrete_k_x))
    return res


def get_optimal(*, freq_hz, x_max_m, prec, theta_max_degrees, pade_order, z_order=4, shift=False, return_meta=False,
                dxs_m=None, adi=False, ratinterp=False):
    k0 = 2 * fm.pi * freq_hz / 3E8
    dxs_m = dxs_m if dxs_m else np.concatenate((
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
        np.linspace(2, 5, 4),
    ))
    c0 = fm.sqrt(2 / (1 + fm.cos(fm.radians(theta_max_degrees)) ** 2)) * 3e8
    k0sh = 2 * fm.pi * freq_hz / c0 if shift else k0
    errors = precision_step(k0, k0sh, theta_max_degrees, dxs_m, dzs_m, pade_order, z_order, adi, ratinterp)
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


theta_max_degrees = 30
prec = 1e-2
x_max_m = 1000
pade_order = (6, 7)

dx_pade, dz_pade = get_optimal(
    freq_hz=300e6,
    x_max_m=x_max_m,
    prec=prec,
    theta_max_degrees=theta_max_degrees,
    pade_order=pade_order,
)

print(f'dx={dx_pade}; dz={dz_pade}')

dx_pade_s, dz_pade_s = get_optimal(
    freq_hz=300e6,
    x_max_m=x_max_m,
    prec=prec,
    theta_max_degrees=theta_max_degrees,
    pade_order=pade_order,
    shift=True
)

print(f'dx={dx_pade_s}; dz={dz_pade_s}')

dx_ratinterp, dz_ratinterp = get_optimal(
    freq_hz=300e6,
    x_max_m=x_max_m,
    prec=prec,
    theta_max_degrees=theta_max_degrees,
    pade_order=pade_order,
    ratinterp=True,
    dxs_m=[0.5*dx_pade, dx_pade, 1.5*dx_pade, 2*dx_pade]
)

print(f'dx={dx_ratinterp}; dz={dz_ratinterp}')

dx_ratinterp_s, dz_ratinterp_s = get_optimal(
    freq_hz=300e6,
    x_max_m=x_max_m,
    prec=prec,
    theta_max_degrees=theta_max_degrees,
    pade_order=pade_order,
    ratinterp=True,
    shift=True,
    dxs_m=[0.5*dx_pade_s, dx_pade_s, 1.5*dx_pade_s, 2*dx_pade_s]
)

print(f'dx={dx_ratinterp_s}; dz={dz_ratinterp_s}')