import numpy as np
import cmath as cm
import math as fm
import propagators.dispersion_relations as disp_rels
import mpmath
import propagators._utils as utils


def opt_coefs_to_grids(coefs_arr):
    dx = coefs_arr[0]
    dz = coefs_arr[1]
    return dx, dz


def opt_coefs_to_coefs(coefs_arr, order):
    n = order[0]
    m = order[1]
    num_coefs = np.array([coefs_arr[2 + 2 * i] + 1j * coefs_arr[2 + 2 * i + 1] for i in range(0, n)])
    den_coefs = np.array([coefs_arr[2+ 2 * n + 2 * i] + 1j * coefs_arr[2 + 2 * n + 2 * i + 1] for i in range(0, m)])
    return num_coefs, den_coefs


def fit_func(coefs_arr):
    dx, dz = opt_coefs_to_grids(coefs_arr)
    return 1 / (dx * dz)


def constraint_ga(coefs_arr, order, theta_max_degrees):
    k0 = 2 * fm.pi
    dx, dz = opt_coefs_to_grids(coefs_arr)
    num_coefs, den_coefs = opt_coefs_to_coefs(coefs_arr, order)
    err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs, k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                        round(theta_max_degrees) * 5) / dx
    return err


def constraint_pade_joined_order(coefs_arr, order, theta_max_degrees):
    k0 = 2 * fm.pi
    dx, dz = opt_coefs_to_grids(coefs_arr)

    def diff2(s):
        return mpmath.acosh(1 + (k0 * dz) ** 2 * s / 2) ** 2 / (k0 * dz) ** 2
    pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=diff2, k0=2*cm.pi, dx=dx)
    num_coefs = np.array([a[0] for a in pade_coefs])
    den_coefs = np.array([a[1] for a in pade_coefs])
    err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs, k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                        round(theta_max_degrees) * 5) / dx
    return err


def constraint_pade_2nd_order(coefs_arr, order, theta_max_degrees):
    k0 = 2 * fm.pi
    dx, dz = opt_coefs_to_grids(coefs_arr)
    pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2*cm.pi, dx=dx)
    num_coefs = np.array([a[0] for a in pade_coefs])
    den_coefs = np.array([a[1] for a in pade_coefs])
    err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs, k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                        round(theta_max_degrees) * 5) / dx
    return err