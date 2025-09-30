import math as fm
import pickle

import numpy as np

from pywaveprop.propagators._utils import pade_propagator_coefs


def second_difference_disp_rel(k_z: complex, dz: float, z=0):
    return np.exp(1j*k_z*z) * (np.exp(-1j*k_z*dz) - 2 + np.exp(1j*k_z*dz))


def fourth_difference_disp_rel(k_z: complex, dz: float, z=0):
    return np.exp(1j*k_z*z) * (np.exp(-1j*k_z*dz) - 2 + np.exp(1j*k_z*dz))**2


def second_order_error_kz(k_z: float, dz: float):
    d = 1 / dz**2 * second_difference_disp_rel(k_z, dz)
    return abs(d - (-k_z**2))


def fourth_order_error_kz(k_z: float, dz: float):
    d = 1 / dz**2 * (second_difference_disp_rel(k_z, dz) - 1/12 * fourth_difference_disp_rel(k_z, dz))
    return abs(d - (-k_z**2))


def rational_approx_error(beta_dx: float, xi: np.array, rational_coefs, c0=1.0+0j):
    product = c0
    xi = xi + 0j
    for a_i, b_i in rational_coefs:
        product *= (1 + a_i * xi) / (1 + b_i * xi)

    ex = np.exp(1j * beta_dx * (np.sqrt(1.0 + xi) - 1.0))
    return abs(ex - product).real


def _decreasing_func_binary_search(f, x_min, x_max, val, rel_prec):
    x_mid = (x_min + x_max) / 2
    f_mid = f(x_mid)
    if abs(f_mid - val) / val < rel_prec:
        return x_mid
    if f_mid > val:
        return _decreasing_func_binary_search(f, x_mid, x_max, val, rel_prec)
    else:
        return _decreasing_func_binary_search(f, x_min, x_mid, val, rel_prec)


def _increasing_func_binary_search(f, x_min, x_max, val, rel_prec):
    x_mid = (x_min + x_max) / 2
    f_mid = f(x_mid)
    if abs(f_mid - val) / val < rel_prec:
        return x_mid
    if f_mid > val:
        return _increasing_func_binary_search(f, x_min, x_mid, val, rel_prec)
    else:
        return _increasing_func_binary_search(f, x_mid, x_max, val, rel_prec)


def func_binary_search(f, x_min, x_max, val, rel_prec):
    f_min, f_max = f(x_min), f(x_max)
    if f_min > f_max:
        if val > f_min or val < f_max:
            return fm.nan
        return _decreasing_func_binary_search(f, x_min, x_max, val, rel_prec)
    else:
        if val > f_max or val < f_min:
            return fm.nan
        return _increasing_func_binary_search(f, x_min, x_max, val, rel_prec)


class _OptTable:

    def __init__(self, order=(7, 8)):
        self.order = order
        self.t_betas = np.concatenate((
            [0.01, 0.05],
            np.linspace(0.1, 1, 10),
            np.linspace(2, 10, 9),
            np.linspace(20, 100, 9),
            np.linspace(200, 1000, 9),
            np.linspace(2000, 10000, 9),
            np.linspace(20000, 50000, 4)
        ))
        self.precs = np.array([1E-10, 1E-9, 1E-8, 1E-7, 1E-6, 1E-5,])
        self.xi_a = np.empty(shape=(len(self.t_betas), len(self.precs)), dtype=float)
        self.xi_b = np.empty(shape=(len(self.t_betas), len(self.precs)), dtype=float)

    def compute(self):
        for i_t_beta, t_beta in enumerate(self.t_betas):
                coefs, _ = pade_propagator_coefs(pade_order=self.order, beta=t_beta, dx=1)
                for i_prec, prec in enumerate(self.precs):
                    self.xi_a[i_t_beta, i_prec] = func_binary_search(
                        lambda xi: rational_approx_error(t_beta, xi, coefs), -1, 0, prec, 1e-3)
                    self.xi_b[i_t_beta, i_prec] = func_binary_search(
                        lambda xi: rational_approx_error(t_beta, xi, coefs), 0, 10, prec, 1e-3)
                    if fm.isnan(self.xi_b[i_t_beta, i_prec]):
                        self.xi_b[i_t_beta, i_prec] = 10
    @classmethod
    def load(cls):
        import os
        if os.path.exists('grid_optimizer.dump'):
            with open('grid_optimizer.dump', 'rb') as f:
                return pickle.load(f)
        else:
            new_optimizer = _OptTable()
            new_optimizer.compute()
            new_optimizer.save()
            return new_optimizer

    def save(self):
        with open('grid_optimizer.dump', 'wb') as f:
            pickle.dump(self, f)

    def get_optimal(self, kz_max, k_min, k_max, required_prec, dx_max=None, dz_max=None) -> (float, float, float): # beta, dx, dz
        dz_max = dz_max if dz_max else 5.0
        a = k_min**2 - kz_max**2
        b = k_max**2
        res = fm.nan, 0.0, 0.0
        for i_t_beta, t_beta in enumerate(self.t_betas):
            for i_prec, prec in enumerate(self.precs):
                r = fm.sqrt(a / (self.xi_a[i_t_beta, i_prec] + 1))
                l = fm.sqrt(b / (self.xi_b[i_t_beta, i_prec] + 1))
                if l > r:
                    continue
                beta = l
                dx = t_beta / beta
                if dx_max and dx > dx_max:
                    continue
                if prec / dx > required_prec:
                    continue

                xi_min = a / beta ** 2 - 1
                prop_op_d_xi = t_beta / (2 * fm.sqrt(1 + xi_min))
                dz_prec = t_beta ** 2 * prec / prop_op_d_xi
                if fourth_order_error_kz(kz_max, dz_max) < dz_prec:
                    dz = dz_max
                else:
                    dz = func_binary_search(lambda a_dz: fourth_order_error_kz(kz_max, a_dz), 0.0001, dz_max,
                                            dz_prec, 1e-3)

                if dx*dz > res[1] * res[2]:
                    res = (beta, dx, dz)

        beta, dx, dz = res
        return beta, dx, dz


table = _OptTable.load()


def get_optimal_grid(kz_max, k_min, k_max, required_prec, dx_max=None, dz_max=None) -> (float, float, float): # beta, dx, dz
    return table.get_optimal(kz_max, k_min, k_max, required_prec, dx_max, dz_max)
