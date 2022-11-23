import numpy as np
from examples.jcs.d2_error import *
import math as fm
from propagators._utils import *


import matplotlib.pyplot as plt
dz = 20
errors_vs_kz = [fourth_order_error_kz(kz, dz) for kz in np.linspace(-fm.pi*3, fm.pi*3, 2000)]
plt.plot(errors_vs_kz)
plt.grid(True)
plt.show()

kz = 0.25#*fm.pi
z_grid = np.linspace(0, 40, 2000)
errors_vs_dz = [fourth_order_error_kz(kz, dz) for dz in z_grid]
plt.plot(z_grid, errors_vs_dz)
plt.grid(True)
plt.show()


def h_error(dz_wl, k_z_max):
    return fourth_order_error_kz(k_z_max, dz_wl)

def tau_error(xi, dx_wl, dz_wl, pade_coefs):
    k0 = 2 * fm.pi
    return abs(cm.exp(1j*k0*dx_wl*(cm.sqrt(1 + xi) - 1)) - np.prod([(1 + a*xi) / (1 + b*xi) for a, b in pade_coefs]))

def precision_step(xi_bounds, dxs_wl: np.array, dzs_wl: np.array, pade_order, z_order=4):
    k0 = 2 * fm.pi
    for dx_wl in dxs_wl:
        coefs = pade_propagator_coefs(diff2=lambda x: x, k0=k0, dx=dx_wl)
        for dz_wl in dzs_wl:
            pass



    k0 = 2*cm.pi
    res = (None, None, None)
    cur_min = 1e100

    if pade_order:
        pade_orders = [pade_order]
    else:
        pade_orders = [(7, 8), (6, 7), (5, 6), (4, 5), (3, 4), (2, 3), (1, 2), (1, 1)]

    if dx_wl:
        dxs = [dx_wl]
    else:
        dxs = np.concatenate((#np.linspace(0.001, 0.01, 10),
                              #np.linspace(0.02, 0.1, 9),
                              #np.linspace(0.2, 1, 9),
                              #np.linspace(2, 10, 9),
                              np.linspace(20, 100, 9),
                              np.linspace(200, 1000, 9),
                              np.linspace(1100, 1900, 9),
                              np.linspace(2000, 10000, 9)))

    if dz_wl:
        dzs = [dz_wl]
    else:
        dzs = np.concatenate((np.array([0.001, 0.009]),
                              np.array([0.01, 0.09]),
                             np.linspace(0.1, 9, 90)))

    dxs.sort()
    dzs.sort()
    for pade_order in pade_orders:
        for dx_wl in dxs:
            updated = False
            if z_order <= 4:
                coefs = pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, k0=k0, dx=dx_wl, spe=False)
            for dz_wl in dzs:
                if z_order > 4:
                    coefs = pade_propagator_coefs(pade_order=pade_order,
                                                  diff2=lambda s: mpmath.acosh(1 + (k0 * dz_wl) ** 2 * s / 2) ** 2 / (k0 * dz_wl) ** 2,
                                                  k0=k0, dx=dx_wl, spe=False)

                errors = []
                for al in np.linspace(0, max_angle_deg, 20):
                    kz = k0 * cm.sin(al * cm.pi / 180)
                    if z_order <= 4:
                        discrete_kx = discrete_k_x(k0, dx_wl, coefs, dz_wl, kz, order=z_order)
                    else:
                        discrete_kx = discrete_k_x(k0, dx_wl, coefs, dz_wl, kz, order=2)
                    real_kx = cm.sqrt(k0 ** 2 - kz ** 2)
                    errors += [abs(real_kx - discrete_kx) / k0]

                val = pade_order[1] / (dx_wl * dz_wl)
                error = max(errors)

                if error >= threshold * dx_wl / max_distance_wl:
                    break

                if error < threshold * dx_wl / max_distance_wl and val < cur_min:
                    res = (dx_wl, dz_wl, pade_order)
                    cur_min = val
                    updated = True

            if not updated:
                break

    return res