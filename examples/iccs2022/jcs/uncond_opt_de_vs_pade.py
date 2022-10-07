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

arr = []
def de_error(dx, dz, bounds, theta_max_degrees, order, mutation=(0.5, 1), recombination=1.0, strategy='randtobest1exp', popsize=15, max_evals=1e6):
    bounds_ga = [bounds] * (order[0] + order[1]) * 2

    print(round(max_evals / (popsize * len(bounds_ga)) - 1))
    result_ga = differential_evolution(
        func=opt_utils.fit_func_ga,
        args=(dx, dz, order, theta_max_degrees),
        bounds=bounds_ga,
        popsize=popsize,
        disp=True,
        mutation=mutation,
        recombination=recombination,
        strategy=strategy,
        tol=1e-9,
        maxiter=round(max_evals / (popsize * len(bounds_ga)) - 1),
        polish=False,
        workers=-1
    )

    print(result_ga)

    return fm.pow(10, result_ga.fun)


def pade_2nd_order_error(order, dx, dz, theta_max_degrees):
    pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2*cm.pi, dx=dx)
    num_coefs = np.array([a[0] for a in pade_coefs])
    den_coefs = np.array([a[1] for a in pade_coefs])
    err = disp_rels.k_x_abs_error_range(2 * cm.pi, dx, dz, num_coefs, den_coefs, k0 * fm.sin(theta_max_degrees * fm.pi / 180),
                                        round(theta_max_degrees) * 20) / dx / k0
    return err


def print_arr(arr):
    print(str(arr[99]) + " " + str(arr[199]) + " " + str(arr[499]) + " " + str(arr[999]) + " " + str(arr[1999]) +
          " " + str(arr[4999]) + " " + str(arr[9999]))


max_evals = 0.3e6
# de_err = de_error(dx=1000.0, dz=2.6, theta_max_degrees=3, order=(6, 7), bounds=(-500, 500), popsize=10, max_evals=max_evals)
# pade_error = pade_2nd_order_error(dx=1000.0, dz=2.6, theta_max_degrees=3, order=(6, 7))
# print("DE error = " + str(de_err) + " Pade error = " + str(pade_error))
# print('gain ' + str(pade_error / de_err))

# de_err = de_error(dx=70.0, dz=0.5, theta_max_degrees=10, order=(6, 7), bounds=(-70, 70), popsize=10, max_evals=max_evals)
# pade_error = pade_2nd_order_error(dx=70.0, dz=0.5, theta_max_degrees=10, order=(6, 7))
# print("DE error = " + str(de_err) + " Pade error = " + str(pade_error))
# print('gain ' + str(pade_error / de_err))

# de_err = de_error(dx=50.0, dz=0.25, theta_max_degrees=22, order=(6, 7), bounds=(-50, 50), popsize=10, max_evals=max_evals)
# pade_error = pade_2nd_order_error(dx=50.0, dz=0.25, theta_max_degrees=22, order=(6, 7))
# print("DE error = " + str(de_err) + " Pade error = " + str(pade_error))
# print('gain ' + str(pade_error / de_err))

# de_err = de_error(dx=20.0, dz=0.2, theta_max_degrees=30, order=(6, 7), bounds=(-20, 20), popsize=10, max_evals=max_evals)
# pade_error = pade_2nd_order_error(dx=20.0, dz=0.2, theta_max_degrees=30, order=(6, 7))
# print("DE error = " + str(de_err) + " Pade error = " + str(pade_error))
# print('gain ' + str(pade_error / de_err))

# de_err = de_error(dx=10.0, dz=0.1, theta_max_degrees=45, order=(6, 7), bounds=(-20, 20), popsize=10, max_evals=max_evals)
# pade_error = pade_2nd_order_error(dx=10.0, dz=0.1, theta_max_degrees=45, order=(6, 7))
# print("DE error = " + str(de_err) + " Pade error = " + str(pade_error))
# print('gain ' + str(pade_error / de_err))

# de_err = de_error(dx=2.0, dz=0.1, theta_max_degrees=60, order=(6, 7), bounds=(-10, 10), popsize=10, max_evals=max_evals)
# pade_error = pade_2nd_order_error(dx=2.0, dz=0.1, theta_max_degrees=60, order=(6, 7))
# print("DE error = " + str(de_err) + " Pade error = " + str(pade_error))
# print('gain ' + str(pade_error / de_err))

# de_err = de_error(dx=1.0, dz=0.2, theta_max_degrees=80, order=(6, 7), bounds=(-2, 2), popsize=10, max_evals=max_evals)
# pade_error = pade_2nd_order_error(dx=1.0, dz=0.2, theta_max_degrees=80, order=(6, 7))
# print("DE error = " + str(de_err) + " Pade error = " + str(pade_error))
# print('gain ' + str(pade_error / de_err))
#

de_err = de_error(dx=1.0, dz=0.3, theta_max_degrees=88, order=(6, 7), bounds=(-1.5, 1.5), popsize=10, max_evals=max_evals*3)
pade_error = pade_2nd_order_error(dx=1.0, dz=0.3, theta_max_degrees=80, order=(6, 7))
print("DE error = " + str(de_err) + " Pade error = " + str(pade_error))
print('gain ' + str(pade_error / de_err))
