import numpy as np

import pyximport
import cmath as cm
import math as fm
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
from scipy.optimize import differential_evolution

import examples.optimization.evol.opt_utils as opt_utils


k0 = 2*cm.pi
theta_max_degrees = 22
order = (6, 7)

print(k0*fm.sin(theta_max_degrees * fm.pi / 180))

dx, dz = 50, 0.25

bounds_ga = [(-50, 50)] * (order[0] + order[1]) * 2

arr = []
def calc(mutation, recombination, strategy):
    global arr
    arr = []
    def append(f):
        global arr
        arr += [f]

    result_ga = differential_evolution(
        func=opt_utils.fit_func_ga,
        args=(dx, dz, order, theta_max_degrees),
        bounds=bounds_ga,
        popsize=50,
        disp=True,
        mutation=mutation,
        recombination=recombination,
        strategy=strategy,
        tol=1e-9,
        maxiter=10000,
        polish=False,
        workers=-1,
        callback=lambda xk, convergence: append(opt_utils.fit_func_ga(xk, dx, dz, order, theta_max_degrees))
    )

    print(result_ga)
    return arr


def print_arr(arr):
    print(str(arr[99]) + " " + str(arr[199]) + " " + str(arr[499]) + " " + str(arr[999]) + " " + str(arr[1999]) +
          " " + str(arr[4999]) + " " + str(arr[9999]))

arr1 = calc(mutation=(0.0, 1.9999999), recombination=1.0, strategy='currenttobest1exp')
arr2 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='currenttobest1exp')
arr3 = calc(mutation=(0.5, 1.0), recombination=0.7, strategy='currenttobest1exp')
arr4 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='best1bin')
arr5 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='best2exp')
arr6 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='rand2exp')
arr7 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='best1exp')
arr8 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='rand1exp')
arr9 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='randtobest1exp')
arr10 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='randtobest1bin')
arr11 = calc(mutation=(0.5, 1.0), recombination=0.9, strategy='randtobest1bin')
arr12 = calc(mutation=(0.5, 1.0), recombination=0.7, strategy='randtobest1bin')
arr13 = calc(mutation=0.5, recombination=1.0, strategy='randtobest1bin')
arr14 = calc(mutation=0.3, recombination=1.0, strategy='randtobest1bin')
arr15 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='currenttobest1bin')
