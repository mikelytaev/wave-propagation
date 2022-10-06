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
theta_max_degrees = 22
order = (6, 7)

print(k0*fm.sin(theta_max_degrees * fm.pi / 180))

dx, dz = 50, 0.25

bounds_ga = [(-50, 50)] * (order[0] + order[1]) * 2

arr = []
def calc(mutation, recombination, strategy, popsize, max_evals=1e6):
    global arr
    arr = []
    def append(f):
        global arr
        arr += [f]

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
        workers=-1,
        callback=lambda xk, convergence: append(opt_utils.fit_func_ga(xk, dx, dz, order, theta_max_degrees))
    )

    print(result_ga)
    return arr


def print_arr(arr):
    print(str(arr[99]) + " " + str(arr[199]) + " " + str(arr[499]) + " " + str(arr[999]) + " " + str(arr[1999]) +
          " " + str(arr[4999]) + " " + str(arr[9999]))


max_evals = 3e6
arr1 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='randtobest1exp', popsize=10, max_evals=max_evals)
# arr2 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='randtobest1exp', popsize=15, max_evals=max_evals)
# arr3 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='randtobest1exp', popsize=20, max_evals=max_evals)
# arr4 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='randtobest1exp', popsize=25, max_evals=max_evals)
arr5 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='randtobest1exp', popsize=50, max_evals=max_evals)
arr6 = calc(mutation=(0.5, 1.0), recombination=1.0, strategy='randtobest1exp', popsize=100, max_evals=max_evals)

plt.figure(figsize=(6, 3.2))
plt.plot(np.linspace(0, max_evals, len(arr1)), np.power(10, arr1), label="10")
# plt.plot(np.linspace(0, max_evals, len(arr2)), arr2, label="15")
# plt.plot(np.linspace(0, max_evals, len(arr3)), arr3, label="20")
# plt.plot(np.linspace(0, max_evals, len(arr4)), arr4, label="25")
plt.plot(np.linspace(0, max_evals, len(arr5)), np.power(10, arr5), label="50")
plt.plot(np.linspace(0, max_evals, len(arr6)), np.power(10, arr6), label="100")

plt.legend()
plt.grid(True)
plt.xlim([0, max_evals])
#plt.ylim([-6, -1])
plt.yscale('log')
plt.xlabel('Number of evaluations')
plt.ylabel('Numerical dispersion error')
plt.tight_layout()
plt.savefig('uncond_convergence_rate.eps')
#plt.show()