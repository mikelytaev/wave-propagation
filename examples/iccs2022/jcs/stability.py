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
from itertools import zip_longest


k0 = 2*cm.pi

arr = []
def de(dx, dz, bounds, theta_max_degrees, order, mutation=(0.5, 1), recombination=1.0, strategy='randtobest1exp', popsize=15, max_evals=1e6):
    bounds_ga = [bounds] * (order[0] + order[1]) * 2

    def stability_constr(x):
        #print('constr')
        abss = []
        for v in np.linspace(0, 1, 50):
            num_coefs, den_coefs = opt_utils.opt_coefs_to_coefs_ga(x, order)
            zipped = list(zip_longest(num_coefs, den_coefs, fillvalue=0.0j))
            l = [((k0*dz)**2 - 4*a*v) / ((k0*dz)**2 - 4*b*v) for a, b in zipped]
            abss += [abs(np.prod(l))]
        return max(abss)

    print(round(max_evals / (popsize * len(bounds_ga)) - 1))
    result_ga = differential_evolution(
        func=opt_utils.fit_func_ga,
        args=(dx, dz, order, theta_max_degrees),
        constraints=(NonlinearConstraint(stability_constr, 0, 1)),
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

    return result_ga



def print_arr(arr):
    print(str(arr[99]) + " " + str(arr[199]) + " " + str(arr[499]) + " " + str(arr[999]) + " " + str(arr[1999]) +
          " " + str(arr[4999]) + " " + str(arr[9999]))


def k_x_angle(dx, dz, num_coefs, den_coefs, kz_arr):
    return np.array([disp_rels.discrete_k_x(k0, dx, dz, num_coefs, den_coefs, kz) for kz in kz_arr])


max_evals = 0.3e6
order = (6, 7)
dx = 50
dz = 0.25
res = de(dx=dx, dz=dz, theta_max_degrees=20, order=order, bounds=(-70, 70), popsize=10, max_evals=max_evals)
print(res)
kz_arr = k0*np.linspace(0, 10, 1000)
ga_coefs_num, ga_coefs_den = opt_utils.opt_coefs_to_coefs_ga(res.x, order)
k_x_ga = k_x_angle(dx, dz, ga_coefs_num, ga_coefs_den, kz_arr)

plt.figure(figsize=(6, 3.2))
plt.plot(kz_arr, (np.imag(k_x_ga)), label='opt imag')
plt.xlabel('k_z')
plt.ylabel('k_x abs. error')
plt.xlim([kz_arr[0], kz_arr[-1]])
#plt.ylim([1e-10, 1e-1])
#plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
