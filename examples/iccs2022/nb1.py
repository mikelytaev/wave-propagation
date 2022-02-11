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
from examples.chebyshev_pade.cheb_pade_coefs import *


k0 = 2*cm.pi
theta_max_degrees = 22
order = (6, 7)

print(k0*fm.sin(theta_max_degrees * fm.pi / 180))

dx, dz = 50, 0.25

bounds_ga = [(-100, 100)] * (order[0] + order[1]) * 2

result_ga = differential_evolution(
    func=opt_utils.fit_func_ga,
    args=(dx, dz, order, theta_max_degrees),
    bounds=bounds_ga,
    popsize=20,
    disp=True,
    mutation=(0.0, 1.9999999),
    recombination=1.0,
    strategy='randtobest1exp',
    tol=1e-9,
    maxiter=30000,
    polish=False,
    workers=-1,
    #callback=lambda xk, convergence: print(xk)
)
print(result_ga)

ga_coefs_num, ga_coefs_den = opt_utils.opt_coefs_to_coefs_ga(result_ga.x, order)


def k_x_angle(dx, dz, num_coefs, den_coefs, kz_arr):
    return np.array([disp_rels.discrete_k_x(k0, dx, dz, num_coefs, den_coefs, kz) for kz in kz_arr])


pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2*cm.pi, dx=dx)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])

ratinterp_coefs, a0 = cheb_pade_coefs(dx, (order[1], order[1]), fm.sin(theta_max_degrees*fm.pi/180)**2, 'ratinterp')
ratinterp_coefs_num = np.array([a[0] for a in ratinterp_coefs])
ratinterp_coefs_den = np.array([b[1] for b in ratinterp_coefs])

angles = np.linspace(0, theta_max_degrees*1.5, 1000)
kz_arr = k0*np.sin(angles*fm.pi/180)
k_x_r = np.sqrt(k0**2 - kz_arr**2)
k_x_ga = k_x_angle(dx, dz, ga_coefs_num, ga_coefs_den, kz_arr)
k_x_pade = k_x_angle(dx, dz, pade_coefs_num, pade_coefs_den, kz_arr)
k_x_ratinterp = k_x_angle(dx, dz, ratinterp_coefs_num, ratinterp_coefs_den, kz_arr)

k_x_ga_error = np.abs(k_x_ga - k_x_r)
k_x_pade_error = np.abs(k_x_pade - k_x_r)
k_x_ratinterp_error = np.abs(k_x_ratinterp - k_x_r)

plt.figure(figsize=(6, 3.2))
plt.plot(angles, k_x_ga_error, label='ga')
plt.plot(angles, k_x_pade_error, label='Pade')
plt.plot(angles, k_x_ratinterp_error, label='ratinterp')
plt.xlabel('Angle (degrees)')
plt.ylabel('k_x abs. error')
plt.xlim([angles[0], angles[-1]])
#plt.ylim([1e-10, 1e-1])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 3.2))
# plt.plot(angles, (np.real(k_x_1)), label='opt real')
# plt.plot(angles, (np.real(k_x_2)), label='Pade real')
plt.plot(kz_arr, (np.imag(k_x_1)), label='opt imag')
plt.plot(kz_arr, (np.imag(k_x_3)), label='Joined Pade imag')
plt.xlabel('k_z')
plt.ylabel('k_x abs. error')
plt.xlim([kz_arr[0], kz_arr[-1]])
#plt.ylim([1e-10, 1e-1])
#plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
