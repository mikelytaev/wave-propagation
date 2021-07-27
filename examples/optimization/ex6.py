import numpy as np

from propagators._utils import *
from scipy.optimize import differential_evolution, minimize, basinhopping
import matplotlib.pyplot as plt


def k_x_abs_error_point(dx, dz, k0, coefs, theta):
    kz = k0 * cm.sin(fm.radians(theta))
    dk_x = discrete_k_x(k=k0, dx=dx, pade_coefs=coefs, dz=dz, kz=kz)
    return abs(dk_x - k_x(k0, kz))


def k_x_abs_error_range(dx, dz, k0, coefs, theta_max):
    errors = [k_x_abs_error_point(dx, dz, k0, coefs, theta) for theta in np.linspace(0, theta_max, 20)]
    return max(errors)


def opt_coefs_to_coefs(coefs_arr):
    complex_len = round(len(coefs_arr) / 2)
    n = round(complex_len / 2)
    m = complex_len - n
    num_coefs = [coefs_arr[2 * i] + 1j * coefs_arr[2 * i + 1] for i in range(0, n)]
    den_coefs = [coefs_arr[2 * n + 2 * i] + 1j * coefs_arr[2 * n + 2 * i + 1] for i in range(0, m)]
    coefs = list(zip_longest(num_coefs, den_coefs, fillvalue=0.0))
    return coefs


def coefs_to_opt_coefs(coefs):
    co = []
    for c in pade_coefs:
        co += [c[0].real, c[0].imag]
    for c in pade_coefs:
        co += [c[1].real, c[1].imag]
    return co


dx = 100


def fit_func(coefs_arr):
    coefs = opt_coefs_to_coefs(coefs_arr)
    err = k_x_abs_error_range(dx, 0.00000001, 2*cm.pi, coefs, 10)
    #print(err)
    return err


order = (4, 4)
pade_coefs = pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2*cm.pi, dx=dx)
co = coefs_to_opt_coefs(pade_coefs)
print(fit_func(co))

bounds = [(-100, 100)] * (order[0] + order[1]) * 2


def print_fun(x, f, accepted):
    print("at minimum %.4f accepted %d" % (f, int(accepted)))


result = differential_evolution(fit_func, bounds, popsize=10, disp=True, strategy='currenttobest1bin', tol=1e-3, maxiter=100000)
#result = minimize(fun=fit_func, x0=np.zeros(12), bounds=bounds, tol=1e-11, method="CG")
#result = minimize(fun=fit_func, x0=np.zeros(12), bounds=bounds, tol=1e-17, method="COBYLA", options={"maxiter": 10000000})
#result = basinhopping(func=fit_func, x0=np.zeros(12), disp=True)
print(result)
coefs = opt_coefs_to_coefs(result.x)


def k_x_angle(*, k_z, dx, coefs, dz, z_order, alpha):
    k0 = 2 * cm.pi

    if z_order > 4:
        z_order = 2

        def diff2(s):
            return mpmath.acosh(1 + (k0 * dz) ** 2 * s / 2) ** 2 / (k0 * dz) ** 2
    else:
        def diff2(s):
            return s

        return np.array([discrete_k_x(k=k0, dx=dx, dz=dz, pade_coefs=coefs, kz=kz, order=z_order) for kz in k_z])


k0 = 2*cm.pi
k_z = np.linspace(0, k0/2, 300)
angles = np.linspace(0, cm.asin(k_z[-1] / k0)*180/cm.pi, 300)
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in k_z])
k_x_1 = k_x_angle(k_z=k_z, dx=dx, coefs=coefs, dz=0.00000001, z_order=2, alpha=0)
k_x_2 = k_x_angle(k_z=k_z, dx=dx, coefs=pade_coefs, dz=0.00000001, z_order=2, alpha=0)

plt.figure(figsize=(6, 3.2))
plt.plot(angles, (np.abs(k_x_1 - k_x_r)), label='opt')
plt.plot(angles, (np.abs(k_x_2 - k_x_r)), label='Pade')
plt.xlabel('Propagation angle, degrees')
plt.ylabel('k_x abs. error')
plt.xlim([0, 15])
#plt.ylim([1e-10, 1e-1])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
