from propagators._utils import *
import numpy as np
import matplotlib.pyplot as plt


def k_x(*, k_z, dx, pade_order, dz, z_order, alpha, m=1):
    f = 300e6
    k0 = 2 * cm.pi * f / 3e8
    mu = 3e8 / f

    if z_order > 4:
        z_order = 2

        def diff2(s):
            return mpmath.acosh(1 + (k0 * dz) ** 2 * s / 2) ** 2 / (k0 * dz) ** 2
    else:
        def diff2(s):
            return s

    coefs = pade_propagator_coefs(pade_order=pade_order, diff2=diff2, k0=k0, dx=dx, alpha=alpha)
    return np.array([discrete_k_x(k=k0, dx=dx, dz=v, pade_coefs=coefs, kz=k_z, order=z_order) for v in dz])

k0 = 2*cm.pi
angle = 3
k_z = k0 * cm.sin(3 / 180 * cm.pi)
k_x_r = cm.sqrt(k0**2 - k_z**2)
dzs = np.linspace(0.01, 10, 1000)
k_x_2 = k_x(k_z=k_z, dx=200, pade_order=(7, 8), dz=dzs, z_order=2, alpha=0)
k_x_2_4 = k_x(k_z=k_z, dx=200, pade_order=(7, 8), dz=dzs, z_order=4, alpha=0)

plt.plot(dzs, (np.abs(k_x_2 - k_x_r)), '--', label='[7/8], 2-d order by z')
plt.plot(dzs, (np.abs(k_x_2_4 - k_x_r)), '--', label='[7/8], 4-d order by z')
plt.xlabel('dz')
#plt.xscale("log")
plt.ylabel('Relative error')
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()