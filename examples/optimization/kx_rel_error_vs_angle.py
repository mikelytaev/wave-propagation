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
    if m == 1:
        return np.array([discrete_k_x(k=k0, dx=dx, dz=dz, pade_coefs=coefs, kz=kz, order=z_order) for kz in k_z])
    else:
        return np.array([discrete_k_x2(k=k0, dx=dx, dz=dz, pade_coefs=coefs, kz=kz, order=z_order) for kz in k_z])

k0 = 2*cm.pi
k_z = np.linspace(0, k0/2, 300)
angles = np.linspace(0, cm.asin(k_z[-1] / k0)*180/cm.pi, 300)
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in k_z])
k_x_1 = k_x(k_z=k_z, dx=300, pade_order=(1, 1), dz=0.0000001, z_order=2, alpha=0)
k_x_2 = k_x(k_z=k_z, dx=300, pade_order=(3, 4), dz=0.0000001, z_order=2, alpha=0)
k_x_3 = k_x(k_z=k_z, dx=300, pade_order=(7, 8), dz=0.0000001, z_order=2, alpha=0)
k_x_4 = k_x(k_z=k_z, dx=300, pade_order=(10, 11), dz=0.0000001, z_order=2, alpha=0)

plt.figure(figsize=(6, 3.2))
plt.plot(angles, (np.abs(k_x_1 - k_x_r) / np.abs(k_x_r)), label='Pade-[1/1]')
plt.plot(angles, (np.abs(k_x_2 - k_x_r) / np.abs(k_x_r)), label='Pade-[3/4]')
plt.plot(angles, (np.abs(k_x_3 - k_x_r) / np.abs(k_x_r)), label='Pade-[7/8]')
plt.plot(angles, (np.abs(k_x_4 - k_x_r) / np.abs(k_x_r)), label='Pade-[10/11]')
plt.xlabel('Propagation angle, degrees')
plt.ylabel('k_x relative error')
plt.xlim([0, 15])
plt.ylim([1e-10, 1e-1])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
