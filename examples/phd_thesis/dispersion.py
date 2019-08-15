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
k_z = np.linspace(0, k0/3, 300)
angles = np.linspace(0, cm.asin(k_z[-1] / k0)*180/cm.pi, 300)
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in k_z])
k_x_1 = k_x(k_z=k_z, dx=100, pade_order=(1, 1), dz=1, z_order=2, alpha=0)
k_x_2 = k_x(k_z=k_z, dx=100, pade_order=(4, 4), dz=1, z_order=2, alpha=0)
k_x_3 = k_x(k_z=k_z, dx=100, pade_order=(8, 8), dz=1, z_order=2, alpha=0)
k_x_4 = k_x(k_z=k_z, dx=100, pade_order=(8, 8), dz=1, z_order=4, alpha=0)
k_x_5 = k_x(k_z=k_z, dx=100, pade_order=(8, 8), dz=1, z_order=5, alpha=0)

plt.plot(angles, (k_x_1 / k0).real, label='[1/1], 2-й порядок по z')
plt.plot(angles, (k_x_2 / k0).real, label='[4/4], 2-й порядок по z')
plt.plot(angles, (k_x_3 / k0).real, label='[8/8], 2-й порядок по z')
plt.plot(angles, (k_x_4 / k0).real, label='[8/8], 4-й порядок по z')
plt.plot(angles, (k_x_5 / k0).real, label='[8/8], Паде порядок по z')
plt.plot(angles, (k_x_r / k0).real, '--', label='истинное')
plt.xlabel('Угол распространения')
plt.ylabel('re(k_x / k)')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(angles, np.log10(np.abs(k_x_1 - k_x_r) / np.abs(k_x_r)), label='[1/1], 2-й порядок по z')
plt.plot(angles, np.log10(np.abs(k_x_2 - k_x_r) / np.abs(k_x_r)), '--', label='[4/4], 2-й порядок по z')
plt.plot(angles, np.log10(np.abs(k_x_3 - k_x_r) / np.abs(k_x_r)), label='[8/8], 2-й порядок по z')
plt.plot(angles, np.log10(np.abs(k_x_4 - k_x_r) / np.abs(k_x_r)), '--', label='[8/8], 4-й порядок по z')
plt.plot(angles, np.log10(np.abs(k_x_5 - k_x_r) / np.abs(k_x_r)), label='[8/8], Паде порядок по z')
plt.xlabel('Угол распространения')
plt.ylabel('Относительная погрешность')
plt.legend()
plt.grid(True)
plt.show()

################################################################################

k_z = np.linspace(0, 2*k0, 300)
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in k_z])
k_x_1 = k_x(k_z=k_z, dx=0.1, pade_order=(1, 1), dz=0.1, z_order=5, alpha=0, m=2)
k_x_2 = k_x(k_z=k_z, dx=0.1, pade_order=(3, 4), dz=0.1, z_order=5, alpha=0, m=2)
k_x_3 = k_x(k_z=k_z, dx=0.1, pade_order=(4, 4), dz=0.1, z_order=5, alpha=57, m=2)

plt.plot(k_z / k0, (k_x_1 / k0).real, label='[1/1]')
plt.plot(k_z / k0, (k_x_2 / k0).real, label='[3/4]')
plt.plot(k_z / k0, (k_x_3 / k0).real, label='[4/4], alpha=57')
plt.plot(k_z / k0, (k_x_r / k0).real, '--', label='истинное')
plt.xlabel('k_z / k')
plt.ylabel('re(k_x / k)')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(k_z / k0, (k_x_1 / k0).imag, label='[1/1]')
plt.plot(k_z / k0, (k_x_2 / k0).imag, label='[3/4]')
plt.plot(k_z / k0, (k_x_3 / k0).imag, label='[4/4], alpha=57')
plt.plot(k_z / k0, (k_x_r / k0).imag, '--', label='истинное')
plt.xlabel('k_z / k')
plt.ylabel('im(k_x / k)')
plt.legend()
plt.grid(True)
plt.show()