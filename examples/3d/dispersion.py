import math as fm
import cmath as cm
import numpy as np
import propagators._utils as utils
import matplotlib.pyplot as plt


n = 200
k0 = 2 * fm.pi
dx = 10
dy = 1
dz = 1
order = (6, 7)
alpha = 0
phi_max_degrees = 30
theta_max_degrees = 60
k_y_min = 0
k_y_max = k0 * fm.cos(fm.radians(phi_max_degrees)) * fm.sin(fm.radians(theta_max_degrees))
k_z_min = 0
k_z_max = k0 * fm.sin(fm.radians(phi_max_degrees)) * fm.sin(fm.radians(theta_max_degrees))


def xi1():
    k_y_grid = np.linspace(k_y_min, k_y_max, n)
    k_z_grid = np.linspace(k_z_min, k_z_max, n)
    k_y_2d_grid, k_z_2d_grid = np.meshgrid(k_y_grid, k_z_grid, indexing='ij')
    return -(k_y_2d_grid / k0) ** 2 - (k_z_2d_grid / k0) ** 2 + alpha


def discrete_k_x():
    xi = xi1()
    pade_coefs, c0 = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=k0, dx=dx)
    t = 1.0 + 0.0j
    for (a, b) in pade_coefs:
        t *= (1 + a*xi) / (1 + b*xi)
    return k0*fm.sqrt(1-alpha) - 1j/dx * np.log(t)


def k_x():
    k_y_grid = np.linspace(0, k_y_max, n)
    k_z_grid = np.linspace(0, k_z_max, n)
    k_y_2d_grid, k_z_2d_grid = np.meshgrid(k_y_grid, k_z_grid, indexing='ij')
    return np.sqrt(k0**2 - k_y_2d_grid**2 - k_z_2d_grid**2)


err = 10*np.log10(np.abs(discrete_k_x() - k_x()))
extent = [k_y_min, k_y_max, k_z_min, k_z_max]
plt.imshow(err)
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()