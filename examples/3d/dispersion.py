import math as fm
import cmath as cm
import numpy as np
import propagators._utils as utils
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


n = 200
k0 = 2 * fm.pi
dx = 60
dy = 1
dz = 1
order = (6, 7)
phi_max_degrees = 30
theta_max_degrees = 20
k_y_min = -k0 * fm.sin(fm.radians(theta_max_degrees))
k_y_max = k0 * fm.sin(fm.radians(theta_max_degrees))
k_z_min = -k0 * fm.sin(fm.radians(theta_max_degrees))
k_z_max = k0 * fm.sin(fm.radians(theta_max_degrees))


def xi1(alpha=0.0):
    k_y_grid = np.linspace(k_y_min, k_y_max, n)
    k_z_grid = np.linspace(k_z_min, k_z_max, n)
    k_y_2d_grid, k_z_2d_grid = np.meshgrid(k_y_grid, k_z_grid, indexing='ij')
    return -(k_y_2d_grid / k0) ** 2 - (k_z_2d_grid / k0) ** 2 + alpha


def xi2(alpha=0.0):
    k_y_grid = np.linspace(k_y_min, k_y_max, n)
    k_z_grid = np.linspace(k_z_min, k_z_max, n)
    k_y_2d_grid, k_z_2d_grid = np.meshgrid(k_y_grid, k_z_grid, indexing='ij')
    return -(2*np.sin(k_y_2d_grid*dy/2) / (k0*dy))**2 - (2*np.sin(k_z_2d_grid*dz/2) / (k0*dz))**2 + alpha


def discrete_k_x_2d(alpha=0.0):
    xi = xi2(alpha)
    pade_coefs, c0 = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=k0, dx=dx, a0=0)
    t = c0
    for (a, b) in pade_coefs:
        t *= (1 + a*xi) / (1 + b*xi)
    return k0*fm.sqrt(1-alpha) - 1j/dx * np.log(t)


def k_x_2d():
    k_y_grid = np.linspace(k_y_min, k_y_max, n)
    k_z_grid = np.linspace(k_z_min, k_z_max, n)
    k_y_2d_grid, k_z_2d_grid = np.meshgrid(k_y_grid, k_z_grid, indexing='ij')
    return np.sqrt(k0**2 - k_y_2d_grid**2 - k_z_2d_grid**2)


err1 = 10*np.log10(np.abs((k_x_2d() - discrete_k_x_2d())/k_x_2d()))
err2 = 10*np.log10(np.abs((k_x_2d() - discrete_k_x_2d(alpha=-fm.sin(fm.radians(theta_max_degrees))**2/2))/k_x_2d()))

# extent = [k_y_min/k0, k_y_max/k0, k_z_min/k0, k_z_max/k0]
# norm = Normalize(0, -30)
# plt.imshow(err1, extent=extent, norm=norm, cmap=plt.get_cmap('binary'))
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.show()


k_y_grid = np.linspace(k_y_min, k_y_max, n)
# plt.plot(k_y_grid/k0, err1[100,:], k_y_grid/k0, err2[100,:])
# plt.grid(True)
# plt.show()


plt.plot(k_y_grid, k_x_2d()[100,:], k_y_grid, discrete_k_x_2d(alpha=0.1)[100,:])
plt.grid(True)
plt.show()