import math as fm
import cmath as cm
import numpy as np
import propagators._utils as utils
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


n = 200
freq_hz = 300e6
c = 3e8
k0 = 2 * fm.pi * freq_hz / c
dx_m = 7
dy_m = 0.09
dz_m = 0.09
order = (6, 7)
phi_max_degrees = 30
theta_max_degrees = 30
k_y_min = -k0 * fm.sin(fm.radians(theta_max_degrees))
k_y_max = k0 * fm.sin(fm.radians(theta_max_degrees))
k_z_min = -k0 * fm.sin(fm.radians(theta_max_degrees))
k_z_max = k0 * fm.sin(fm.radians(theta_max_degrees))


def xi1(alpha=0.0):
    k_y_grid = np.linspace(k_y_min, k_y_max, n)
    k_z_grid = np.linspace(k_z_min, k_z_max, n)
    k_y_2d_grid, k_z_2d_grid = np.meshgrid(k_y_grid, k_z_grid, indexing='ij')
    return -(k_y_2d_grid / k0) ** 2 - (k_z_2d_grid / k0) ** 2 + alpha


def xi2(k0, k0sh):
    k_y_grid = np.linspace(k_y_min, k_y_max, n)
    k_z_grid = np.linspace(k_z_min, k_z_max, n)
    alpha = (k0 / k0sh) ** 2 - 1
    k_y_2d_grid, k_z_2d_grid = np.meshgrid(k_y_grid, k_z_grid, indexing='ij')
    z = 1
    xi_y = -(2 * np.sin(k_y_2d_grid * dz_m / 2) / (k0sh * dz_m)) ** 2 - z * 4 / 3 * np.sin(k_y_2d_grid * dz_m / 2) ** 4 / (
                k0sh * dz_m) ** 2 + alpha / 2
    xi_z = -(2 * np.sin(k_z_2d_grid * dz_m / 2) / (k0sh * dz_m)) ** 2 - z * 4 / 3 * np.sin(k_z_2d_grid * dz_m / 2) ** 4 / (
                k0sh * dz_m) ** 2 + alpha / 2
    return xi_y + xi_z


def discrete_k_x_2d(shift=False):
    c0 = fm.sqrt(2 / (1 + fm.cos(fm.radians(theta_max_degrees)) ** 2)) * 3e8
    k0sh = 2 * fm.pi * freq_hz / c0 if shift else k0
    xi = xi2(k0, k0sh)
    print(f"{np.min(xi)} {np.max(xi)}")
    pade_coefs, c0 = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=k0sh, dx=dx_m)
    t = c0
    for (a, b) in pade_coefs:
        t *= (1 + a*xi) / (1 + b*xi)
    return k0sh - 1j/dx_m * np.log(t)


def k_x_2d():
    k_y_grid = np.linspace(k_y_min, k_y_max, n)
    k_z_grid = np.linspace(k_z_min, k_z_max, n)
    k_y_2d_grid, k_z_2d_grid = np.meshgrid(k_y_grid, k_z_grid, indexing='ij')
    return np.sqrt(k0**2 - k_y_2d_grid**2 - k_z_2d_grid**2)


err1 = 10*np.log10(np.abs((k_x_2d() - discrete_k_x_2d(shift=False))))
err2 = 10*np.log10(np.abs((k_x_2d() - discrete_k_x_2d(shift=True))))

# extent = [k_y_min/k0, k_y_max/k0, k_z_min/k0, k_z_max/k0]
# norm = Normalize(0, -100)
# plt.imshow(err2, extent=extent, norm=norm, cmap=plt.get_cmap('binary'))
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.show()


k_y_grid = np.linspace(k_y_min, k_y_max, n)
plt.plot(k_y_grid/k0, err1[100,:], k_y_grid/k0, err2[100,:])
plt.grid(True)
plt.show()


# plt.plot(k_y_grid/k0, k_x_2d()[100,:], k_y_grid/k0, discrete_k_x_2d(alpha=0.125)[100,:])
# plt.grid(True)
# plt.show()