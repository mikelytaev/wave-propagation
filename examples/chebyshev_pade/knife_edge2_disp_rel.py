import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
import propagators.dispersion_relations as disp_rels
import math as fm
import cmath as cm
from cheb_pade_coefs import *
import matplotlib.pyplot as plt


def k_x_angle(dx, dz, num_coefs, den_coefs, k0, kz_arr):
    return np.array([disp_rels.discrete_k_x(k0, dx, dz, num_coefs, den_coefs, kz) for kz in kz_arr])

freq = 3000e6
wl = 3e8 / freq
k0 = 2*fm.pi / wl
dx_wl = 2
dz_wl = 0.0000001
coefs, a0 = cheb_pade_coefs(dx_wl, (9, 9), fm.sin(87.5*fm.pi/180)**2, 'aaa')
coefs_num = np.array([a[0] for a in coefs])
coefs_den = np.array([b[1] for b in coefs])
kz_arr = np.linspace(0, 2*k0, 10000)
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in kz_arr])
k_x_1 = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_num, coefs_den, k0, kz_arr)


plt.figure(figsize=(6, 3.2))
plt.plot(kz_arr, (np.abs(k_x_1 - k_x_r)), label='opt')
plt.xlabel('k_z')
plt.ylabel('k_x abs. error')
plt.xlim([kz_arr[0], kz_arr[-1]])
#plt.ylim([1e-10, 1e-1])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(6, 3.2))
# plt.plot(angles, (np.real(k_x_1)), label='opt real')
#plt.plot(kz_arr, (np.imag(k_x_r)), label='Pade real')
plt.plot(kz_arr, (np.real(k_x_1)), label='opt imag')
plt.xlabel('k_z')
plt.ylabel('k_x abs. error')
plt.xlim([kz_arr[0], kz_arr[-1]])
#plt.ylim([1e-10, 1e-1])
#plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()