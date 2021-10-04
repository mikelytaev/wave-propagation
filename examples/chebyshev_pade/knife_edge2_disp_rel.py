import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
import propagators.dispersion_relations as disp_rels
import math as fm
import cmath as cm
from cheb_pade_coefs import *
import matplotlib.pyplot as plt
import propagators._utils as utils


def k_x_angle(dx, dz, num_coefs, den_coefs, k0, kz_arr):
    return np.array([disp_rels.discrete_k_x_4th_order(k0, dx, dz, num_coefs, den_coefs, kz) for kz in kz_arr])

freq = 3000e6
wl = 3e8 / freq
k0 = 2*fm.pi / wl
dx_wl = 50
dz_wl = 0.1
pade_order = (6, 7)
max_angle_degrees = 20
coefs, a0 = cheb_pade_coefs(dx_wl, pade_order, fm.sin(max_angle_degrees*fm.pi/180)**2, 'ratinterp')
coefs_num_ratinterp = np.array([a[0] for a in coefs])
coefs_den_ratinterp = np.array([b[1] for b in coefs])
coefs, a0 = cheb_pade_coefs(dx_wl, pade_order, fm.sin(max_angle_degrees*fm.pi/180)**2, 'aaa')
coefs_num_aaa = np.array([a[0] for a in coefs])
coefs_den_aaa = np.array([b[1] for b in coefs])
pade_coefs = utils.pade_propagator_coefs(pade_order=pade_order, diff2=lambda x: x, k0=2*cm.pi, dx=dx_wl)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])
kz_arr = np.linspace(0, 1*k0, 10000)
angles = [cm.asin(kz/k0).real*180/fm.pi for kz in kz_arr]
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in kz_arr])
k_x_ratinterp = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_num_ratinterp, coefs_den_ratinterp, k0, kz_arr)
k_x_aaa = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_num_aaa, coefs_den_aaa, k0, kz_arr)
k_x_pade = k_x_angle(dx_wl * wl, dz_wl * wl, pade_coefs_num, pade_coefs_den, k0, kz_arr)


plt.figure(figsize=(6, 3.2))
plt.plot(angles, (np.abs(k_x_ratinterp - k_x_r)), label='ratinterp')
plt.plot(angles, (np.abs(k_x_aaa - k_x_r)), label='aaa')
plt.plot(angles, (np.abs(k_x_pade - k_x_r)), label='Pade')
plt.xlabel('angle')
plt.ylabel('k_x abs. error')
plt.xlim([angles[0], angles[-1]])
#plt.ylim([1e-10, 1e-1])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


kz_arr = np.linspace(0, 2*k0, 10000)
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in kz_arr])
k_x_ratinterp = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_num_ratinterp, coefs_den_ratinterp, k0, kz_arr)
k_x_aaa = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_num_aaa, coefs_den_aaa, k0, kz_arr)
k_x_pade = k_x_angle(dx_wl * wl, dz_wl * wl, pade_coefs_num, pade_coefs_den, k0, kz_arr)

plt.figure(figsize=(6, 3.2))
plt.plot(kz_arr, (np.imag(k_x_pade)), label='imag Pade')
plt.plot(kz_arr, (np.imag(k_x_aaa)), label='imag AAA')
plt.plot(kz_arr, (np.imag(k_x_ratinterp)), label='imag ratinterp')
plt.xlabel('k_z')
plt.ylabel('k_x abs. error')
plt.xlim([kz_arr[0], kz_arr[-1]])
#plt.ylim([1e-10, 1e-1])
#plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3.2))
plt.plot(kz_arr, (np.real(k_x_pade)), label='real Pade')
plt.plot(kz_arr, (np.real(k_x_aaa)), label='real AAA')
plt.plot(kz_arr, (np.real(k_x_ratinterp)), label='real ratinterp')
plt.xlabel('k_z')
plt.ylabel('k_x abs. error')
plt.xlim([kz_arr[0], kz_arr[-1]])
#plt.ylim([1e-10, 1e-1])
#plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()