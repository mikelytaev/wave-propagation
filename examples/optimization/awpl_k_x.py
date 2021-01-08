from propagators._utils import *
import numpy as np
import matplotlib.pyplot as plt

from propagators._utils import d_k_x

k0 = 2*cm.pi
k_z = np.linspace(0, k0/2, 300)
angles = np.linspace(0, fm.asin(k_z[-1] / k0)*180/cm.pi, 300)
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in k_z])
dx = 500
k_x_1 = d_k_x(k_z=k_z, dx=dx, pade_order=(1, 1), dz=0.0000001, z_order=2, alpha=0)
k_x_2 = d_k_x(k_z=k_z, dx=dx, pade_order=(3, 4), dz=0.0000001, z_order=2, alpha=0)
k_x_3 = d_k_x(k_z=k_z, dx=dx, pade_order=(7, 8), dz=0.0000001, z_order=2, alpha=0)
k_x_4 = d_k_x(k_z=k_z, dx=dx, pade_order=(10, 11), dz=0.0000001, z_order=2, alpha=0)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(12, 3.2))
ax1.plot(angles, (np.abs(k_x_1 - k_x_r) / np.abs(k_x_r)).real, label='Pade-[1/1]')
ax1.plot(angles, (np.abs(k_x_2 - k_x_r) / np.abs(k_x_r)).real, label='Pade-[3/4]')
ax1.plot(angles, (np.abs(k_x_3 - k_x_r) / np.abs(k_x_r)).real, label='Pade-[7/8]')
ax1.plot(angles, (np.abs(k_x_4 - k_x_r) / np.abs(k_x_r)).real, label='Pade-[10/11]')
ax1.set_xlabel('Propagation angle, degrees')
ax1.set_ylabel('k_x relative error')
ax1.set_xlim([0, 15])
ax1.set_yscale("log")
#ax1.set_ylim([1e-10, 1e-1])
ax1.set_ylim([2e-6, 1e-1])
ax1.legend()
ax1.set_title("(a)")
ax1.grid()

angle = 5
k_z = k0 * fm.sin(angle / 180 * cm.pi)
k_x_r = cm.sqrt(k0**2 - k_z**2)
dz = 0.8
dxs = np.linspace(50, 1000, 50)
k_x_11 = d_k_x(k_z=k_z, dx=dxs, pade_order=(1, 1), dz=dz, z_order=4)
k_x_34 = d_k_x(k_z=k_z, dx=dxs, pade_order=(3, 4), dz=dz, z_order=4)
k_x_78 = d_k_x(k_z=k_z, dx=dxs, pade_order=(7, 8), dz=dz, z_order=4)
k_x_10_11 = d_k_x(k_z=k_z, dx=dxs, pade_order=(10, 11), dz=dz, z_order=4)

ax2.plot(dxs, (np.abs(k_x_11 - k_x_r))/k0, label='[1/1], 4th order by z')
ax2.plot(dxs, (np.abs(k_x_34 - k_x_r))/k0, label='[3/4], 4th order by z')
ax2.plot(dxs, (np.abs(k_x_78 - k_x_r))/k0, label='[7/8], 4th order by z')
ax2.plot(dxs, (np.abs(k_x_10_11 - k_x_r))/k0, label='[10/11], 4th order by z')
ax2.set_xlabel('dx, wavelengths')
ax2.set_xlim([0, 1000])
#ax2.set_ylim([2e-6, 4e-3])
#ax2.set_ylabel('k_x relative error')
ax2.set_yscale("log")
ax2.legend(loc='upper left')
ax2.set_title("(b)")
ax2.grid()

angle = 5
k_z = k0 * fm.sin(angle / 180 * cm.pi)
k_x_r = fm.sqrt(k0**2 - k_z**2)
dzs = np.linspace(0.01, 10, 1000)
k_x_2 = d_k_x(k_z=k_z, dx=500, pade_order=(10, 11), dz=dzs, z_order=2, alpha=0)
k_x_2_4 = d_k_x(k_z=k_z, dx=500, pade_order=(10, 11), dz=dzs, z_order=4, alpha=0)
k_x_2_78 = d_k_x(k_z=k_z, dx=500, pade_order=(7, 8), dz=dzs, z_order=2, alpha=0)
k_x_2_4_78 = d_k_x(k_z=k_z, dx=500, pade_order=(7, 8), dz=dzs, z_order=4, alpha=0)

ax3.plot(dzs, (np.abs(k_x_2 - k_x_r))/k0, label='[10/11], 2nd order by z')
ax3.plot(dzs, (np.abs(k_x_2_4 - k_x_r))/k0, label='[10/11], 4th order by z')
ax3.plot(dzs, (np.abs(k_x_2_78 - k_x_r))/k0, '--', label='[7/8], 2nd order by z')
ax3.plot(dzs, (np.abs(k_x_2_4_78 - k_x_r))/k0, '--', label='[7/8], 4th order by z')
ax3.set_xlabel('dz, wavelengths')
ax3.set_xlim([0, 10])
#ax3.set_ylabel('k_x relative error')
ax3.set_yscale("log")
ax3.legend()
ax3.set_title("(c)")
ax3.grid()
f.tight_layout()
plt.subplots_adjust(top=0.915, wspace=0.075)
plt.show()
