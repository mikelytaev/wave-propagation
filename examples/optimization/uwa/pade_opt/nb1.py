import numpy as np

from examples.optimization.uwa.pade_opt.utils import get_optimal

from propagators._utils import *

# dz = 20
# errors_vs_kz = [fourth_order_error_kz(kz, dz) for kz in np.linspace(-fm.pi*3, fm.pi*3, 2000)]
# plt.plot(errors_vs_kz)
# plt.grid(True)
#plt.show()

# kz = 0.25#*fm.pi
# z_grid = np.linspace(0, 40, 2000)
# errors_vs_dz = [fourth_order_error_kz(kz, dz) for dz in z_grid]
# plt.plot(z_grid, errors_vs_dz)
# plt.grid(True)
#plt.show()


k0 = 2*fm.pi
theta_max_degrees = 33
k_z_max = k0*fm.sin(theta_max_degrees*fm.pi/180)
xi_bounds = [-k_z_max**2/k0**2-0.23*0, 0]
# z_grid = np.linspace(0.01, 5, 100)
# errors = precision_step(xi_bounds, k_z_max, [10], z_grid, (7, 8))
#
# plt.plot(z_grid, 10*np.log10(errors[0]))
# plt.grid(True)
# plt.show()

dx, dz = get_optimal(1000, 1e-2, xi_bounds[0], k_z_max, shift_pade=False)
print(f"dx = {dx}; dz = {dz}")

dx, dz = get_optimal(1000, 1e-2, xi_bounds[0], k_z_max, shift_pade=True)
print(f"dx = {dx}; dz = {dz}")
