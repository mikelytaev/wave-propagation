from joined_cheb_pade_coefs import *
import math as fm
import cmath as cm
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
import propagators.dispersion_relations as disp_rels
import matplotlib.pyplot as plt
import mpmath
import propagators._utils as utils


def k_x_angle(dx, dz, num_coefs, den_coefs, k0, kz_arr):
    return np.array([disp_rels.discrete_k_x(k0, dx, dz, num_coefs, den_coefs, kz) for kz in kz_arr])


dx_wl = 50
theta_max = 20
dz_wl = 1 / (2 * fm.sin(theta_max*fm.pi/180))
print("dz_wl = " + str(dz_wl))
k0 = 2*fm.pi

order = (7, 8)
max_spc_val = fm.sin(fm.pi*dz_wl*fm.sin(theta_max * fm.pi/180))**2 / ((2*fm.pi*dz_wl)**2)*4*0.99
print(max_spc_val * ((2*fm.pi*dz_wl)**2))
a, b, a0 = joined_cheb_pade_coefs(dx_wl, dz_wl, order, max_spc_val, 'chebpade')

def diff2(s):
    return mpmath.acosh(1 + (k0 * dz_wl) ** 2 * s / 2) ** 2 / (k0 * dz_wl) ** 2

joined_pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=diff2, k0=2*cm.pi, dx=dx_wl)
joined_pade_coefs_num = np.array([a[0] for a in joined_pade_coefs])
joined_pade_coefs_den = np.array([a[1] for a in joined_pade_coefs])

pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2*cm.pi, dx=dx_wl)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])

kz_arr = np.linspace(0, 1*k0, 10000)
angles = [cm.asin(kz/k0).real*180/fm.pi for kz in kz_arr]
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in kz_arr])
k_x_ratinterp = k_x_angle(dx_wl, dz_wl, a, b, k0, kz_arr)
k_x_3 = k_x_angle(dx_wl, dz_wl, joined_pade_coefs_num, joined_pade_coefs_den, k0, kz_arr)
k_x_pade = k_x_angle(dx_wl, dz_wl, pade_coefs_num, pade_coefs_den, k0, kz_arr)


plt.figure(figsize=(6, 3.2))
plt.plot(angles, (np.abs(k_x_ratinterp - k_x_r)), label='ratinterp')
plt.plot(angles, (np.abs(k_x_3 - k_x_r)), label='Joined Pade')
plt.plot(angles, (np.abs(k_x_pade - k_x_r)), label='Pade')
plt.xlabel('angle')
plt.ylabel('k_x abs. error')
plt.xlim([angles[0], angles[-1]])
plt.ylim([1e-10, 1e-1])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

kz_arr = np.linspace(0, 1.5*k0, 10000)
angles = [cm.asin(kz/k0).real*180/fm.pi for kz in kz_arr]
k_x_r = np.array([cm.sqrt(k0**2 - kz**2) for kz in kz_arr])
k_x_ratinterp = k_x_angle(dx_wl, dz_wl, a, b, k0, kz_arr)
k_x_3 = k_x_angle(dx_wl, dz_wl, joined_pade_coefs_num, joined_pade_coefs_den, k0, kz_arr)
plt.figure(figsize=(6, 3.2))
# plt.plot(angles, (np.real(k_x_1)), label='opt real')
plt.plot(kz_arr, (np.imag(k_x_3)), label='Joined Pade imag')
plt.plot(kz_arr, (np.imag(k_x_ratinterp)), label='Joined Cheb.-Pade imag')
plt.xlabel('k_z')
plt.ylabel('k_x abs. error')
plt.xlim([kz_arr[0], kz_arr[-1]])
#plt.ylim([1e-10, 1e-1])
#plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from rwp.sspade import *
from rwp.vis import *


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 500
env.terrain = Terrain(ground_material=PerfectlyElectricConducting())
env.terrain = Terrain(elevation=lambda x: pyramid(x, 10, 250, 5e3), ground_material=PerfectlyElectricConducting())

ant = GaussAntenna(freq_hz=3000e6, height=env.z_max/2, beam_width=3, eval_angle=0, polarz='H')

max_propagation_angle = theta_max
max_range_m = 10e3

joined_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=5,
                                                      exp_pade_order=order,
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=dz_wl,
                                                      z_output_filter=1,
                                                      two_way=False
                                                  ))
joined_pade_field = joined_pade_task.calculate()

joined_cheb_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=2,
                                                      exp_pade_coefs=list(zip_longest(a, b, fillvalue=0.0j)),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=dz_wl,
                                                      z_output_filter=1,
                                                      two_way=False,
                                                      inv_z_transform_rtol=1e-4
                                                  ))
joined_cheb_pade_field = joined_cheb_pade_task.calculate()

joined_pade_vis = FieldVisualiser(joined_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Joined Pade")
plt = joined_pade_vis.plot2d(min=-80, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

joined_cheb_pade_vis = FieldVisualiser(joined_cheb_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Opt")
plt = joined_cheb_pade_vis.plot2d(min=-80, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

joined_cheb_pade_vis.plot_hor(250, joined_pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

f, ax1 = plt.subplots(1, 1, sharey=True)
joined_cheb_pade_vis.plot_ver(max_range_m, ax1, joined_pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()
