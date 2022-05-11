from joined_cheb_pade_coefs import *
import math as fm
import cmath as cm
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
import propagators.dispersion_relations as disp_rels
import matplotlib.pyplot as plt
import mpmath
import propagators._utils as utils
import matplotlib as mpl
from rwp.petool import PETOOLPropagationTask


def k_x_angle(dx, dz, order, num_coefs, den_coefs, k0, kz_arr):
    if order == 2:
        return np.array([disp_rels.discrete_k_x(k0, dx, dz, num_coefs, den_coefs, kz) for kz in kz_arr])
    if order == 4:
        return np.array([disp_rels.discrete_k_x_4th_order(k0, dx, dz, num_coefs, den_coefs, kz) for kz in kz_arr])
    return None


dx_wl = 50
theta_max = 22
dz_wl = 1 / (2 * fm.sin(theta_max*fm.pi/180))*0.8
print("dz_wl = " + str(dz_wl))
k0 = 2*fm.pi
order = (7, 8)

max_spc_val = fm.sin(fm.pi*dz_wl*fm.sin(theta_max * fm.pi/180))**2 / ((2*fm.pi*dz_wl)**2)*4*0.99
print(max_spc_val * ((2*fm.pi*dz_wl)**2))
joined_ratinterp_num, joined_ratinterp_den, a0 = joined_cheb_pade_coefs(dx_wl, dz_wl, order, max_spc_val, 'ratinterp')
cheb_pade_coefs_num, cheb_pade_coefs_den, a0 = cheb_pade_coefs(dx_wl, order, max_spc_val, 'ratinterp')


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
k_x_joined_ratinterp = k_x_angle(dx_wl, dz_wl, 2, joined_ratinterp_num, joined_ratinterp_den, k0, kz_arr)
k_x_joined_pade = k_x_angle(dx_wl, dz_wl, 2, joined_pade_coefs_num, joined_pade_coefs_den, k0, kz_arr)
k_x_pade_2nd = k_x_angle(dx_wl, dz_wl, 2, pade_coefs_num, pade_coefs_den, k0, kz_arr)
k_x_pade_4th = k_x_angle(dx_wl, dz_wl, 4, pade_coefs_num, pade_coefs_den, k0, kz_arr)
k_x_cheb_pade_2nd = k_x_angle(dx_wl, dz_wl, 2, cheb_pade_coefs_num, cheb_pade_coefs_den, k0, kz_arr)
k_x_cheb_pade_4th = k_x_angle(dx_wl, dz_wl, 4, cheb_pade_coefs_num, cheb_pade_coefs_den, k0, kz_arr)

mpl.rcParams["legend.loc"] = 'upper left'
plt.figure(figsize=(6, 3.2))
plt.plot(angles, (np.abs(k_x_joined_ratinterp - k_x_r)), label='Rat.interp. of discrete propagator')
plt.plot(angles, (np.abs(k_x_joined_pade - k_x_r)), label='Pade approx. of discrete propagator')
plt.plot(angles, (np.abs(k_x_pade_2nd - k_x_r)), label='Pade approx. of semi-discrete propagator (2nd order)')
plt.plot(angles, (np.abs(k_x_pade_4th - k_x_r)), label='Pade approx. of semi-discrete propagator (4th order)')
#plt.plot(angles, (np.abs(k_x_cheb_pade_2nd - k_x_r)), label='ratinterp 2nd order')
#plt.plot(angles, (np.abs(k_x_cheb_pade_4th - k_x_r)), label='ratinterp 4th order')
plt.xlabel('Angle (degrees)')
plt.ylabel('k_x abs. error')
plt.xlim([angles[0], angles[-1]])
plt.ylim([1e-10, 1e0])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3.2))
plt.plot(angles, (np.imag(k_x_joined_ratinterp)), label='Rat.interp. of discrete propagator')
plt.plot(angles, (np.imag(k_x_joined_pade)), label='Pade approx. of discrete propagator')
plt.plot(angles, (np.imag(k_x_pade_2nd)), label='Pade approx. of semi-discrete propagator (2nd order)')
plt.plot(angles, (np.imag(k_x_pade_4th)), label='Pade approx. of semi-discrete propagator (4th order)')
#plt.plot(angles, (np.imag(k_x_cheb_pade_2nd)), label='ratinterp 2nd order')
#plt.plot(angles, (np.imag(k_x_cheb_pade_4th)), label='ratinterp 4th order')
plt.xlabel('Angle (degrees)')
plt.ylabel('Im(k_x/k0)')
plt.xlim([angles[0], angles[-1]])
#plt.ylim([1e-10, 1e0])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

angles = np.linspace(0, theta_max, 10000)
xis = -np.sin(fm.pi * dz_wl * np.sin(np.array(angles) * fm.pi/180))**2 / ((2*fm.pi * dz_wl)**2)*4
vals_joined_pade = np.array([disp_rels.rational_approx(joined_pade_coefs_num, joined_pade_coefs_den, xi) for xi in xis])
vals_joined_ratinterp = np.array([disp_rels.rational_approx(joined_ratinterp_num, joined_ratinterp_den, xi) for xi in xis])
t = 1+(2*fm.pi*dz_wl)**2 * xis/2
t2 = np.array([cm.acosh(z) for z in t])
vals = np.exp(1j * 2 * fm.pi * dx_wl * (np.sqrt(1+(1/(2*fm.pi*dz_wl)**2 * t2**2))-1))

plt.figure(figsize=(6, 3.2))
plt.plot(xis, (np.abs(vals_joined_ratinterp - vals)), label='Rat.interp. of discrete propagator')
plt.plot(xis, (np.abs(vals_joined_pade - vals)), label='Pade approx. of discrete propagator')
plt.xlabel('xi')
plt.ylabel('Abs. error')
plt.xlim([xis[0], xis[-1]])
plt.ylim([1e-12, 1e0])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from rwp.sspade import *
from rwp.vis import *


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 200
env.terrain = Terrain(elevation=lambda x: pyramid2(x, 20, 100, 1.5e3), ground_material=PerfectlyElectricConducting())

ant = GaussAntenna(freq_hz=3000e6, height=env.z_max/2, beam_width=3, eval_angle=0, polarz='H')

max_propagation_angle = theta_max
max_range_m = 3e3

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

joined_pade_task_2 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=5,
                                                      exp_pade_order=order,
                                                      dx_wl=dx_wl/3,
                                                      x_output_filter=1,
                                                      dz_wl=dz_wl,
                                                      z_output_filter=1,
                                                      two_way=False
                                                  ))
joined_pade_field_2 = joined_pade_task_2.calculate()

petool_task_2 = PETOOLPropagationTask(antenna=ant,
                                             env=env,
                                             two_way=False,
                                             max_range_m=max_range_m,
                                             dx_wl=dx_wl/3,
                                             n_dx_out=1,
                                             dz_wl=dz_wl,
                                             n_dz_out=1)
petool_field_2 = petool_task_2.calculate()

pade_2nd_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=2,
                                                      exp_pade_order=order,
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=dz_wl,
                                                      z_output_filter=1,
                                                      two_way=False
                                                  ))
pade_2nd_field = pade_2nd_task.calculate()

pade_4th_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=4,
                                                      exp_pade_order=order,
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=dz_wl,
                                                      z_output_filter=1,
                                                      two_way=False
                                                  ))
pade_4th_field = pade_4th_task.calculate()

joined_cheb_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=2,
                                                      exp_pade_coefs=list(zip_longest(joined_ratinterp_num, joined_ratinterp_den, fillvalue=0.0j)),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=dz_wl,
                                                      z_output_filter=1,
                                                      two_way=False,
                                                      inv_z_transform_rtol=1e-4
                                                  ))
joined_cheb_pade_field = joined_cheb_pade_task.calculate()

petool_task = PETOOLPropagationTask(antenna=ant,
                                             env=env,
                                             two_way=False,
                                             max_range_m=max_range_m,
                                             dx_wl=dx_wl,
                                             n_dx_out=1,
                                             dz_wl=dz_wl,
                                             n_dz_out=1)
petool_field = petool_task.calculate()

petool_vis = FieldVisualiser(petool_field, env=env, trans_func=lambda x: 2 * x, label='Метод расщ. Фурье',
                             x_mult=1E-3)

joined_pade_vis = FieldVisualiser(joined_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Joined Pade")
plt = joined_pade_vis.plot2d(min=-120, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

pade_2nd_vis = FieldVisualiser(pade_2nd_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Pade 2nd")
plt = pade_2nd_vis.plot2d(min=-120, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

pade_4th_vis = FieldVisualiser(pade_4th_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Pade 4th")
plt = pade_4th_vis.plot2d(min=-120, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

joined_cheb_pade_vis = FieldVisualiser(joined_cheb_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3, label="Opt")
plt = joined_cheb_pade_vis.plot2d(min=-120, max=0, show_terrain=True)
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

import matplotlib.pyplot as plt
mpl.rcParams['axes.titlesize'] = 'medium'

f, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(0, 10)
extent = [joined_pade_field.x_grid[0]*1e-3, joined_pade_field.x_grid[-1]*1e-3, joined_pade_field.z_grid[0], joined_pade_field.z_grid[-1]]

err = np.abs(2*petool_field.field - 20*np.log10(np.abs(joined_pade_field.field[1:,:-1])+1e-16))
im = ax.imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax.grid()
#ax.set_title('Δz=2.0λ, 2nd order')
ax.set_xlabel('Range (km)')
ax.set_ylabel('Height (m)')

terrain_grid = np.array([joined_pade_vis.env.terrain.elevation(v) for v in joined_pade_vis.x_grid / joined_pade_vis.x_mult])
ax.plot(joined_pade_vis.x_grid, terrain_grid, 'k')
ax.fill_between(joined_pade_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')

f.colorbar(im, shrink=0.6, location='right', fraction=0.046, pad=0.04)
#f.tight_layout()
plt.show()


f, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(0, 10)
extent = [joined_pade_field.x_grid[0]*1e-3, joined_pade_field.x_grid[-1]*1e-3, joined_pade_field.z_grid[0], joined_pade_field.z_grid[-1]]

err = np.abs(2*petool_field_2.field - 20*np.log10(np.abs(joined_pade_field_2.field[1:,:-1])+1e-16))
im = ax.imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax.grid()
#ax.set_title('Δz=2.0λ, 2nd order')
ax.set_xlabel('Range (km)')
ax.set_ylabel('Height (m)')

terrain_grid = np.array([joined_pade_vis.env.terrain.elevation(v) for v in joined_pade_vis.x_grid / joined_pade_vis.x_mult])
ax.plot(joined_pade_vis.x_grid, terrain_grid, 'k')
ax.fill_between(joined_pade_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')

f.colorbar(im, shrink=0.6, location='right', fraction=0.046, pad=0.04)
#f.tight_layout()
plt.show()



f, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(0, 10)
extent = [joined_pade_field.x_grid[0]*1e-3, joined_pade_field.x_grid[-1]*1e-3, joined_pade_field.z_grid[0], joined_pade_field.z_grid[-1]]

err = np.abs(2*petool_field.field - 20*np.log10(np.abs(pade_4th_field.field[1:,:-1])+1e-16))
im = ax.imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax.grid()
#ax.set_title('Δz=2.0λ, 2nd order')
ax.set_xlabel('Range (km)')
ax.set_ylabel('Height (m)')

terrain_grid = np.array([joined_pade_vis.env.terrain.elevation(v) for v in joined_pade_vis.x_grid / joined_pade_vis.x_mult])
ax.plot(joined_pade_vis.x_grid, terrain_grid, 'k')
ax.fill_between(joined_pade_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')

f.colorbar(im, shrink=0.6, location='right', fraction=0.046, pad=0.04)
#f.tight_layout()
plt.show()



f, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(0, 10)
extent = [joined_pade_field.x_grid[0]*1e-3, joined_pade_field.x_grid[-1]*1e-3, joined_pade_field.z_grid[0], joined_pade_field.z_grid[-1]]

err = np.abs(2*petool_field.field - 20*np.log10(np.abs(joined_cheb_pade_field.field[1:,:-1])+1e-16))
im = ax.imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax.grid()
#ax.set_title('Δz=2.0λ, 2nd order')
ax.set_xlabel('Range (km)')
ax.set_ylabel('Height (m)')

terrain_grid = np.array([joined_pade_vis.env.terrain.elevation(v) for v in joined_pade_vis.x_grid / joined_pade_vis.x_mult])
ax.plot(joined_pade_vis.x_grid, terrain_grid, 'k')
ax.fill_between(joined_pade_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')

f.colorbar(im, shrink=0.6, location='right', fraction=0.046, pad=0.04)
#f.tight_layout()
plt.show()