from rwp.sspade import *
from rwp.vis import *
from examples.chebyshev_pade.cheb_pade_coefs import *

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
import propagators.dispersion_relations as disp_rels

import propagators._utils as utils


def k_x_angle(dx, dz, num_coefs, den_coefs, k0, kz_arr):
    return np.array([disp_rels.discrete_k_x_4th_order(k0, dx, dz, num_coefs, den_coefs, kz) for kz in kz_arr])


order = (6, 7)
dx_wl = 50
dz_wl = 0.25
method = "ratinterp"
max_propagation_angle = 20
max_range = 4e3
pyramid_angle = 20


wl = 0.1
max_range_m = max_range
coefs, a0 = cheb_pade_coefs(dx_wl, order, fm.sin(max_propagation_angle*fm.pi/180)**2, method)

env = Troposphere(flat=True)
env.z_max = 300
env.terrain = Terrain(elevation=lambda x: pyramid2(x, pyramid_angle, 150, max_range_m/2), ground_material=PerfectlyElectricConducting())

ant = GaussAntenna(wavelength=wl, height=150, beam_width=4, eval_angle=0, polarz='H')


cheb_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=4,
                                                      exp_pade_coefs=coefs,
                                                      exp_pade_a0_coef=a0,
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=dz_wl,
                                                      z_output_filter=8,
                                                      two_way=False
                                                  ))
cheb_pade_field = cheb_pade_task.calculate()

pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
HelmholtzPropagatorComputationalParams(
    terrain_method=TerrainMethod.staircase,
    max_propagation_angle=max_propagation_angle,
    exp_pade_order=order,
    modify_grid=False,
    z_order=4,
    dx_wl=dx_wl,
    x_output_filter=1,
    dz_wl=dz_wl,
    z_output_filter=8,
    two_way=False
))
pade_field = pade_task.calculate()

from rwp.petool import PETOOLPropagationTask
petool_task = PETOOLPropagationTask(antenna=ant,
                                             env=env,
                                             two_way=False,
                                             max_range_m=max_range_m,
                                             dx_wl=dx_wl,
                                             n_dx_out=1,
                                             dz_wl=dz_wl,
                                             n_dz_out=8)
petool_field = petool_task.calculate()


cheb_pade_vis = FieldVisualiser(cheb_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
                                label='Cheb.-Pade.-[7/7]', x_mult=1E-3)

pade_vis = FieldVisualiser(pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
                                label='Pade-[7/7]', x_mult=1E-3)

petool_vis = FieldVisualiser(petool_field, env=env, trans_func=lambda x: 2 * x, label='Метод расщ. Фурье',
                             x_mult=1E-3)
plt = petool_vis.plot2d(0, -100, True)
plt.xlabel("Range (km)")
plt.ylabel("Height (m)")
plt.tight_layout()
plt.grid(True)
plt.show()

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['axes.titlesize'] = 'medium'
f, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(0, 10)
extent = [cheb_pade_field.x_grid[0]*1e-3, cheb_pade_field.x_grid[-1]*1e-3, cheb_pade_field.z_grid[0], cheb_pade_field.z_grid[-1]]

err = np.abs(2*petool_field.field - 20*np.log10(np.abs(cheb_pade_field.field[1:,:])+1e-16))
im = ax.imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax.grid()
#ax.set_title('Δz=2.0λ, 2nd order')
ax.set_xlabel('Range (km)')
ax.set_ylabel('Height (m)')

terrain_grid = np.array([cheb_pade_vis.env.terrain.elevation(v) for v in cheb_pade_vis.x_grid / cheb_pade_vis.x_mult])
ax.plot(cheb_pade_vis.x_grid, terrain_grid, 'k')
ax.fill_between(cheb_pade_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')

f.colorbar(im, shrink=0.6, location='right', fraction=0.046, pad=0.04)
#f.tight_layout()
plt.show()

f, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 3.2), constrained_layout=True)
norm = Normalize(0, 10)
extent = [cheb_pade_field.x_grid[0] * 1e-3, cheb_pade_field.x_grid[-1] * 1e-3, cheb_pade_field.z_grid[0],
          cheb_pade_field.z_grid[-1]]

err = np.abs(2 * petool_field.field - 20 * np.log10(np.abs(pade_field.field[1:, :]) + 1e-16))
im = ax.imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax.grid()
#ax.set_title('Δz=2.0λ, 2nd order')
ax.set_xlabel('Range (km)')
ax.set_ylabel('Height (m)')

terrain_grid = np.array(
    [cheb_pade_vis.env.terrain.elevation(v) for v in cheb_pade_vis.x_grid / cheb_pade_vis.x_mult])
ax.plot(cheb_pade_vis.x_grid, terrain_grid, 'k')
ax.fill_between(cheb_pade_vis.x_grid, terrain_grid * 0, terrain_grid, color='brown')

f.colorbar(im, shrink=0.6, location='right', fraction=0.046, pad=0.04)
# f.tight_layout()
plt.show()

plt = cheb_pade_vis.plot_hor_over_terrain(20, petool_vis)
plt.xlabel('Range (km)')
plt.ylabel('20log|u| (dB)')
#plt.xlim([2, 8])
plt.ylim([-120, -20])
plt.grid(True)
plt.tight_layout()
plt.show()

coefs_num = np.array([a[0] for a in coefs])
coefs_den = np.array([b[1] for b in coefs])

coefs2, a0 = cheb_pade_coefs(dx_wl, (order[1], order[1]), fm.sin(max_propagation_angle*fm.pi/180)**2, method)
coefs2_num = np.array([a[0] for a in coefs2])
coefs2_den = np.array([b[1] for b in coefs2])

pade_coefs = utils.pade_propagator_coefs(pade_order=order, diff2=lambda x: x, k0=2 * cm.pi, dx=dx_wl)
pade_coefs_num = np.array([a[0] for a in pade_coefs])
pade_coefs_den = np.array([a[1] for a in pade_coefs])

pade_coefs2 = utils.pade_propagator_coefs(pade_order=(order[1], order[1]), diff2=lambda x: x, k0=2 * cm.pi, dx=dx_wl)
pade_coefs2_num = np.array([a[0] for a in pade_coefs2])
pade_coefs2_den = np.array([a[1] for a in pade_coefs2])

coefs_aaa, a0 = cheb_pade_coefs(dx_wl, order, fm.sin(max_propagation_angle*fm.pi/180)**2, 'aaa')
coefs_aaa_num = np.array([a[0] for a in coefs_aaa])
coefs_aaa_den = np.array([b[1] for b in coefs_aaa])

coefs_aaa2, a0 = cheb_pade_coefs(dx_wl, (order[1], order[1]), fm.sin(max_propagation_angle*fm.pi/180)**2, 'aaa')
coefs_aaa_num2 = np.array([a[0] for a in coefs_aaa2])
coefs_aaa_den2 = np.array([b[1] for b in coefs_aaa2])

coefs_chebpade, a0 = cheb_pade_coefs(dx_wl, order, fm.sin(max_propagation_angle*fm.pi/180)**2, 'chebpade')
coefs_chebpade_num = np.array([a[0] for a in coefs_chebpade])
coefs_chebpade_den = np.array([b[1] for b in coefs_chebpade])

coefs_chebpade2, a0 = cheb_pade_coefs(dx_wl, (order[1], order[1]), fm.sin(max_propagation_angle*fm.pi/180)**2, 'chebpade')
coefs_chebpade_num2 = np.array([a[0] for a in coefs_chebpade2])
coefs_chebpade_den2 = np.array([b[1] for b in coefs_chebpade2])

k0 = 2 * fm.pi / wl
kz_arr = np.linspace(0, fm.sin((max_propagation_angle + 10) * fm.pi / 180) * k0, 10000)
angles = [cm.asin(kz / k0).real * 180 / fm.pi for kz in kz_arr]
k_x_r = np.array([cm.sqrt(k0 ** 2 - kz ** 2) for kz in kz_arr])
k_x_ratinterp = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_num, coefs_den, k0, kz_arr)
k_x_pade = k_x_angle(dx_wl * wl, dz_wl * wl, pade_coefs_num, pade_coefs_den, k0, kz_arr)
k_x_aaa = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_aaa_num, coefs_aaa_den, k0, kz_arr)
k_x_chebpade = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_chebpade_num, coefs_chebpade_den, k0, kz_arr)

plt.figure(figsize=(6, 3.2))
plt.plot(angles, (np.abs(k_x_pade - k_x_r)), label='Pade')
plt.plot(angles, (np.abs(k_x_chebpade - k_x_r)), label='Cheb.-Pade')
plt.plot(angles, (np.abs(k_x_aaa - k_x_r)), label='AAA')
plt.plot(angles, (np.abs(k_x_ratinterp - k_x_r)), label=method)
plt.xlabel('Angle (degrees)')
plt.ylabel('k_x abs. error')
plt.xlim([angles[0], angles[-1]])
plt.ylim([1e-10, 1e1])
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

kz_arr = np.linspace(0, 1.1 * k0, 10000)
k_x_ratinterp = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_num, coefs_den, k0, kz_arr)
k_x_pade = k_x_angle(dx_wl * wl, dz_wl * wl, pade_coefs_num, pade_coefs_den, k0, kz_arr)
k_x_ratinterp2 = k_x_angle(dx_wl * wl, dz_wl * wl, coefs2_num, coefs2_den, k0, kz_arr)
k_x_pade2 = k_x_angle(dx_wl * wl, dz_wl * wl, pade_coefs2_num, pade_coefs2_den, k0, kz_arr)

k_x_aaa = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_aaa_num, coefs_aaa_den, k0, kz_arr)
k_x_chebpade = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_chebpade_num, coefs_chebpade_den, k0, kz_arr)
k_x_aaa2 = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_aaa_num2, coefs_aaa_den2, k0, kz_arr)
k_x_chebpade2 = k_x_angle(dx_wl * wl, dz_wl * wl, coefs_chebpade_num2, coefs_chebpade_den2, k0, kz_arr)

plt.figure(figsize=(6, 3.2))
plt.plot(kz_arr/k0, (np.imag(k_x_pade))/k0, label='imag Pade-[6/7]')
plt.plot(kz_arr/k0, (np.imag(k_x_ratinterp))/k0, label='imag ' + method+"-[6/7]")
plt.plot(kz_arr/k0, (np.imag(k_x_pade2))/k0, label='imag Pade-[7/7]')
plt.plot(kz_arr/k0, (np.imag(k_x_ratinterp2))/k0, label='imag ' + method+"-[7/7]")
plt.xlabel('k_z/k0')
plt.ylabel('imag(k_x/k0)')
plt.xlim([kz_arr[0]/k0, kz_arr[-1]/k0])
# plt.ylim([1e-10, 1e-1])
# plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3.2))
plt.plot(kz_arr/k0, (np.real(k_x_pade))/k0, label='real Pade-[6/7]')
plt.plot(kz_arr/k0, (np.real(k_x_ratinterp))/k0, label='real ' + method+"-[6/7]")
plt.plot(kz_arr/k0, (np.real(k_x_pade2))/k0, label='real Pade-[7/7]')
plt.plot(kz_arr/k0, (np.real(k_x_ratinterp2))/k0, label='real ' + method+"-[7/7]")
plt.xlabel('k_z/k0')
plt.ylabel('real(k_x/k0)')
plt.xlim([kz_arr[0]/k0, kz_arr[-1]/k0])
# plt.ylim([1e-10, 1e-1])
# plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 3.2))
plt.plot(kz_arr/k0, (np.imag(k_x_aaa))/k0, label='imag AAA-[6/7]')
plt.plot(kz_arr/k0, (np.imag(k_x_chebpade))/k0, label="imag Cheb.Pade-[6/7]")
plt.plot(kz_arr/k0, (np.imag(k_x_aaa2))/k0, label='imag AAA-[7/7]')
plt.plot(kz_arr/k0, (np.imag(k_x_chebpade2))/k0, label="imag Cheb.Pade-[7/7]")
plt.xlabel('k_z/k0')
plt.ylabel('imag(k_x/k0)')
plt.xlim([kz_arr[0]/k0, kz_arr[-1]/k0])
# plt.ylim([1e-10, 1e-1])
# plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3.2))
plt.plot(kz_arr/k0, (np.real(k_x_aaa))/k0, label='real AAA-[6/7]')
plt.plot(kz_arr/k0, (np.real(k_x_chebpade))/k0, label="real Cheb.Pade-[6/7]")
plt.plot(kz_arr/k0, (np.real(k_x_aaa2))/k0, label='real AAA-[7/7]')
plt.plot(kz_arr/k0, (np.real(k_x_chebpade2))/k0, label="real Cheb.Pade-[7/7]")
plt.xlabel('k_z/k0')
plt.ylabel('real(k_x/k0)')
plt.xlim([kz_arr[0]/k0, kz_arr[-1]/k0])
# plt.ylim([1e-10, 1e-1])
# plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

