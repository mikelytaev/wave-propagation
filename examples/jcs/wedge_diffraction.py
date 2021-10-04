from rwp.sspade import *
from rwp.vis import *
from scipy.io import loadmat
from examples.chebyshev_pade.cheb_pade_coefs import *


logging.basicConfig(level=logging.DEBUG)

max_propagation_angle = 20
dx_wl = 50
wl = 0.1
max_range_m = 10e3
coefs, a0 = cheb_pade_coefs(dx_wl, (6, 7), fm.sin(max_propagation_angle*fm.pi/180)**2, 'ratinterp')

env = Troposphere(flat=True)
env.z_max = 300
p = lambda x: pyramid2(x, 5, 150, 4e3)
x_grid = np.arange(0, max_range_m, dx_wl * wl*10)
p_grid = [p(x) for x in x_grid]
interp_f = interp1d(x_grid, p_grid, kind='previous', fill_value='extrapolate')
interp_ff = lambda x : interp_f([x])[0]
env.terrain = Terrain(elevation=interp_ff, ground_material=PerfectlyElectricConducting())
#env.knife_edges = [KnifeEdge(range=3e3, height=150)]

ant = GaussAntenna(wavelength=wl, height=150, beam_width=4, eval_angle=0, polarz='H')


etalon_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      z_order=5,
                                                      exp_pade_order=(9, 10),
                                                      dx_wl=dx_wl/2,
                                                      x_output_filter=2,
                                                      dz_wl=0.25,
                                                      z_output_filter=8,
                                                      two_way=False
                                                  ))
etalon_field = etalon_task.calculate()


# pade_task_f = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
#                                                   HelmholtzPropagatorComputationalParams(
#                                                       terrain_method=TerrainMethod.staircase,
#                                                       max_propagation_angle=max_propagation_angle,
#                                                       modify_grid=False,
#                                                       z_order=4,
#                                                       exp_pade_order=(6, 7),
#                                                       dx_wl=dx_wl,
#                                                       x_output_filter=4,
#                                                       dz_wl=0.2,
#                                                       z_output_filter=20,
#                                                       two_way=False
#                                                   ))
# pade_field_f = pade_task_f.calculate()
#
#
# pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
#                                                   HelmholtzPropagatorComputationalParams(
#                                                       terrain_method=TerrainMethod.staircase,
#                                                       max_propagation_angle=max_propagation_angle,
#                                                       modify_grid=False,
#                                                       z_order=4,
#                                                       exp_pade_order=(6, 7),
#                                                       dx_wl=dx_wl,
#                                                       x_output_filter=1,
#                                                       dz_wl=0.1,
#                                                       z_output_filter=20,
#                                                       two_way=False
#                                                   ))
# pade_field = pade_task.calculate()

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
                                                      dz_wl=0.25,
                                                      z_output_filter=8,
                                                      two_way=False
                                                  ))
cheb_pade_field = cheb_pade_task.calculate()

# from rwp.petool import PETOOLPropagationTask
# petool_task_elevated = PETOOLPropagationTask(antenna=ant,
#                                              env=env,
#                                              two_way=False,
#                                              max_range_m=max_range_m,
#                                              dx_wl=dx_wl,
#                                              n_dx_out=1,
#                                              dz_wl=0.1,
#                                              n_dz_out=20)
# petool_field_elevated = petool_task_elevated.calculate()

etalon_vis = FieldVisualiser(etalon_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[7/8]', x_mult=1E-3)

# pade_vis_f = FieldVisualiser(pade_field_f, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
#                              label='Pade-[7/7]', x_mult=1E-3)
#
# pade_vis = FieldVisualiser(pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
#                              label='Pade-[7/7]', x_mult=1E-3)

cheb_pade_vis = FieldVisualiser(cheb_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
                                label='AAA-[7/7]', x_mult=1E-3)

# f, ax = plt.subplots(1, 3, sharey=True, figsize=(6, 3.2), constrained_layout=True)
# norm = Normalize(-120, -20)
# extent = [pade_vis.x_grid[0], pade_vis.x_grid[-1], pade_vis.z_grid[0], pade_vis.z_grid[-1]]
#
# im = ax[0].imshow(pade_vis_f.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
# terrain_grid = np.array([pade_vis_f.env.terrain.elevation(v) for v in pade_vis_f.x_grid / pade_vis_f.x_mult])
# ax[0].plot(pade_vis_f.x_grid, terrain_grid, 'k')
# ax[0].fill_between(pade_vis_f.x_grid, terrain_grid*0, terrain_grid, color='brown')
# ax[0].grid()
# ax[0].set_title(pade_vis_f.label)
# ax[0].set_xlabel('Range (km)')
# ax[0].set_ylabel('Height (m)')
# ax[0].set_xlim([2.5, 5.5])
# ax[0].set_ylim([0, 200])
#
# ax[1].imshow(pade_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
# terrain_grid = np.array([pade_vis.env.terrain.elevation(v) for v in pade_vis.x_grid / pade_vis.x_mult])
# ax[1].plot(pade_vis.x_grid, terrain_grid, 'k')
# ax[1].fill_between(pade_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
# ax[1].grid()
# ax[1].set_title(pade_vis.label)
# ax[1].set_xlabel('Range (km)')
# ax[1].set_xlim([2.5, 5.5])
# ax[1].set_ylim([0, 200])
#
# ax[2].imshow(cheb_pade_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
# terrain_grid = np.array([cheb_pade_vis.env.terrain.elevation(v) for v in cheb_pade_vis.x_grid / cheb_pade_vis.x_mult])
# ax[2].plot(cheb_pade_vis.x_grid, terrain_grid, 'k')
# ax[2].fill_between(cheb_pade_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
# ax[2].grid()
# ax[2].set_title('AAA- [7/7]')
# ax[2].set_xlabel('Range (km)')
# ax[2].set_xlim([2.5, 5.5])
# ax[2].set_ylim([0, 200])
#
# f.colorbar(im, ax=ax[:], shrink=0.6, location='bottom')
# plt.show()

import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 'medium'
f, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 6), constrained_layout=True)
norm = Normalize(0, 5)
extent = [etalon_field.x_grid[0]*1e-3, etalon_field.x_grid[-1]*1e-3, etalon_field.z_grid[0], etalon_field.z_grid[-1]]

err = np.abs(20*np.log10(np.abs(etalon_field.field[:,:])+1e-16) - 20*np.log10(np.abs(cheb_pade_field.field[:,:])+1e-16))
im = ax.imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax.grid()
ax.set_title('Δz=2.0λ, 2nd order')
ax.set_ylabel('Height (m)')

f.colorbar(im, shrink=0.6, location='bottom')
#f.tight_layout()
plt.show()

# mpl.rcParams['axes.titlesize'] = 'medium'
# f, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 6), constrained_layout=True)
# norm = Normalize(0, 5)
# extent = [etalon_field.x_grid[0]*1e-3, etalon_field.x_grid[-1]*1e-3, etalon_field.z_grid[0], etalon_field.z_grid[-1]]
#
# err = np.abs(2*petool_field_elevated.field - 20*np.log10(np.abs(cheb_pade_field.field[0:-1,:])+1e-16))
# im = ax.imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
# ax.grid()
# ax.set_title('Δz=2.0λ, 2nd order')
# ax.set_ylabel('Height (m)')
#
# f.colorbar(im, shrink=0.6, location='bottom')
# #f.tight_layout()
# plt.show()

# petool_vis = FieldVisualiser(petool_field_elevated, env=env, trans_func=lambda x: 2 * x, label='Метод расщ. Фурье', x_mult=1E-3)
# plt = petool_vis.plot2d(0, -100, True)
# plt.show()

plt = cheb_pade_vis.plot_hor_over_terrain(5, etalon_vis)
plt.xlabel('Range (km)')
plt.ylabel('20log|u| (dB)')
plt.xlim([2, 8])
plt.ylim([-120, -20])
plt.grid(True)
plt.tight_layout()
plt.show()