from rwp.sspade import *
from rwp.vis import *
from rwp.petool import PETOOLPropagationTask
import matplotlib


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=False)
env.z_max = 300
env.terrain = Terrain(elevation=lambda x: pyramid(x, 5, 50, 50e3) +
                                          pyramid(x, 5, 100, 70e3) +
                                          pyramid(x, 5, 150, 90e3),
                      ground_material=PerfectlyElectricConducting())
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 50], fill_value="extrapolate")
env.M_profile = lambda x, z: profile1d(z)

ant = GaussAntenna(freq_hz=3000e6, height=150, beam_width=4, eval_angle=0, polarz='H')

max_propagation_angle = 3
max_range_m = 100e3

pade_task_2f = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=3,
                                                      z_output_filter=1,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2f = pade_task_2f.calculate()

pade_task_2 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=0.5,
                                                      z_output_filter=6,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2 = pade_task_2.calculate()

pade_task_joined = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=3,
                                                      storage=PickleStorage()
                                                  ))
pade_field_joined = pade_task_joined.calculate()

petool_task = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=max_range_m, dx_wl=500, n_dx_out=1, dz_wl=3)
petool_field = petool_task.calculate()

pade_vis_2f = FieldVisualiser(pade_field_2f, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='dz=3.0 (2й порядок)', x_mult=1E-3)
pade_vis_2 = FieldVisualiser(pade_field_2, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='dz=0.5 (2й порядок)', x_mult=1E-3)
pade_vis_joined = FieldVisualiser(pade_field_joined, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='dz=3.0 (Паде порядок)', x_mult=1E-3)
petool_vis = FieldVisualiser(petool_field, env=env, trans_func=lambda x: x, label='Метод расщепления Фурье', x_mult=1E-3)
petool_vis.field[0, :] = -160

f, ax = plt.subplots(2, 2, sharey=True, figsize=(6, 2.5*2), constrained_layout=True)
norm = Normalize(-80, 0)
extent = [pade_vis_joined.x_grid[0], pade_vis_joined.x_grid[-1], pade_vis_joined.z_grid[0], pade_vis_joined.z_grid[-1]]
ax[0][0].imshow(pade_vis_joined.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
terrain_grid = np.array([pade_vis_joined.env.terrain.elevation(v) for v in pade_vis_joined.x_grid / pade_vis_joined.x_mult])
ax[0][0].plot(pade_vis_joined.x_grid, terrain_grid, 'k')
ax[0][0].fill_between(pade_vis_joined.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[0][0].grid()
ax[0][0].set_xlim([40, 100])
ax[0][0].set_ylim([0, 170])
ax[0][0].set_title('(а)')
ax[0][0].set_ylabel('Высота, м')

ax[0][1].imshow(pade_vis_2.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[0][1].plot(pade_vis_joined.x_grid, terrain_grid, 'k')
ax[0][1].fill_between(pade_vis_joined.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[0][1].grid()
ax[0][1].set_xlim([40, 100])
ax[0][1].set_ylim([0, 170])
ax[0][1].set_title('(б)')

im = ax[1][0].imshow(pade_vis_2f.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[1][0].plot(pade_vis_joined.x_grid, terrain_grid, 'k')
ax[1][0].fill_between(pade_vis_joined.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[1][0].grid()
ax[1][0].set_title('(в)')
ax[1][0].set_xlim([40, 100])
ax[1][0].set_ylim([0, 170])
ax[1][0].set_xlabel('Расстояние, км')
ax[1][0].set_ylabel('Высота, м')

ax[1][1].imshow(petool_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[1][1].plot(pade_vis_joined.x_grid, terrain_grid, 'k')
ax[1][1].fill_between(pade_vis_joined.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[1][1].grid()
ax[1][1].set_title('(г)')
ax[1][1].set_xlim([40, 100])
ax[1][1].set_ylim([0, 170])
ax[1][1].set_xlabel('Расстояние, км')

f.colorbar(im, ax=ax[1, :], shrink=0.6, location='bottom')
#f.tight_layout()
plt.show()

plt = pade_vis_joined.plot_hor_over_terrain(150, petool_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
#plt.xlim([0.5, max_range_m*1e-3])
plt.ylim([-100, 0])
plt.grid(True)
plt.tight_layout()
plt.show()

matplotlib.rcParams["legend.loc"] = 'upper left'
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 3.2), sharey=True)
pade_vis_joined.plot_ver(72.5 * 1E3, ax1, pade_vis_2f)
ax1.set_xlim([-60, -10])
ax1.set_ylim([0, 100])
ax1.set_ylabel('Высота, м')
ax1.set_xlabel('10log|u| (дБ)')
ax1.set_title("(а)")
ax1.grid()
pade_vis_joined.plot_ver(100 * 1E3, ax2, pade_vis_2f)
ax2.set_xlim([-60, -10])
ax2.set_ylim([0, 100])
ax2.set_xlabel('10log|u| (дБ)')
ax2.set_title("(б)")
ax2.grid()
f.tight_layout()
f.show()