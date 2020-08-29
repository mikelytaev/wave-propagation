from rwp.sspade import *
from rwp.vis import *
#from rwp.petool import PETOOLPropagationTask
from scipy.io import loadmat


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 300
env.terrain = Terrain(elevation=lambda x: pyramid(x, 10, 150, 3e3), ground_material=PerfectlyElectricConducting())

ant = GaussAntenna(freq_hz=3000e6, height=150, beam_width=4, eval_angle=0, polarz='H')

max_propagation_angle = 10
max_range_m = 10e3

pade_task_2 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=100,
                                                      dz_wl=0.05,
                                                      z_output_filter=40,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2 = pade_task_2.calculate()

pade_task_2f = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=100,
                                                      dz_wl=2.0,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2f = pade_task_2f.calculate()

pade_task_joined = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=100,
                                                      dz_wl=2.0,
                                                      storage=PickleStorage()
                                                  ))
pade_field_joined = pade_task_joined.calculate()

pade_vis_2 = FieldVisualiser(pade_field_2, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=100, dz=0.05 (2nd order)', x_mult=1E-3)
pade_vis_2f = FieldVisualiser(pade_field_2f, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=100, dz=2.0 (2nd order)', x_mult=1E-3)
pade_vis_joined = FieldVisualiser(pade_field_joined, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=500, dz=2.0 (Pade)', x_mult=1E-3)


arr = loadmat("prop_fact.mat")
prop_fact = arr['prop_fact_final']
utd_field = deepcopy(pade_field_2)
utd_field.field = (np.array(prop_fact).T - 10*fm.log10(0.1) - np.tile(10 * np.log10(utd_field.x_grid), (utd_field.z_grid.shape[0], 1)).T*0) / 2 - 10 + 3.2
utd_vis = FieldVisualiser(utd_field, env=env, trans_func=lambda x: x, label='GO+UTD', x_mult=1E-3)

f, ax = plt.subplots(2, 2, sharey=True, figsize=(6, 2.5*2), constrained_layout=True)
norm = Normalize(-70, 0)
extent = [pade_vis_joined.x_grid[0], pade_vis_joined.x_grid[-1], pade_vis_joined.z_grid[0], pade_vis_joined.z_grid[-1]]
ax[0][0].imshow(utd_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
terrain_grid = np.array([utd_vis.env.terrain.elevation(v) for v in utd_vis.x_grid / utd_vis.x_mult])
ax[0][0].plot(utd_vis.x_grid, terrain_grid, 'k')
ax[0][0].fill_between(utd_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[0][0].grid()
ax[0][0].set_title('(а)')
ax[0][0].set_ylabel('Высота, м')

ax[0][1].imshow(pade_vis_joined.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[0][1].plot(utd_vis.x_grid, terrain_grid, 'k')
ax[0][1].fill_between(utd_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[0][1].grid()
ax[0][1].set_title('(б)')

ax[1][0].imshow(pade_vis_2.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[1][0].plot(utd_vis.x_grid, terrain_grid, 'k')
ax[1][0].fill_between(utd_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[1][0].grid()
ax[1][0].set_title('(в)')
ax[1][0].set_xlabel('Расстояние, км')
ax[1][0].set_ylabel('Высота, м')

im = ax[1][1].imshow(pade_vis_2f.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[1][1].plot(utd_vis.x_grid, terrain_grid, 'k')
ax[1][1].fill_between(utd_vis.x_grid, terrain_grid*0, terrain_grid, color='brown')
ax[1][1].grid()
ax[1][1].set_title('(г)')
ax[1][1].set_xlabel('Расстояние, км')
#ax[0][1].set_ylabel('Высота, м')

f.colorbar(im, ax=ax[1, :], shrink=0.6, location='bottom')
#f.tight_layout()
plt.show()
