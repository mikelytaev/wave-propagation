from rwp.sspade import *
from rwp.vis import *
#from rwp.petool import PETOOLPropagationTask
from scipy.io import loadmat


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 300
env.terrain = Terrain(elevation=lambda x: pyramid(x, 10, 150, 3e3), ground_material=PerfectlyElectricConducting())

ant = GaussAntenna(freq_hz=3000e6, height=150, beam_width=4, eval_angle=6, polarz='H')

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
                                                      dz_wl=2,
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

plt = pade_vis_2.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade_vis_2f.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade_vis_joined.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = utd_vis.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade_vis_2.plot_hor_over_terrain(1, pade_vis_2f, pade_vis_joined, utd_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.xlim([0.5, max_range_m*1e-3])
plt.ylim([-100, 0])
plt.grid(True)
plt.tight_layout()
plt.show()

f, (ax1) = plt.subplots(1, 1, sharey=True)
pade_vis_2.plot_ver(5.2 * 1E3, ax1, pade_vis_2f, pade_vis_joined, utd_vis)
ax1.set_ylabel('Height (m)')
ax1.set_xlabel('10log|u| (dB)')
ax1.grid()
f.tight_layout()
f.show()

f, (ax1) = plt.subplots(1, 1, sharey=True)
pade_vis_2.plot_ver(150 * 1E3, ax1, pade_vis_2f, pade_vis_joined, utd_vis)
ax1.set_ylabel('Height (m)')
ax1.set_xlabel('10log|u| (dB)')
ax1.grid()
f.tight_layout()
f.show()

plt.figure(figsize=(6, 3.2))
err = np.abs(10*np.log10(np.abs(pade_field_2f.field)+1e-16) - 10*np.log10(np.abs(pade_field_joined.field)+1e-16))
np.max(err)
norm = Normalize(0, 5)
extent = [pade_vis_2.x_grid[0], pade_vis_2.x_grid[-1], pade_vis_2.z_grid[0], pade_vis_2.z_grid[-1]]
plt.imshow(err.T[::-1, :], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('binary'))
plt.colorbar(fraction=0.046, pad=0.04)
terrain_grid = np.array([pade_vis_2.env.terrain.elevation(v) for v in pade_vis_2.x_grid / pade_vis_2.x_mult])
plt.plot(pade_vis_2.x_grid, terrain_grid, 'k')
plt.fill_between(pade_vis_2.x_grid, terrain_grid*0, terrain_grid, color='brown')
plt.show()
