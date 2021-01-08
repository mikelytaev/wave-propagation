from rwp.sspade import *
from rwp.vis import *
from rwp.petool import PETOOLPropagationTask


logging.basicConfig(level=logging.DEBUG)
env_elevated = Troposphere(flat=False)
env_elevated.z_max = 600
env_elevated.terrain = Terrain(ground_material=PerfectlyElectricConducting())
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 50], fill_value="extrapolate")
env_elevated.M_profile = lambda x, z: profile1d(z) + 320

env_evaporation = Troposphere(flat=False)
env_evaporation.z_max = 600
env_evaporation.terrain = Terrain(ground_material=PerfectlyElectricConducting())
env_evaporation.M_profile = lambda x, z: evaporation_duct(30, z)

ant = GaussAntenna(freq_hz=10000e6, height=30, beam_width=3, eval_angle=0, polarz='H')

max_propagation_angle = 3
max_range_m = 200e3

pade_task_joined_elevated = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env_elevated, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=1000,
                                                      x_output_filter=3,
                                                      dz_wl=0.5,
                                                      z_output_filter=6,
                                                      storage=PickleStorage()
                                                  ))
pade_field_joined_elevated = pade_task_joined_elevated.calculate()

pade_task_joined_evaporation = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env_evaporation, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=1000,
                                                      x_output_filter=3,
                                                      dz_wl=0.5,
                                                      z_output_filter=6,
                                                      storage=PickleStorage()
                                                  ))
pade_field_joined_evaporation = pade_task_joined_evaporation.calculate()

petool_task_elevated = PETOOLPropagationTask(antenna=ant, env=env_elevated, two_way=False, max_range_m=max_range_m, dx_wl=1000, n_dx_out=3, dz_wl=3)
petool_field_elevated = petool_task_elevated.calculate()

petool_task_evaporation = PETOOLPropagationTask(antenna=ant, env=env_evaporation, two_way=False, max_range_m=max_range_m, dx_wl=1000, n_dx_out=3, dz_wl=3)
petool_field_evaporation = petool_task_evaporation.calculate()

# pade_vis_2f = FieldVisualiser(pade_field_2f, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
#                              label='2-й порядок по z', x_mult=1E-3)
pade_vis_joined_elevated = FieldVisualiser(pade_field_joined_elevated, env=env_elevated, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
                                           label='Паде аппрокс. по z', x_mult=1E-3)
pade_vis_joined_evaporation = FieldVisualiser(pade_field_joined_evaporation, env=env_evaporation, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)),
                                           label='Паде аппрокс. по z', x_mult=1E-3)
petool_vis_elevated = FieldVisualiser(petool_field_elevated, env=env_elevated, trans_func=lambda x: 2 * x, label='Метод расщ. Фурье', x_mult=1E-3)
petool_vis_evaporation = FieldVisualiser(petool_field_evaporation, env=env_evaporation, trans_func=lambda x: 2 * x, label='Метод расщ. Фурье', x_mult=1E-3)

########################################################

norm = Normalize(-100, 0)
extent = [pade_vis_joined_elevated.x_grid[0], pade_vis_joined_elevated.x_grid[-1], pade_vis_joined_elevated.z_grid[0], pade_vis_joined_elevated.z_grid[-1]]
f, ax = plt.subplots(1, 2, sharey=True, figsize=(6, 3.2), constrained_layout=True)
ax1, ax2 = ax
ax1.imshow(pade_vis_joined_evaporation.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
im = ax2.imshow(pade_vis_joined_elevated.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax1.grid()
ax2.grid()
ax1.set_xlabel('Range (km)')
ax1.set_ylabel('Height (m)')
ax2.set_xlabel('Range (km)')
f.colorbar(im, ax=[ax1, ax2], shrink=0.6, location='bottom')
#f.tight_layout()
#plt.subplots_adjust(bottom=0.15, top=0.966, left=0.103, right=0.910, wspace=0.1)
plt.show()


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 3.2), constrained_layout=True)
err_evaporation = (np.abs(20 * np.log10(np.abs(petool_vis_evaporation.field[:, 0:-1])+1e-16) - 20 * np.log10(np.abs(pade_vis_joined_evaporation.field[:, :]) + 1e-16)))
err_elevated = (np.abs(20 * np.log10(np.abs(petool_vis_elevated.field[:, 0:-1])+1e-16) - 20 * np.log10(np.abs(pade_vis_joined_elevated.field[:, :]) + 1e-16)))
norm = Normalize(0, 10)
extent = [pade_vis_joined_elevated.x_grid[0], pade_vis_joined_elevated.x_grid[-1], pade_vis_joined_elevated.z_grid[0], pade_vis_joined_elevated.z_grid[-1]]
im = ax1.imshow(err_evaporation.T[::-1, :], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('binary'))
ax2.imshow(err_elevated.T[::-1, :], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('binary'))
f.colorbar(im, ax=[ax1, ax2], shrink=0.6, location='bottom')
ax1.grid()
ax2.grid()
ax1.set_xlabel('Range (km)')
ax1.set_ylabel('Height (m)')
ax2.set_xlabel('Range (km)')
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(5, 3.2), constrained_layout=True)
ax1.plot(env_evaporation.M_profile(0, pade_vis_joined_elevated.z_grid), pade_vis_joined_elevated.z_grid)
ax1.set_ylim([pade_vis_joined_elevated.z_grid[0], pade_vis_joined_elevated.z_grid[-1]])
ax2.plot(env_elevated.M_profile(0, pade_vis_joined_elevated.z_grid), pade_vis_joined_elevated.z_grid)
ax1.grid()
ax2.grid()
ax1.set_xlabel('M-units')
ax1.set_ylabel('Height (m)')
ax2.set_xlabel('M-units')
plt.show()