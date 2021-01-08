from rwp.sspade import *
from rwp.vis import *
import matplotlib as mpl


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 300
env.terrain = Terrain(ground_material=PerfectlyElectricConducting())
env.knife_edges = [KnifeEdge(range=1.5e3, height=100)]

ant = GaussAntenna(freq_hz=600e6, height=100, beam_width=3, eval_angle=0, polarz='H')

max_propagation_angle = 10
max_range_m = 3.0e3
dx_wl = 1

################################################################

etalon_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=0.05,
                                                      z_output_filter=40,
                                                      storage=PickleStorage()
                                                  ))
etalon_field = etalon_task.calculate()

pade_task_2_2_0 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=2.0,
                                                      z_output_filter=1,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2_2_0 = pade_task_2_2_0.calculate()

pade_task_2_1_0 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=1.0,
                                                      z_output_filter=2,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2_1_0 = pade_task_2_1_0.calculate()

pade_task_2_0_5 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=0.5,
                                                      z_output_filter=4,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2_0_5 = pade_task_2_0_5.calculate()

pade_task_2_0_25 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=0.25,
                                                      z_output_filter=8,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2_0_25 = pade_task_2_0_25.calculate()


pade_task_4_2_0 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=2.0,
                                                      z_output_filter=1,
                                                      storage=PickleStorage()
                                                  ))
pade_field_4_2_0 = pade_task_4_2_0.calculate()

pade_task_4_1_0 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=1.0,
                                                      z_output_filter=2,
                                                      storage=PickleStorage()
                                                  ))
pade_field_4_1_0 = pade_task_4_1_0.calculate()

pade_task_4_0_5 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=0.5,
                                                      z_output_filter=4,
                                                      storage=PickleStorage()
                                                  ))
pade_field_4_0_5 = pade_task_4_0_5.calculate()

pade_task_4_0_25 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=0.25,
                                                      z_output_filter=8,
                                                      storage=PickleStorage()
                                                  ))
pade_field_4_0_25 = pade_task_4_0_25.calculate()


pade_task_5_2_0 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=2.0,
                                                      z_output_filter=1,
                                                      storage=PickleStorage()
                                                  ))
pade_field_5_2_0 = pade_task_5_2_0.calculate()

pade_task_5_1_0 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=1.0,
                                                      z_output_filter=2,
                                                      storage=PickleStorage()
                                                  ))
pade_field_5_1_0 = pade_task_5_1_0.calculate()

pade_task_5_0_5 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=0.5,
                                                      z_output_filter=4,
                                                      storage=PickleStorage()
                                                  ))
pade_field_5_0_5 = pade_task_5_0_5.calculate()

pade_task_5_0_25 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=0.25,
                                                      z_output_filter=8,
                                                      storage=PickleStorage()
                                                  ))
pade_field_5_0_25 = pade_task_5_0_25.calculate()

etalon_vis = FieldVisualiser(etalon_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3)
plt = etalon_vis.plot2d(min=-120, max=0)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

###############################################################

mpl.rcParams['axes.titlesize'] = 'medium'
f, ax = plt.subplots(3, 4, sharey=True, figsize=(8, 6), constrained_layout=True)
norm = Normalize(0, 20)
extent = [pade_field_2_0_25.x_grid[0]*1e-3, pade_field_2_0_25.x_grid[-1]*1e-3, pade_field_2_0_25.z_grid[0], pade_field_2_0_25.z_grid[-1]]

err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(pade_field_2_2_0.field)+1e-16))
ax[0][0].imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[0][0].grid()
ax[0][0].set_title('Δz=2.0λ, 2nd order')
ax[0][0].set_ylabel('Height (m)')

err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(pade_field_2_1_0.field)+1e-16))
ax[0][1].imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[0][1].grid()
ax[0][1].set_title('Δz=1.0λ, 2nd order')

err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(pade_field_2_0_5.field)+1e-16))
ax[0][2].imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[0][2].grid()
ax[0][2].set_title('Δz=0.5λ, 2nd order')

err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(pade_field_2_0_25.field)+1e-16))
ax[0][3].imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[0][3].grid()
ax[0][3].set_title('Δz=0.25λ, 2nd order')


err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(pade_field_4_2_0.field)+1e-16))
ax[1][0].imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[1][0].grid()
ax[1][0].set_title('Δz=2.0λ, 4th order')
ax[1][0].set_ylabel('Height (m)')

err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(pade_field_4_1_0.field)+1e-16))
ax[1][1].imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[1][1].grid()
ax[1][1].set_title('Δz=1.0λ, 4th order')

err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(pade_field_4_0_5.field)+1e-16))
ax[1][2].imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[1][2].grid()
ax[1][2].set_title('Δz=0.5λ, 4th order')

err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(pade_field_4_0_25.field)+1e-16))
ax[1][3].imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[1][3].grid()
ax[1][3].set_title('Δz=0.25λ, 4th order')


err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(pade_field_5_2_0.field)+1e-16))
ax[2][0].imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[2][0].grid()
ax[2][0].set_title('Δz=2.0λ, joined')
ax[2][0].set_xlabel('Range (km)')
ax[2][0].set_ylabel('Height (m)')

err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(pade_field_5_1_0.field)+1e-16))
ax[2][1].imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[2][1].grid()
ax[2][1].set_title('Δz=1.0λ, joined')
ax[2][1].set_xlabel('Range (km)')

err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(pade_field_5_0_5.field)+1e-16))
ax[2][2].imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[2][2].grid()
ax[2][2].set_title('Δz=0.5λ, joined')
ax[2][2].set_xlabel('Range (km)')

err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(pade_field_5_0_25.field)+1e-16))
im = ax[2][3].imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
ax[2][3].grid()
ax[2][3].set_title('Δz=0.25λ, joined')
ax[2][3].set_xlabel('Range (km)')

f.colorbar(im, ax=ax[2, :], shrink=0.6, location='bottom')
#f.tight_layout()
plt.show()