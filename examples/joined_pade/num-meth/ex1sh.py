from rwp.sspade import *
from rwp.vis import *
from rwp.tworay import *
from rwp.petool import PETOOLPropagationTask
from scipy.io import loadmat


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 300
env.terrain = Terrain(ground_material=PerfectlyElectricConducting())

ant = GaussAntenna(freq_hz=3000e6, height=150, beam_width=0.5, eval_angle=10, polarz='H')

max_propagation_angle = 10
max_range_m = 3.0e3

pade_task_2_0_25 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=100,
                                                      dz_wl=0.25,
                                                      z_output_filter=8,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2_0_25 = pade_task_2_0_25.calculate()

pade_task_2_0_5 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=100,
                                                      dz_wl=0.5,
                                                      z_output_filter=4,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2_0_5 = pade_task_2_0_5.calculate()

pade_task_2_1 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=100,
                                                      dz_wl=1,
                                                      z_output_filter=2,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2_1 = pade_task_2_1.calculate()

pade_task_2_2 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=100,
                                                      dz_wl=2,
                                                      z_output_filter=1,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2_2 = pade_task_2_2.calculate()

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

pade_task_cn = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(1, 1),
                                                      dx_wl=100/40,
                                                      x_output_filter=40,
                                                      dz_wl=0.25,
                                                      storage=PickleStorage()
                                                  ))
pade_field_cn = pade_task_cn.calculate()

petool_task_4 = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=max_range_m, dx_wl=100, dz_wl=4)
petool_field_4 = petool_task_4.calculate()

petool_task_2 = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=max_range_m, dx_wl=100, dz_wl=2)
petool_field_2 = petool_task_2.calculate()

trm = TwoRayModel(src=ant, env=env)
x_grid_m = pade_field_joined.x_grid
z_grid_m = pade_field_joined.z_grid
trm_f = trm.calculate(x_grid_m, z_grid_m)
trm_field = Field(x_grid=x_grid_m, z_grid=z_grid_m, freq_hz=ant.freq_hz)
trm_field.field[:, :] = trm_f

pade_vis_2_0_25 = FieldVisualiser(pade_field_2_0_25, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=100, dz=0.25 (2nd order)', x_mult=1E-3)
pade_vis_2_0_5 = FieldVisualiser(pade_field_2_0_5, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=100, dz=0.5 (2nd order)', x_mult=1E-3)
pade_vis_2_1 = FieldVisualiser(pade_field_2_1, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=100, dz=1.0 (2nd order)', x_mult=1E-3)
pade_vis_2_2 = FieldVisualiser(pade_field_2_2, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=100, dz=2.0 (2nd order)', x_mult=1E-3)
pade_vis_joined = FieldVisualiser(pade_field_joined, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=500, dz=2.0 (Pade)', x_mult=1E-3)
pade_vis_cn = FieldVisualiser(pade_field_cn, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=500, dz=2.0 (Pade)', x_mult=1E-3)
trm_vis = FieldVisualiser(trm_field, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v))-10,
                          label='Two-ray model', x_mult=1E-3)
petool_vis_4 = FieldVisualiser(petool_field_4, env=env, trans_func=lambda x: x, label='Метод расщепления Фурье', x_mult=1E-3)
petool_vis_2 = FieldVisualiser(petool_field_2, env=env, trans_func=lambda x: x, label='Метод расщепления Фурье', x_mult=1E-3)

f, ax = plt.subplots(4, 2, sharey=True, figsize=(6, 2.0*4), constrained_layout=True)
norm = Normalize(-50, 0)
extent = [pade_vis_joined.x_grid[0], pade_vis_joined.x_grid[-1], pade_vis_joined.z_grid[0], pade_vis_joined.z_grid[-1]]
ax[0][0].imshow(trm_vis.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[0][0].grid()
ax[0][0].set_title('(а)')
ax[0][0].set_ylabel('Высота, м')

ax[0][1].imshow(pade_vis_joined.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[0][1].grid()
ax[0][1].set_title('(б)')

ax[1][0].imshow(pade_vis_2_0_25.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[1][0].grid()
ax[1][0].set_title('(в)')
#ax[1][0].set_xlabel('Расстояние, км')
ax[1][0].set_ylabel('Высота, м')

ax[1][1].imshow(pade_vis_2_0_5.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[1][1].grid()
ax[1][1].set_title('(г)')
#ax[1][1].set_xlabel('Расстояние, км')
#ax[0][1].set_ylabel('Высота, м')

ax[2][0].imshow(pade_vis_2_1.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[2][0].grid()
ax[2][0].set_title('(д)')
ax[2][0].set_ylabel('Высота, м')

im = ax[2][1].imshow(pade_vis_2_2.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[2][1].grid()
ax[2][1].set_title('(е)')
#ax[0][1].set_ylabel('Высота, м')

ax[3][0].imshow(petool_vis_4.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[3][0].grid()
ax[3][0].set_title('(ж)')
ax[3][0].set_xlabel('Расстояние, км')
ax[3][0].set_ylabel('Высота, м')

ax[3][1].imshow(petool_vis_2.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax[3][1].grid()
ax[3][1].set_title('(з)')
ax[3][1].set_xlabel('Расстояние, км')
#ax[0][1].set_ylabel('Высота, м')

f.colorbar(im, ax=ax[3, :], shrink=0.6, location='bottom')
#f.tight_layout()
plt.show()

plt = pade_vis_cn.plot2d(min=-50, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()