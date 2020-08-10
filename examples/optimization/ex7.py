from rwp.sspade import *
from rwp.vis import *
from rwp.petool import PETOOLPropagationTask


logging.basicConfig(level=logging.DEBUG)
env = Troposphere()
env.terrain = Terrain(ground_material=SaltWater())
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 50], fill_value="extrapolate")
env.M_profile = lambda x, z: profile1d(z)
env.z_max = 1000

max_range_m = 300E3

ant = GaussAntenna(freq_hz=3000e6, height=70, beam_width=2, eval_angle=0, polarz='H')

pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=3,
                                                      #modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      #dz_wl=0.1,
                                                      storage=PickleStorage()
                                                  ))
pade_field = pade_task.calculate()

pade_task_5 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=3,
                                                      #modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      #dz_wl=0.1,
                                                      storage=PickleStorage()
                                                  ))
pade_field_5 = pade_task_5.calculate()

petool_task = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=max_range_m, dx_wl=500, n_dx_out=1, dz_wl=1)
petool_field = petool_task.calculate()

# env.z_max = 3000
# petool_task_m = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=150e3, dx_wl=500, n_dx_out=1,
#                                       dz_wl=3, n_dz_out=2)
# petool_field_m = petool_task_m.calculate()

pade_vis = FieldVisualiser(pade_field, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Pade', x_mult=1E-3)
pade_vis_5 = FieldVisualiser(pade_field, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Pade M', x_mult=1E-3)
petool_vis = FieldVisualiser(petool_field, env=env, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)
#petool_vis_m = FieldVisualiser(petool_field_m, env=env, trans_func=lambda x: x, label='SSF (PETOOL) z_max=3000 m', x_mult=1E-3)

plt = pade_vis.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade_vis_5.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = petool_vis.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

# plt = petool_vis_m.plot2d(min=-100, max=0, show_terrain=True)
# plt.xlabel('Range (km)')
# plt.ylabel('Height (m)')
# plt.ylim([0, 300])
# plt.tight_layout()
# plt.show()

plt = petool_vis.plot_hor_over_terrain(ant.height_m, pade_vis_5, pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.xlim([0.5, max_range_m*1e-3])
plt.ylim([-100, 0])
plt.grid(True)
plt.tight_layout()
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
pade_vis.plot_ver(10 * 1E3, ax1, pade_vis_5, petool_vis)
ax1.set_ylabel('Height (m)')
ax1.set_xlabel('10log|u| (dB)')

pade_vis.plot_ver(100 * 1E3, ax2, pade_vis_5, petool_vis)
ax2.set_ylabel('Height (m)')
ax2.set_xlabel('10log|u| (dB)')
f.tight_layout()
f.show()