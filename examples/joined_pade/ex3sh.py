from rwp.sspade import *
from rwp.vis import *
#from rwp.petool import PETOOLPropagationTask


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=False)
env.z_max = 300
env.terrain = Terrain(ground_material=PerfectlyElectricConducting())
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 50], fill_value="extrapolate")
env.M_profile = lambda x, z: profile1d(z)

ant = GaussAntenna(freq_hz=3000e6, height=150, beam_width=3, eval_angle=0, polarz='H')

max_propagation_angle = 3
max_range_m = 200e3

pade_task_2f = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=1000,
                                                      dz_wl=0.5,
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
                                                      dx_wl=1000,
                                                      dz_wl=0.5,
                                                      storage=PickleStorage()
                                                  ))
pade_field_joined = pade_task_joined.calculate()

pade_vis_2f = FieldVisualiser(pade_field_2f, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='2nd order', x_mult=1E-3)
pade_vis_joined = FieldVisualiser(pade_field_joined, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade order', x_mult=1E-3)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 2.5), constrained_layout=True)
norm = Normalize(-70, 0)
extent = [pade_vis_joined.x_grid[0], pade_vis_joined.x_grid[-1], pade_vis_joined.z_grid[0], pade_vis_joined.z_grid[-1]]
ax1.imshow(pade_vis_2f.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax1.grid()
ax1.set_title('(а)')
ax1.set_ylabel('Высота, м')
ax1.set_xlabel('Расстояние, км')

im = ax2.imshow(pade_vis_joined.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
ax2.grid()
ax2.set_title('(б)')
ax2.set_xlabel('Расстояние, км')

f.colorbar(im, ax=[ax1, ax2], shrink=0.6, location='bottom')
#f.tight_layout()
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
pade_vis_joined.plot_ver(72.5 * 1E3, ax1, pade_vis_2f)
ax1.set_xlim([-50, -10])
ax1.set_ylim([0, 300])
ax1.set_ylabel('Height (m)')
ax1.set_xlabel('10log|u| (dB)')
ax1.grid()
pade_vis_joined.plot_ver(100 * 1E3, ax2, pade_vis_2f)
ax2.set_xlim([-50, -10])
ax2.set_ylim([0, 300])
ax2.set_ylabel('Height (m)')
ax2.set_xlabel('10log|u| (dB)')
ax2.grid()
#f.tight_layout()
f.show()