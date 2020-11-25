from rwp.sspade import *
from rwp.vis import *
from scipy.io import loadmat


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 300
max_propagation_angle = 20
env.terrain = Terrain(elevation=lambda x: pyramid(x, max_propagation_angle, 150, 3e3), ground_material=PerfectlyElectricConducting())

ant = GaussAntenna(freq_hz=3000e6, height=150, beam_width=4, eval_angle=0, polarz='H')

max_range_m = 10e3

pade_task_2 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=max_propagation_angle,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=50,
                                                      dz_wl=0.25,
                                                      z_output_filter=8,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2 = pade_task_2.calculate()

pade_vis_2 = FieldVisualiser(pade_field_2, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=100, dz=0.25 (2nd order)', x_mult=1E-3)

plt = pade_vis_2.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.grid(True)
plt.tight_layout()
plt.show()