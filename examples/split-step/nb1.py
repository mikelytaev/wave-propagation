from rwp.sspade import *
from rwp.vis import *


logging.basicConfig(level=logging.DEBUG)
env = Troposphere()
env.z_max = 300
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 50], fill_value="extrapolate")
env.M_profile = lambda x, z: profile1d(z)
env.knife_edges = [KnifeEdge(range=75e3, height=150)]

ant = GaussAntenna(freq_hz=3000e6, height=30, beam_width=4, eval_angle=0, polarz='H')

pade_task_4 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=150e3, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      modify_grid=True,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=0.8,
                                                      two_way=False,
                                                      storage=PickleStorage()
                                                  ))
pade_field_4 = pade_task_4.calculate()

pade_task_5 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=100e3, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=20,
                                                      modify_grid=True,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(10, 11),
                                                      #dx_wl=500,
                                                      #dz_wl=1,
                                                      two_way=False,
                                                      storage=PickleStorage()
                                                  ))
pade_field_5 = pade_task_5.calculate()

pade_vis_4 = FieldVisualiser(pade_field_4, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Pade+NLBC 4 (Proposed)', x_mult=1E-3)
pade_vis_5 = FieldVisualiser(pade_field_5, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Pade+NLBC 5 (Proposed)', x_mult=1E-3)

plt = pade_vis_4.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade_vis_5.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade_vis_4.plot_hor(5, pade_vis_5)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.grid(True)
plt.tight_layout()
plt.show()