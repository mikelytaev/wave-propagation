from rwp.sspade import *
from rwp.vis import *
from matplotlib.patches import Rectangle


logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
env.z_max = 200
env.terrain = Terrain(ground_material=PerfectlyElectricConducting())
env.knife_edges = [KnifeEdge(range=0.2e3, height=100)]

ant = GaussAntenna(freq_hz=300e6, height=100, beam_width=15, eval_angle=0, polarz='H')

max_propagation_angle = 10
max_range_m = 0.4e3
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
                                                      z_output_filter=20
                                                  ))
etalon_field = etalon_task.calculate()

pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(7, 8),
                                                      dx_wl=dx_wl,
                                                      x_output_filter=1,
                                                      dz_wl=0.3,
                                                      z_output_filter=3
                                                  ))
pade_field = pade_task.calculate()

de_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
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
                                                      z_output_filter=4
                                                  ))
de_field = de_task.calculate()

etalon_vis = FieldVisualiser(etalon_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1)
plt = etalon_vis.plot2d(min=-120, max=0)
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.gca().add_patch(Rectangle((200, 0), 2, 100, edgecolor='black'))
plt.tight_layout()
plt.savefig('knife_edge_de.eps')
#plt.show()

pade_vis = FieldVisualiser(pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1)
plt = pade_vis.plot2d(min=-120, max=0)
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.gca().add_patch(Rectangle((200, 0), 2, 100, edgecolor='black'))
plt.tight_layout()
plt.savefig('knife_edge_pade.eps')
#plt.show()

plt.figure(figsize=(6, 3.2))
norm = Normalize(0, 1)
extent = [etalon_field.x_grid[0], etalon_field.x_grid[-1], etalon_field.z_grid[0], etalon_field.z_grid[-1]]

err = np.abs(20*np.log10(np.abs(etalon_field.field)+1e-16) - 20*np.log10(np.abs(de_field.field)+1e-16))*0.02
plt.imshow(err.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
plt.grid()
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.colorbar(fraction=0.046, pad=0.04)
plt.gca().add_patch(Rectangle((200, 0), 2, 100, edgecolor='black'))
plt.tight_layout()
plt.savefig('knife_edge_error_de_vs_etalon.eps')
#plt.show()