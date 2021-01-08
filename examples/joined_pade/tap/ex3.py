from podpac.datalib.terraintiles import TerrainTiles
from podpac import Coordinates, clinspace
from podpac import settings
import matplotlib.pyplot as plt
from rwp.environment import *
from rwp.terrain import *
from rwp.sspade import *
from rwp.vis import *


settings['DEFAULT_CACHE'] = ['disk']
node = TerrainTiles(tile_format='geotiff', zoom=11)
coords, x_grid = inv_geodesic_problem(21.226706, -158.365515, 21.937204, -157.606865, 5000)
lats = [c[0] for c in coords]
lons = [c[1] for c in coords]
c = Coordinates([lats, lons], dims=['lat', 'lon'])
o = node.eval(c)
eval = np.array([o.data[i, i] for i in range(0, len(x_grid))])
eval = np.array([max(a, 0) for a in eval])

eval_func = interp1d(x=x_grid, y=eval*0.3, fill_value="extrapolate")

plt.plot(x_grid, eval)
plt.show()

env = Troposphere(flat=True)
env.z_max = 500
env.terrain = Terrain(elevation=eval_func, ground_material=PerfectlyElectricConducting())
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 40], fill_value="extrapolate")
env.M_profile = lambda x, z: profile1d(z)

ant = GaussAntenna(freq_hz=1500e6, height=100, beam_width=3, eval_angle=0, polarz='H')

max_propagation_angle = 10
max_range_m = x_grid[-1]
dx_wl = 1

################################################################
logging.basicConfig(level=logging.DEBUG)
joined_pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      #two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=5,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=100,
                                                      x_output_filter=1,
                                                      dz_wl=3,
                                                      z_output_filter=1,
                                                      storage=PickleStorage()
                                                  ))
joined_pade_field = joined_pade_task.calculate()

joined_pade_vis = FieldVisualiser(joined_pade_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v)), x_mult=1E-3,
                                  label='Δz=3.0λ, joined')

pade_2_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      #two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=100,
                                                      x_output_filter=1,
                                                      dz_wl=0.5,
                                                      z_output_filter=6,
                                                      storage=PickleStorage()
                                                  ))
pade_2_field = pade_2_task.calculate()

pade_2_vis = FieldVisualiser(pade_2_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v))-50, x_mult=1E-3,
                             label='Δz=0.5λ, 2nd order')

pade_2_task_f = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=max_range_m, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      #two_way=True,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=100,
                                                      x_output_filter=1,
                                                      dz_wl=3,
                                                      z_output_filter=1,
                                                      storage=PickleStorage()
                                                  ))
pade_2_f_field = pade_2_task_f.calculate()

pade_2_f_vis = FieldVisualiser(pade_2_f_field, env=env, trans_func=lambda v: 20 * cm.log10(1e-16 + abs(v))-100, x_mult=1E-3,
                               label='Δz=3.0λ, 2nd order')

plt = joined_pade_vis.plot2d(min=-160, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.grid(True)
plt.tight_layout()
plt.show()

plt = joined_pade_vis.plot_hor_over_terrain(5, pade_2_vis, pade_2_f_vis)
plt.xlabel('Range (km)')
plt.ylabel('20log(u)')
plt.grid(True)
plt.tight_layout()
plt.show()
