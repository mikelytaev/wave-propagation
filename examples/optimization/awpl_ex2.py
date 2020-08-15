import numpy as np
import cmath as cm
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pyproj


bounds = (29.636637, 60.112502, 31.288706, 60.699130)
west, south, east, north = bounds

import pickle

with open('elevation.pickle', 'rb') as f:
    elevation = pickle.load(f)

geod = pyproj.Geod(ellps='WGS84')

azimuth1, azimuth2, distance = geod.inv(west, south, east, north)

long_grid = np.linspace(west, east, elevation.shape[0])
lat_grid = np.linspace(north, south, elevation.shape[1])

elev_int = interp2d(long_grid, lat_grid, elevation.T)

def elev_int_1d(x):
    if not 0 < x < distance:
        return 0.0
    v = x / distance
    return max(elev_int(west + (east - west) * v, south + (north - south) * v)[0], 0)


from rwp.sspade import *
from rwp.vis import *

logging.basicConfig(level=logging.DEBUG)
env = Troposphere()
env.z_max = 1000
env.terrain = Terrain(elevation=elev_int_1d, ground_material=FreshWater())
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 40], fill_value="extrapolate")
env.M_profile = lambda x, z: profile1d(z)

ant = GaussAntenna(freq_hz=3000e6, height=70, beam_width=4, eval_angle=0, polarz='H')

pade_task_4 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=150e3, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=0.8,
                                                      storage=PickleStorage()
                                                  ))
pade_field_4 = pade_task_4.calculate()

pade_task_2 = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=150e3, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=0.1,
                                                      z_output_filter=8,
                                                      inv_z_transform_rtol=1e-4,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2 = pade_task_2.calculate()

pade_task_2f = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=150e3, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=2,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      dz_wl=0.8,
                                                      storage=PickleStorage()
                                                  ))
pade_field_2f = pade_task_2f.calculate()

pade_vis_4 = FieldVisualiser(pade_field_4, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=500, dz=0.8 (4th order)', x_mult=1E-3)
pade_vis_2 = FieldVisualiser(pade_field_2, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=500, dz=0.1 (2th order)', x_mult=1E-3)
pade_vis_2f = FieldVisualiser(pade_field_2f, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                             label='Pade-[10/11], dx=500, dz=0.8 (2th order)', x_mult=1E-3)

plt = pade_vis_2.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 3.2))
err = (np.abs(10*np.log10(np.abs(pade_field_4.field)+1e-16) - 10*np.log10(np.abs(pade_field_2.field)+1e-16)))
np.max(err)
norm = Normalize(0, 5)
extent = [pade_vis_4.x_grid[0], pade_vis_4.x_grid[-1], pade_vis_4.z_grid[0], pade_vis_4.z_grid[-1]]
ax1.imshow(err.T[::-1, :], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('binary'))
#plt.colorbar(fraction=0.046, pad=0.04)
terrain_grid = np.array([pade_vis_4.env.terrain.elevation(v) for v in pade_vis_4.x_grid / pade_vis_4.x_mult])
ax1.plot(pade_vis_4.x_grid, terrain_grid, 'k')
ax1.fill_between(pade_vis_4.x_grid, terrain_grid*0, terrain_grid, color='black')

err = np.abs(10*np.log10(np.abs(pade_field_4.field)+1e-16) - 10*np.log10(np.abs(pade_field_2f.field)+1e-16))
np.max(err)
norm = Normalize(0, 5)
extent = [pade_vis_4.x_grid[0], pade_vis_4.x_grid[-1], pade_vis_4.z_grid[0], pade_vis_4.z_grid[-1]]
im2 = ax2.imshow(err.T[::-1, :], extent=extent, aspect='auto', norm=norm, cmap=plt.get_cmap('binary'))
f.colorbar(im2, fraction=0.046, pad=0.04)
terrain_grid = np.array([pade_vis_4.env.terrain.elevation(v) for v in pade_vis_4.x_grid / pade_vis_4.x_mult])
ax2.plot(pade_vis_4.x_grid, terrain_grid, 'k')
ax2.fill_between(pade_vis_4.x_grid, terrain_grid*0, terrain_grid, color='black')
ax1.set_xlabel('Range (km)')
ax1.set_ylabel('Height (m)')
ax2.set_xlabel('Range (km)')
f.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.966, left=0.118, right=0.940, wspace=0.1)
plt.show()