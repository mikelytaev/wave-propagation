#import elevation
import numpy as np
import cmath as cm
#from osgeo import gdal
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pyproj

#bounds = (-158.4, 21, -157.4, 22)
bounds = (29.636637, 60.112502, 31.288706, 60.699130)
west, south, east, north = bounds
#elevation.clip(bounds=bounds, output='lo.tif')
#elevation.clean()

# gdal.UseExceptions()

# ds = gdal.Open('C:\\Users\\Mikhail\\PycharmProjects\\wave-propagation\\lo.tif')
# band = ds.GetRasterBand(1)
# elevation = band.ReadAsArray()

import pickle
# with open('elevation.pickle', 'wb') as f:
#     pickle.dump(elevation, f)

with open('elevation.pickle', 'rb') as f:
    elevation = pickle.load(f)

# plt.imshow(elevation, cmap='gist_earth', norm=Normalize(0, 1000))
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.show()

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

# size = elevation.shape[0]
# t = np.array([elev_int(west + (east - west) * v, south + (north - south) * v) for v in np.linspace(0, 1, size)])
# t = np.array([v if v > 0 else 0 for v in t])
# x_grid = np.linspace(0, distance, size) * 1e-3
# plt.plot(x_grid, t)
# plt.show()

# tt = (t[1::] - t[0:-1:]) / ((x_grid[1] - x_grid[0]) * 1e3)
# ttt = np.array([cm.atan(v) * 180 / cm.pi for v in tt])
# plt.plot(abs(ttt))
# plt.show()

from rwp.sspade import *
from rwp.vis import *
from rwp.petool import PETOOLPropagationTask

logging.basicConfig(level=logging.DEBUG)
env = Troposphere()
env.ground_material = FreshWater()
env.z_max = 300
env.terrain = Terrain(elev_int_1d)
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 50], fill_value="extrapolate")
#env.vegetation = [Impediment(x1=36e3, x2=101e3, height=18, material=CustomMaterial(eps=1.004, sigma=180e-6))]
env.M_profile = lambda x, z: profile1d(z)

ant = GaussAntenna(freq_hz=3000e6, height=70, beam_width=4, eval_angle=0, polarz='H')

pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=150e3, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      modify_grid=False,
                                                      grid_optimizator_abs_threshold=5e-3,
                                                      z_order=4,
                                                      exp_pade_order=(10, 11),
                                                      dx_wl=500,
                                                      #dz_wl=0.1,
                                                      storage=PickleStorage()
                                                  ))
pade_field = pade_task.calculate()

#env.terrain = Terrain(lambda x: elev_int_1d(x)-0.00001)
petool_task = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=150e3, dx_wl=500, n_dx_out=1, dz_wl=1)
petool_field = petool_task.calculate()

env.z_max = 3000
petool_task_m = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=150e3, dx_wl=500, n_dx_out=1,
                                      dz_wl=3, n_dz_out=2)
petool_field_m = petool_task_m.calculate()

pade_vis = FieldVisualiser(pade_field, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Pade+NLBC (Proposed)', x_mult=1E-3)
petool_vis = FieldVisualiser(petool_field, env=env, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)
petool_vis_m = FieldVisualiser(petool_field_m, env=env, trans_func=lambda x: x, label='SSF (PETOOL) z_max=3000 m', x_mult=1E-3)

plt = pade_vis.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = petool_vis.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = petool_vis_m.plot2d(min=-100, max=0, show_terrain=True)
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.ylim([0, 300])
plt.tight_layout()
plt.show()

plt = petool_vis.plot_hor_over_terrain(10, petool_vis_m, pade_vis)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.xlim([0.5, 150])
plt.ylim([-100, 0])
plt.grid(True)
plt.tight_layout()
plt.show()