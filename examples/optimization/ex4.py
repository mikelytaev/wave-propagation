import elevation
import numpy as np
import cmath as cm
from osgeo import gdal
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pyproj

#bounds = (-158.4, 21, -157.4, 22)
bounds = (29.636637, 60.112502, 31.288706, 60.699130)
west, south, east, north = bounds
#elevation.clip(bounds=bounds, output='lo.tif')
#elevation.clean()

gdal.UseExceptions()

ds = gdal.Open('/home/mikhail/.cache/elevation/SRTM1/lo.tif')
band = ds.GetRasterBand(1)
elevation = band.ReadAsArray()

plt.imshow(elevation, cmap='gist_earth', norm=Normalize(0, 1000))
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()

geod = pyproj.Geod(ellps='WGS84')

azimuth1, azimuth2, distance = geod.inv(west, south, east, north)

long_grid = np.linspace(west, east, elevation.shape[0])
lat_grid = np.linspace(north, south, elevation.shape[1])

elev_int = interp2d(long_grid, lat_grid, elevation.T)

def elev_int_1d(x):
    v = x / distance
    return max(elev_int(west + (east - west) * v, south + (north - south) * v)[0], 0)

size = elevation.shape[0]
t = np.array([elev_int(west + (east - west) * v, south + (north - south) * v) for v in np.linspace(0, 1, size)])
t = np.array([v if v > 0 else 0 for v in t])
x_grid = np.linspace(0, distance, size) * 1e-3
plt.plot(x_grid, t)
plt.show()

tt = (t[1::] - t[0:-1:]) / ((x_grid[1] - x_grid[0]) * 1e3)
ttt = np.array([cm.atan(v) * 180 / cm.pi for v in tt])
plt.plot(abs(ttt))
plt.show()

from rwp.sspade import *
from rwp.vis import *
#from rwp.petool import PETOOLPropagationTask

logging.basicConfig(level=logging.DEBUG)
env = Troposphere()
env.ground_material = FreshWater()
env.z_max = 300
env.terrain = Terrain(elev_int_1d)
profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 50], fill_value="extrapolate")
#env.vegetation = [Impediment(x1=36e3, x2=101e3, height=18, material=CustomMaterial(eps=1.004, sigma=180e-6))]
env.M_profile = lambda x, z: profile1d(z)

ant = GaussAntenna(freq_hz=3000e6, height=30, beam_width=2, eval_angle=0, polarz='H')

pade_task = TroposphericRadioWaveSSPadePropagator(antenna=ant, env=env, max_range_m=distance, comp_params=
                                                  HelmholtzPropagatorComputationalParams(
                                                      terrain_method=TerrainMethod.staircase,
                                                      max_propagation_angle=5,
                                                      modify_grid=True,
                                                      grid_optimizator_abs_threshold=1e-3
                                                  ))
pade_field = pade_task.calculate()

# petool_task = PETOOLPropagationTask(antenna=ant, env=env, two_way=False, max_range_m=distance, dx_wl=400, n_dx_out=1, dz_wl=3)
# petool_field = petool_task.calculate()

pade_vis = FieldVisualiser(pade_field, env=env, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                           label='Pade-[7/8] + NLBC', x_mult=1E-3)
#petool_vis = FieldVisualiser(petool_field, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)

plt = pade_vis.plot2d(min=-70, max=0, show_terrain=True)
plt.title('10log|u|')
plt.xlabel('Range (km)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = pade_vis.plot_hor_over_terrain(2)
plt.xlabel('Range (km)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.show()

# plt = petool_vis.plot2d(min=-70, max=0)
# plt.title('10log|u|')
# plt.xlabel('Range (km)')
# plt.ylabel('Height (m)')
# plt.tight_layout()
# plt.show()