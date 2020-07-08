import elevation
import numpy as np
import cmath as cm

#bounds = (-158.4, 21, -157.4, 22)
bounds = (29.636637, 60.112502, 31.288706, 60.699130)
west, south, east, north = bounds
elevation.clip(bounds=bounds, output='lo.tif')
elevation.clean()

from osgeo import gdal
gdal.UseExceptions()

ds = gdal.Open('/home/mikhail/.cache/elevation/SRTM1/lo.tif')
band = ds.GetRasterBand(1)
elevation = band.ReadAsArray()

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
plt.imshow(elevation, cmap='gist_earth', norm=Normalize(0, 1000))
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()

import pyproj
geod = pyproj.Geod(ellps='WGS84')

azimuth1, azimuth2, distance = geod.inv(west, south, east, north)

size = elevation.shape[0]
t = np.array([elevation[i, i] if elevation[i, i] > 0 else 0 for i in np.arange(0, size)])
x_grid = np.linspace(0, distance, size) * 1e-3
plt.plot(x_grid, t)
plt.show()

tt = (t[1::] - t[0:-1:]) / ((x_grid[1] - x_grid[0]) * 1e3)
ttt = np.array([cm.atan(v) * 180 / cm.pi for v in tt])
plt.plot(ttt)
plt.show()
