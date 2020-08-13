from podpac.datalib.terraintiles import TerrainTiles
from podpac import Coordinates, clinspace
from podpac import settings
import matplotlib.pyplot as plt
from rwp.environment import *


settings['DEFAULT_CACHE'] = ['disk']

# create terrain tiles node
node = TerrainTiles(tile_format='geotiff', zoom=11)

#lat = 60.5
#lon = 30
lat, lon = 53.548254, 157.328588
dir = 135
x_grid = np.linspace(0, 200000, 10000)
#coords = geodesic_problem(lat, lon, dir, x_grid)
coords, x_grid = inv_geodesic_problem(60.112502, 29.636637, 60.699130, 31.288706, 1000)
lats = [c[0] for c in coords]
lons = [c[1] for c in coords]

# create coordinates to get tiles
#c = Coordinates([clinspace(10, 11, 2), clinspace(10, 11, 2)], dims=['lat', 'lon'])
c = Coordinates([lats, lons], dims=['lat', 'lon'])

# evaluate node
o = node.eval(c)

eval = np.array([o.data[i, i] for i in range(0, len(x_grid))])

plt.plot(x_grid, eval)
plt.show()

angles = np.arctan((eval[1::] - eval[:-1:]) / (x_grid[1] - x_grid[0])) * 180 / cm.pi
plt.plot(x_grid[:-1:], angles)
plt.show()