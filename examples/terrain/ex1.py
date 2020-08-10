from podpac.datalib.terraintiles import TerrainTiles
from podpac import Coordinates, clinspace
import matplotlib.pyplot as plt


# create terrain tiles node
node = TerrainTiles(tile_format='geotiff', zoom=12)

# create coordinates to get tiles
c = Coordinates([clinspace(59, 59.1, 1000), clinspace(31, 31.1, 1000)], dims=['lat', 'lon'])

# evaluate node
o = node.eval(c)

# plot the elevation
fig = plt.figure(dpi=90)
o.plot(vmin=0, cmap='terrain')
plt.show()