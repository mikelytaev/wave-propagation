from rwp.kediffraction import *
from rwp.antennas import *
from rwp.environment import *
from rwp.WPVis import *

env = Troposphere(flat=True)
#env.knife_edges = [KnifeEdge(range=500, height=150)]
antenna = Source(wavelength=1)
kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=env, max_range_m=300, max_height_m=300)
field = kdc.calculate()

vis = FieldVisualiser(field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='ke')
plt = vis.plot2d(min=-30, max=0)
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()