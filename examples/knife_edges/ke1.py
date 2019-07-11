from rwp.kediffraction import *
from rwp.antennas import *
from rwp.environment import *
from rwp.WPVis import *

logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
#env.knife_edges = [KnifeEdge(range=250, height=50), KnifeEdge(range=750, height=150)]
env.knife_edges = [KnifeEdge(range=150, height=100)]
antenna = Source(wavelength=1)
kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=env, max_range_m=300, max_height_m=300)
field = kdc.calculate()

vis = FieldVisualiser(field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='ke')
plt = vis.plot2d(min=-50, max=0)
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()