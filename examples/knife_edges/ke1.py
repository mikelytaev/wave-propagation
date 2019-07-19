from rwp.kediffraction import *
from rwp.antennas import *
from rwp.environment import *
from rwp.WPVis import *

env = Troposphere(flat=True)
env.z_max = 500
#env.knife_edges = [KnifeEdge(range=500, height=50)]
antenna = Source(wavelength=1, height_m=50)

kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=env, max_range_m=5000)
field = kdc.calculate()

vis = FieldVisualiser(field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='ke')
plt = vis.plot2d(min=-40, max=0)
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.show()

plt = vis.plot_hor(50)
plt.xlabel('Range (m)')
plt.ylabel('10log|u| (dB)')
plt.show()