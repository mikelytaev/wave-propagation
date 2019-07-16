from rwp.kediffraction import *
from rwp.antennas import *
from rwp.environment import *
from rwp.WPVis import *

logging.basicConfig(level=logging.DEBUG)
env = Troposphere(flat=True)
#env.knife_edges = [KnifeEdge(range=250, height=50), KnifeEdge(range=750, height=150)]
env.knife_edges = [KnifeEdge(range=200, height=50)]#, KnifeEdge(range=800, height=50)]
#antenna = Source(wavelength=1, height_m=50)
antenna = GaussAntenna(wavelength=1, height=50, beam_width=15, eval_angle=0, polarz='H')
kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=env, max_range_m=300, max_height_m=150)
field = kdc.calculate()

# g = kdc.ts._int_green_f(1e-5, 100, 100, kdc.ts.p_computational_grid)
# plt.plot(kdc.ts.p_computational_grid, abs(g))
# plt.show()
#
# rhs = kdc.ts.debug_data.rhs
# plt.plot(kdc.ts.p_computational_grid/kdc.ts.k0, np.real(rhs))
# plt.show()
#
# plt.plot(kdc.ts.p_computational_grid/kdc.ts.k0, np.imag(rhs))
# plt.show()
#
# phi = kdc.ts.debug_data.phi
# plt.plot(kdc.ts.p_computational_grid/kdc.ts.k0, np.real(phi))
# plt.show()
#
# phi = kdc.ts.debug_data.phi
# plt.plot(kdc.ts.p_computational_grid/kdc.ts.k0, np.imag(phi))
# plt.show()


vis = FieldVisualiser(field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)), label='ke')
plt = vis.plot2d(min=-40, max=0)
plt.xlabel('Range (m)')
plt.ylabel('Height (m)')
plt.tight_layout()
plt.show()

plt = vis.plot_hor(50)
plt.xlabel('Range (m)')
plt.ylabel('10log|u| (dB)')
plt.tight_layout()
plt.show()
