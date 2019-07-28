from rwp.ssf import *
from rwp.antennas import GaussSource3D
from rwp.vis import FieldVisualiser3D

logging.basicConfig(level=logging.DEBUG)

env = EMEnvironment3d(x_min=0, x_max=1000, y_min=-100, y_max=100, z_min=0, z_max=210)
env.knife_edges = [KnifeEdge3d(x1=500, x2=500, y1=-50, y2=50, height=100)]
ant = GaussSource3D(freq_hz=300E6, height=50, ver_beamwidth=30, hor_beamwidth=30, polarz='H')

comp_params = SSF3DPropagatorComputationParameters(dx_wl=1, dy_wl=1, dz_wl=1)
ssf_propagator = SSF3DPropagationTask(src=ant, env=env, comp_params=comp_params)
field = ssf_propagator.calculate()

vis = FieldVisualiser3D(field=field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)))
#plt = vis.plot_xy(z0=50, min_val=-70, max_val=0)
plt = vis.plot_yz(x0=50, min_val=-50, max_val=0)
plt.show()